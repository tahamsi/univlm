from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
    PreTrainedTokenizerBase,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CHECKPOINT_DIR = DEFAULT_MODELS_DIR / "checkpoints"


@dataclass
class UniVLConfig:
    vision_model: str = "google/vit-base-patch16-224"
    text_model: str = "google/flan-t5-large"
    num_queries: int = 32
    qformer_hidden_size: int = 768
    qformer_layers: int = 6
    qformer_heads: int = 8
    qformer_ffn: int = 2048
    special_tokens: Sequence[str] = (
        "<CAPTION>",
        "<QUESTION>",
        "<ANSWER>",
        "<OD>",
        "<OCR>",
    )
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Sequence[str] = ("q", "k", "v", "o")
    use_qformer_adapter: bool = True
    use_decoder_adapter: bool = True
    adapter_dim: int = 256
    pretrained_text_lora: str | None = None
    pretrained_qformer_adapter: str | None = None


DEFAULT_VLM_MODELS = {
    "caption": "Salesforce/blip-image-captioning-base",
    "vqa": "Salesforce/blip-vqa-base",
}


class AdapterLayer(nn.Module):
    def __init__(self, hidden_size: int, adapter_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.activation(self.down(x)))


class UniVLModel(nn.Module):
    """A lightweight uni-modal VL model that fuses ViT vision features with a seq2seq LM."""

    def __init__(self, config: UniVLConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Vision encoder (frozen)
        self.vision_encoder = AutoModel.from_pretrained(config.vision_model, trust_remote_code=False)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        vision_hidden = self.vision_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_hidden, config.qformer_hidden_size)

        # 2. Q-Former style transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.qformer_hidden_size,
            nhead=config.qformer_heads,
            dim_feedforward=config.qformer_ffn,
            batch_first=True,
        )
        self.qformer = nn.TransformerEncoder(encoder_layer, num_layers=config.qformer_layers)
        self.query_embed = nn.Parameter(
            torch.randn(config.num_queries, config.qformer_hidden_size)
        )

        # 3. Text decoder (seq2seq LM, frozen backbone)
        self.text_model = AutoModelForSeq2SeqLM.from_pretrained(config.text_model, trust_remote_code=False)
        for param in self.text_model.parameters():
            param.requires_grad = False
        text_hidden = self.text_model.config.d_model
        self.query_to_text = nn.Linear(config.qformer_hidden_size, text_hidden)

        self.qformer_adapter = (
            AdapterLayer(config.qformer_hidden_size, config.adapter_dim)
            if config.use_qformer_adapter
            else nn.Identity()
        )
        self.decoder_adapter = (
            AdapterLayer(text_hidden, config.adapter_dim)
            if config.use_decoder_adapter
            else nn.Identity()
        )

        if config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError(
                    "LoRA support requires the 'peft' package. Install with `pip install peft`."
                ) from exc

            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=list(config.lora_target_modules),
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            self.text_model = get_peft_model(self.text_model, lora_cfg)
        if config.pretrained_text_lora:
            self.load_text_lora(config.pretrained_text_lora)
        self.text_model.print_trainable_parameters()

        self.to(self.device)

    def resize_text_embeddings(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Resize token embeddings when tokenizer gains new special tokens."""
        self.text_model.resize_token_embeddings(len(tokenizer))

    def load_text_lora(self, path: str) -> None:
        path = Path(path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"LoRA checkpoint not found: {path}")
        if not hasattr(self.text_model, "load_adapter"):
            raise RuntimeError("Text model is not a PEFT model; cannot load LoRA weights.")
        self.text_model.load_adapter(path, adapter_name="pretrained_lora")
        self.text_model.set_adapter("pretrained_lora")

    def image_to_queries(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image tensors into projected query embeddings."""
        vision_outputs = self.vision_encoder(pixel_values=pixel_values.to(self.device))
        if hasattr(vision_outputs, "last_hidden_state"):
            image_embeds = vision_outputs.last_hidden_state
        else:
            image_embeds = vision_outputs[0]
        image_embeds = self.vision_proj(image_embeds)

        batch_queries = self.query_embed.unsqueeze(0).expand(image_embeds.size(0), -1, -1)
        seq = torch.cat([batch_queries, image_embeds], dim=1)
        qformer_out = self.qformer(seq)
        query_states = qformer_out[:, : self.config.num_queries, :]
        query_states = self.qformer_adapter(query_states)
        return self.query_to_text(query_states)

    def _prepare_encoder_inputs(
        self,
        image_queries: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide either input_ids or inputs_embeds, not both.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be supplied.")
            inputs_embeds = self.text_model.get_encoder().embed_tokens(input_ids.to(self.device))
        else:
            inputs_embeds = inputs_embeds.to(self.device)

        combined_embeds = torch.cat([image_queries, inputs_embeds], dim=1)
        combined_embeds = self.decoder_adapter(combined_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            query_mask = torch.ones(
                (attention_mask.size(0), image_queries.size(1)),
                device=self.device,
                dtype=attention_mask.dtype,
            )
            combined_attention = torch.cat([query_mask, attention_mask], dim=1)
        else:
            combined_attention = None

        return combined_embeds, combined_attention

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_queries = self.image_to_queries(pixel_values)
        encoder_inputs, combined_attention = self._prepare_encoder_inputs(
            image_queries,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        return self.text_model(
            inputs_embeds=encoder_inputs,
            attention_mask=combined_attention,
            labels=labels.to(self.device) if labels is not None else None,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        self.eval()
        image_queries = self.image_to_queries(pixel_values)
        encoder_inputs, combined_attention = self._prepare_encoder_inputs(
            image_queries,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return self.text_model.generate(
            inputs_embeds=encoder_inputs,
            attention_mask=combined_attention,
            **generate_kwargs,
        )


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def validate_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def load_model_and_tokenizer(
    config: UniVLConfig | None,
    device: torch.device | None = None,
    *,
    checkpoint_dir: Path | None = None,
) -> tuple[UniVLModel, AutoTokenizer, AutoImageProcessor]:
    checkpoint_state: dict | None = None
    if checkpoint_dir:
        checkpoint_path = checkpoint_dir / "univl_model.pt"
        if checkpoint_path.is_file():
            checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
            cfg_dict = checkpoint_state.get("config")
            if cfg_dict:
                config = UniVLConfig(**cfg_dict)
        if config is None:
            raise ValueError("Checkpoint does not include configuration data.")
    if config is None:
        raise ValueError("Configuration must be provided when no checkpoint is supplied.")

    tokenizer_dir = checkpoint_dir / "tokenizer" if checkpoint_dir else None
    if tokenizer_dir and tokenizer_dir.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.text_model, use_fast=True)
        tokenizer.add_special_tokens({"additional_special_tokens": list(config.special_tokens)})

    model = UniVLModel(config, device=device)
    model.resize_text_embeddings(tokenizer)

    processor_dir = checkpoint_dir / "image_processor" if checkpoint_dir else None
    if processor_dir and processor_dir.is_dir():
        image_processor = AutoImageProcessor.from_pretrained(processor_dir)
    else:
        image_processor = AutoImageProcessor.from_pretrained(config.vision_model)

    if checkpoint_state:
        model.load_state_dict(checkpoint_state["state_dict"], strict=False)

    return model, tokenizer, image_processor


def generate_caption(
    model: UniVLModel,
    tokenizer: PreTrainedTokenizerBase,
    image_processor: AutoImageProcessor,
    image: Image.Image,
    *,
    prompt: str,
    max_new_tokens: int = 64,
    num_beams: int = 3,
) -> str:
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(model.device)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        pixel_values=pixel_values,
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def answer_question(
    model: UniVLModel,
    tokenizer: PreTrainedTokenizerBase,
    image_processor: AutoImageProcessor,
    image: Image.Image,
    question: str,
    *,
    prompt_prefix: str = "<QUESTION> ",
    max_new_tokens: int = 32,
    num_beams: int = 3,
) -> str:
    prompt = f"{prompt_prefix}{question}"
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(model.device)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        pixel_values=pixel_values,
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


@torch.no_grad()
def caption_with_pretrained_vlm(
    model_id: str,
    image: Image.Image,
    *,
    prompt: str | None,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> str:
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    processor_kwargs = {"images": image, "return_tensors": "pt"}
    if prompt:
        processor_kwargs["text"] = prompt
    inputs = processor(**processor_kwargs)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return processor.batch_decode(output, skip_special_tokens=True)[0].strip()


@torch.no_grad()
def vqa_with_pretrained_vlm(
    model_id: str,
    image: Image.Image,
    question: str,
    *,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> str:
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id).to(device)
    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return processor.batch_decode(output, skip_special_tokens=True)[0].strip()


class UniVLDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: AutoImageProcessor,
        *,
        max_input_length: int = 64,
        max_target_length: int = 64,
    ) -> None:
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        with manifest_path.open("r", encoding="utf-8") as fh:
            self.samples = [json.loads(line) for line in fh if line.strip()]
        if not self.samples:
            raise ValueError(f"No samples found in manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_path = Path(sample["image"]).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        prompt = sample.get("prompt", "<CAPTION>")
        target_text = sample.get("text")
        if target_text is None:
            raise ValueError("Each manifest entry must include a 'text' field for supervision.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        labels = self.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_target_length,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }

    def collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        inputs = self.tokenizer.pad(
            [
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                }
                for item in batch
            ],
            return_tensors="pt",
        )
        labels = self.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in batch],
            return_tensors="pt",
        )["input_ids"]

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }


def train_univl(
    model: UniVLModel,
    tokenizer: PreTrainedTokenizerBase,
    image_processor: AutoImageProcessor,
    *,
    manifest: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_input_length: int,
    max_target_length: int,
    output_dir: Path,
    max_grad_norm: float,
) -> Path:
    dataset = UniVLDataset(
        manifest,
        tokenizer,
        image_processor,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Ensure LoRA or adapters are enabled.")

    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            batch = {key: tensor.to(model.device) for key, tensor in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * batch["pixel_values"].size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": model.config.__dict__,
    }
    torch.save(checkpoint, output_dir / "univl_model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    image_processor.save_pretrained(output_dir / "image_processor")
    if hasattr(model.text_model, "save_pretrained"):
        try:
            model.text_model.save_pretrained(output_dir / "text_lora")
        except Exception:
            pass
    return output_dir


__all__ = [
    "AdapterLayer",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_DATA_DIR",
    "DEFAULT_MODELS_DIR",
    "DEFAULT_VLM_MODELS",
    "UniVLConfig",
    "UniVLModel",
    "answer_question",
    "caption_with_pretrained_vlm",
    "generate_caption",
    "load_model_and_tokenizer",
    "resolve_device",
    "train_univl",
    "validate_image",
    "vqa_with_pretrained_vlm",
]
