from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from models.univlm_runtime import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    UniVLConfig,
    resolve_device,
)
from scripts.univlm_interfaces import (
    DEFAULT_VLM_MODELS,
    pretrained_caption_interface,
    pretrained_vqa_interface,
    univlm_caption_interface,
    univlm_training_interface,
    univlm_vqa_interface,
)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum tokens to generate.")
    parser.add_argument("--num-beams", type=int, default=3, help="Beam search width.")


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniVLM CLI for training and downstream tasks.")
    parser.add_argument("--vision-model", default=UniVLConfig.vision_model, help="Vision backbone model id.")
    parser.add_argument("--text-model", default=UniVLConfig.text_model, help="Text seq2seq model id.")
    parser.add_argument("--num-queries", type=int, default=UniVLConfig.num_queries, help="Number of learnable queries.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Torch device override.")
    parser.add_argument("--checkpoint-dir", help="Directory containing UniVLM checkpoints.")
    parser.add_argument("--pretrained-text-lora", help="Optional PEFT LoRA checkpoint to load.")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_DIR),
        help="Folder containing training data and manifests (default: ./data).",
    )
    parser.add_argument(
        "--models-root",
        default=str(DEFAULT_MODELS_DIR),
        help="Folder to stage checkpoints (default: ./models).",
    )
    parser.add_argument(
        "--vlm-model",
        help=(
            "Use a pretrained vision-language model (e.g., Salesforce/blip-image-captioning-base) "
            "instead of UniVL for caption/VQA inference. Pass 'auto' to use a sensible default."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    cap_parser = subparsers.add_parser("caption", help="Generate a caption for an image.")
    cap_parser.add_argument("image_path", help="Path to the image file.")
    cap_parser.add_argument(
        "--prompt",
        default="<CAPTION>",
        help="Prompt token/text for caption generation (default: <CAPTION>).",
    )
    _add_generation_args(cap_parser)

    vqa_parser = subparsers.add_parser("vqa", help="Answer a question about an image.")
    vqa_parser.add_argument("image_path", help="Path to the image file.")
    vqa_parser.add_argument("question", help="Question to ask about the image.")
    vqa_parser.add_argument(
        "--prompt-prefix",
        default="<QUESTION> ",
        help="Prefix included before the question (default: <QUESTION> ).",
    )
    _add_generation_args(vqa_parser)

    train_parser = subparsers.add_parser("train", help="Fine-tune UniVLM using LoRA/adapters.")
    train_parser.add_argument("manifest", help="Path (relative to data-root) to a JSONL manifest.")
    train_parser.add_argument(
        "--output-dir",
        default="univlm-lora",
        help="Directory name (relative to models-root) for checkpoints.",
    )
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5, help="Optimizer learning rate.")
    train_parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay coefficient.")
    train_parser.add_argument("--max-input-length", type=int, default=64, help="Prompt token limit.")
    train_parser.add_argument("--max-target-length", type=int, default=64, help="Target token limit.")
    train_parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold applied per batch.",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_vlm_model(command: str, requested: str | None) -> str | None:
    if requested is None:
        return None
    if requested.lower() == "auto":
        return DEFAULT_VLM_MODELS.get(command)
    return requested


def _maybe_relative(path_str: str, root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    device = resolve_device(args.device)

    data_root = Path(args.data_root).expanduser()
    models_root = Path(args.models_root).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir).expanduser() if args.checkpoint_dir else DEFAULT_CHECKPOINT_DIR

    vlm_model = _resolve_vlm_model(args.command, args.vlm_model)
    if vlm_model and args.command == "train":
        raise SystemExit("--vlm-model cannot be combined with the train command.")

    config = UniVLConfig(
        vision_model=args.vision_model,
        text_model=args.text_model,
        num_queries=args.num_queries,
        pretrained_text_lora=args.pretrained_text_lora,
    )

    if vlm_model and args.command in {"caption", "vqa"}:
        if args.command == "caption":
            prompt_text = (args.prompt or "").strip() or None
            text = pretrained_caption_interface(
                vlm_model,
                Path(args.image_path),
                prompt=prompt_text,
                generation_kwargs={
                    "max_new_tokens": args.max_new_tokens,
                    "num_beams": args.num_beams,
                },
                device=device,
            )
            print(f"Caption: {text or '[empty output]'}")
            return
        answer = pretrained_vqa_interface(
            vlm_model,
            Path(args.image_path),
            args.question,
            generation_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
            },
            device=device,
        )
        print(f"Answer: {answer or '[empty output]'}")
        return

    if args.command == "caption":
        caption = univlm_caption_interface(
            Path(args.image_path),
            prompt=args.prompt,
            generation_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
            },
            config=config,
            checkpoint_dir=checkpoint_dir if checkpoint_dir.exists() else None,
            device=device,
        )
        print(f"Caption: {caption or '[empty output]'}")
    elif args.command == "vqa":
        answer = univlm_vqa_interface(
            Path(args.image_path),
            args.question,
            prompt_prefix=args.prompt_prefix,
            generation_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
            },
            config=config,
            checkpoint_dir=checkpoint_dir if checkpoint_dir.exists() else None,
            device=device,
        )
        print(f"Answer: {answer or '[empty output]'}")
    elif args.command == "train":
        manifest_path = _maybe_relative(args.manifest, data_root)
        output_dir = _maybe_relative(args.output_dir, models_root)
        result_dir = univlm_training_interface(
            manifest_path,
            config=config,
            checkpoint_dir=checkpoint_dir if checkpoint_dir.exists() else None,
            device=device,
            output_dir=output_dir,
            train_kwargs={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length,
                "max_grad_norm": args.max_grad_norm,
            },
        )
        print(f"Training complete. Checkpoints saved to {result_dir}")
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
