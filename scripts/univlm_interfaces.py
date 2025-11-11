from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.univlm_runtime import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_VLM_MODELS,
    UniVLConfig,
    answer_question,
    caption_with_pretrained_vlm,
    generate_caption,
    load_model_and_tokenizer,
    train_univl,
    validate_image,
    vqa_with_pretrained_vlm,
)


def _runtime_from_checkpoint(
    config: UniVLConfig,
    *,
    checkpoint_dir: Path | None,
    device: torch.device,
) -> tuple[Any, Any, Any]:
    model, tokenizer, image_processor = load_model_and_tokenizer(
        config,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    return model, tokenizer, image_processor


def univlm_caption_interface(
    image_path: Path,
    *,
    prompt: str,
    generation_kwargs: dict[str, Any],
    config: UniVLConfig,
    checkpoint_dir: Path | None,
    device: torch.device,
) -> str:
    image = validate_image(image_path)
    model, tokenizer, image_processor = _runtime_from_checkpoint(
        config,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    return generate_caption(
        model,
        tokenizer,
        image_processor,
        image,
        prompt=prompt,
        **generation_kwargs,
    )


def univlm_vqa_interface(
    image_path: Path,
    question: str,
    *,
    prompt_prefix: str,
    generation_kwargs: dict[str, Any],
    config: UniVLConfig,
    checkpoint_dir: Path | None,
    device: torch.device,
) -> str:
    image = validate_image(image_path)
    model, tokenizer, image_processor = _runtime_from_checkpoint(
        config,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    return answer_question(
        model,
        tokenizer,
        image_processor,
        image,
        question=question,
        prompt_prefix=prompt_prefix,
        **generation_kwargs,
    )


def univlm_training_interface(
    manifest_path: Path,
    *,
    config: UniVLConfig,
    checkpoint_dir: Path | None,
    device: torch.device,
    output_dir: Path,
    train_kwargs: dict[str, Any],
) -> Path:
    manifest = manifest_path.expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    runtime_checkpoint = checkpoint_dir if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
    model, tokenizer, image_processor = _runtime_from_checkpoint(
        config,
        checkpoint_dir=runtime_checkpoint if runtime_checkpoint.exists() else None,
        device=device,
    )
    return train_univl(
        model,
        tokenizer,
        image_processor,
        manifest=manifest,
        **train_kwargs,
        output_dir=output_dir,
    )


def pretrained_caption_interface(
    model_name: str,
    image_path: Path,
    *,
    prompt: str | None,
    generation_kwargs: dict[str, Any],
    device: torch.device,
) -> str:
    image = validate_image(image_path)
    return caption_with_pretrained_vlm(
        model_name,
        image,
        prompt=prompt,
        device=device,
        **generation_kwargs,
    )


def pretrained_vqa_interface(
    model_name: str,
    image_path: Path,
    question: str,
    *,
    generation_kwargs: dict[str, Any],
    device: torch.device,
) -> str:
    image = validate_image(image_path)
    return vqa_with_pretrained_vlm(
        model_name,
        image,
        question=question,
        device=device,
        **generation_kwargs,
    )


__all__ = [
    "DEFAULT_VLM_MODELS",
    "pretrained_caption_interface",
    "pretrained_vqa_interface",
    "univlm_caption_interface",
    "univlm_training_interface",
    "univlm_vqa_interface",
]
