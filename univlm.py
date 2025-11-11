from __future__ import annotations

from models.univlm_runtime import (
    AdapterLayer,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_VLM_MODELS,
    UniVLConfig,
    UniVLModel,
    answer_question,
    caption_with_pretrained_vlm,
    generate_caption,
    load_model_and_tokenizer,
    resolve_device,
    train_univl,
    validate_image,
    vqa_with_pretrained_vlm,
)
from scripts.univlm_cli import main

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
    "main",
    "resolve_device",
    "train_univl",
    "validate_image",
    "vqa_with_pretrained_vlm",
]


if __name__ == "__main__":
    main()
