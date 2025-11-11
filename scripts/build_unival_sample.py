#!/usr/bin/env python
from pathlib import Path

import json
from datasets import load_dataset

OUTPUT_DIR = Path("data/univl_samples")
IMAGE_DIR = OUTPUT_DIR / "images"
MANIFEST = OUTPUT_DIR / "example.jsonl"

CAPTION_SPLIT = dict(dataset="cifar10", split="train", select=500)
VQA_SPLIT = dict(dataset="cifar10", split="test", select=500)


def process_caption_examples() -> list[dict[str, str]]:
    ds = load_dataset(
        CAPTION_SPLIT["dataset"],
        split=f"{CAPTION_SPLIT['split']}[:{CAPTION_SPLIT['select']}]",
    )
    entries: list[dict[str, str]] = []
    for idx, row in enumerate(ds):
        image = row["img"].convert("RGB") if "img" in row else row["image"].convert("RGB")
        label = row["label"]
        label_name = ds.features["label"].int2str(label)
        text = f"A photograph of a {label_name}."
        filename = IMAGE_DIR / f"caption_{idx:03d}.jpg"
        filename.parent.mkdir(parents=True, exist_ok=True)
        image.save(filename)
        entries.append(
            {
                "image": str(filename.resolve()),
                "prompt": "<CAPTION>",
                "text": text,
            }
        )
    return entries


def process_vqa_examples() -> list[dict[str, str]]:
    ds = load_dataset(
        VQA_SPLIT["dataset"],
        split=f"{VQA_SPLIT['split']}[:{VQA_SPLIT['select']}]",
    )
    entries: list[dict[str, str]] = []
    for idx, row in enumerate(ds):
        image = row["img"].convert("RGB") if "img" in row else row["image"].convert("RGB")
        label = row["label"]
        label_name = ds.features["label"].int2str(label)
        filename = IMAGE_DIR / f"vqa_{idx:03d}.jpg"
        filename.parent.mkdir(parents=True, exist_ok=True)
        image.save(filename)
        entries.append(
            {
                "image": str(filename.resolve()),
                "prompt": "<QUESTION> What object is shown in this image?",
                "text": label_name,
            }
        )
    return entries


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_entries = process_caption_examples() + process_vqa_examples()
    with MANIFEST.open("w", encoding="utf-8") as fh:
        for item in all_entries:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_entries)} samples to {MANIFEST}")


if __name__ == "__main__":
    main()
