# UniVLM

UniVLM is a lightweight vision-language method that I designed to explore how far minimalist cross-modal adapters can go when paired with strong pretrained components. A frozen ViT encoder extracts visual tokens, a compact Q-former projects them through a learnable query bank, and a frozen seq2seq LM (FLAN-T5 by default) decodes textual outputs. Only the query projection, adapters, and optional LoRA layers are trainable, which keeps fine-tuning fast while still supporting captioning, VQA, OCR-style prompts, and other downstream reasoning tasks.

## Project layout

```
├── data/                # Datasets, manifests, and helper assets
├── models/              # Runtime module + fine-tuned checkpoints
├── scripts/             # CLI + automation utilities
├── adapters/            # Optional adapter weights
├── univlm.py            # Thin entrypoint that mirrors florence-2
```

All helper scripts now live under `scripts/`, datasets belong in `data/`, and checkpoints produced by fine-tuning default to `models/` so the structure mirrors the Florence-2 project.

## Getting started

1. Create an environment with Python 3.10+ and install the dependencies you need (`torch`, `transformers`, `peft`, `datasets`, `einops`, `timm`, `Pillow`).
2. Optionally generate a toy manifest with `python scripts/build_unival_sample.py`, which emits images and a JSONL file under `data/univl_samples/` (see the dedicated section below).
3. Invoke the CLI via the root entrypoint:
   ```bash
   python univlm.py --help
   ```

## Command line interface

The CLI (implemented in `scripts/univlm_cli.py`) exposes training and inference flows that understand the new directory layout.

### Captioning
```bash
python univlm.py caption data/example.jpg --prompt "<CAPTION>" --max-new-tokens 72
```

### Visual question answering
```bash
python univlm.py vqa data/example.jpg "What vehicle is shown?" --prompt-prefix "<QUESTION> "
```

### Training
```bash
python univlm.py train univl_samples/example.jsonl --output-dir univlm-latest --epochs 3 --batch-size 4
```
- Manifests are resolved relative to `data/` by default (override with `--data-root`).
- Outputs land in `models/<output-dir>` unless you point elsewhere with `--models-root`.
- Pass `--vlm-model auto` to borrow a stock BLIP model for quick experiments without loading UniVLM weights.

## Programmatic interfaces

`scripts/univlm_interfaces.py` wraps the runtime so you can embed UniVLM inside notebooks or services without copying CLI logic:
- `univlm_caption_interface` and `univlm_vqa_interface` load checkpoints, preprocess inputs, and return decoded strings.
- `univlm_training_interface` orchestrates manifest ingestion plus checkpoint export, guaranteeing that artifacts stay under `models/`.
- `pretrained_caption_interface` / `pretrained_vqa_interface` expose BLIP fallbacks for sanity checks.

## Method overview

1. **Frozen ViT vision encoder** – extracts patch embeddings that remain stable during fine-tuning.
2. **Learnable query bank + Q-former** – a handful of queries attend over the visual tokens, producing compact multimodal representations.
3. **Adapter-enhanced decoder** – the projected queries are concatenated with text embeddings before being decoded by a frozen FLAN-T5. Shallow adapters plus optional LoRA ranks give just enough capacity to specialize without catastrophic forgetting.
4. **Minimal training loop** – manifests specify `{image, prompt, text}` triples. Prompts act as control tokens (`<CAPTION>`, `<QUESTION>`, etc.) so a single checkpoint can serve multiple downstream instructions.

Because only adapters, LoRA layers, and the projection stack receive gradients, UniVLM can be fine-tuned on consumer GPUs with modest datasets. Saved checkpoints contain the PEFT weights, tokenizer, and image processor so downstream tasks reuse the exact preprocessing recipe.

## Relationship to BLIP-2 and Florence-2

UniVLM borrows the *architecture* intuition from BLIP-2 (frozen ViT + lightweight Q-former + frozen language model) and the *promptable interface* from Florence-2 (task-conditioned special tokens such as `<CAPTION>`, `<QUESTION>`, `<OCR>`). The result is a hybrid recipe:

- BLIP-2–style query adapters keep the trainable parameter count tiny.
- Florence-2–style control tokens let one checkpoint pivot between captioning, VQA, OCR, and detection-style reasoning without separate heads.
- Optional fallbacks to full Florence / BLIP checkpoints make it easy to compare gains or bootstrap datasets.

If you already have BLIP-2 or Florence-2 weights, point `--vlm-model` to those model IDs (or local snapshots) for zero-shot evaluation while you fine-tune UniVLM-specific adapters.

## Using pretrained models inside UniVLM

There are two ways to reuse pretrained assets:

1. **LoRA / adapter checkpoints** – place a previous UniVLM run under `models/<name>` and pass `--checkpoint-dir models/<name>` so the CLI loads `univl_model.pt`, tokenizer, image processor, and optional PEFT weights before continuing training or running inference.
2. **External VLMs** – provide `--vlm-model auto` (or a specific Hugging Face repo ID like `Salesforce/blip-image-captioning-base`) when running `caption` or `vqa`. The CLI will switch to the BLIP runtime in `scripts/univlm_interfaces.py`, letting you compare against strong baselines or use them as teacher models.

Because the runtime module exposes `load_model_and_tokenizer` and the interface layer wraps all preprocessing, you can also import UniVLM into notebooks and call `univlm_caption_interface(...)` or `pretrained_caption_interface(...)` directly for more customized workflows.

## Ideas to push the method further

- **Dual-stage prompting** – prepend Florence-style task tags *and* learn per-task query embeddings so the Q-former itself adapts to OCR vs. caption reasoning.
- **Self-distillation from Florence-2** – run Florence-2 in teacher mode (via `--vlm-model`) to auto-label niche datasets, then fine-tune UniVLM on those pseudo pairs to close the gap while staying lightweight.
- **Dynamic adapter routing** – add a small controller that selects which adapter blocks or LoRA ranks to activate per prompt, giving more capacity without increasing inference cost for easy tasks.
- **Contrastive grounding loss** – mix the generative objective with a BLIP-style contrastive loss on image/text pairs to sharpen visual grounding, especially for retrieval-style downstream tasks.

## Sample data generation (`scripts/build_unival_sample.py`)

Use the helper script when you need a quick manifest for experimentation or debugging:

```bash
python scripts/build_unival_sample.py
```

This downloads small CIFAR-10 splits via `datasets`, saves JPEGs to `data/univl_samples/images/`, and writes `data/univl_samples/example.jsonl` containing `{image, prompt, text}` entries for both captioning and VQA-style prompts.

- Point the `train` CLI command at this manifest: `python univlm.py train univl_samples/example.jsonl --output-dir univlm-samples`
- Regenerate anytime to refresh with a different random subset by tweaking the constants at the top of the script.

## Saving and loading checkpoints

- The runtime module lives in `models/univlm_runtime.py` and exposes `load_model_and_tokenizer`, `train_univl`, `generate_caption`, and other helpers.
- Every training run emits:
  - `univl_model.pt` (state dict + configuration)
  - `tokenizer/` and `image_processor/` folders
  - `text_lora/` when LoRA is enabled
- Place additional custom weights anywhere under `models/` and point the CLI at them via `--checkpoint-dir`.

With this structure, UniVLM mirrors the Florence-2 project: scripts are centralized, datasets live under `data/`, fine-tuned artifacts accumulate under `models/`, and `univlm.py` is a thin façade over the CLI so it can be used both as a module and as an executable.
