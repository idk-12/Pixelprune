# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VLMEvalKit is a modular evaluation framework for vision-language models (VLMs). It supports 200+ models and 100+ benchmarks, with both local and API-based evaluation.

## Setup & Installation

```bash
conda activate vllm  # required environment for running evaluations
pip install -e .
# Copy and populate with API keys (OpenAI, Gemini, etc.)
cp .env.example .env
```

Dataset files are auto-downloaded to `$LMUData` (defaults to `~/LMUData`).

## Running Evaluations

**Single model, single dataset:**
```bash
python run.py --data MMBench_DEV_EN --model llava_v1_5_7b
```

**Multiple models/datasets:**
```bash
python run.py --data MMBench_DEV_EN MMMU_DEV_VAL --model llava_v1_5_7b qwen_chat
```

**Inference only (skip judge evaluation):**
```bash
python run.py --data MMBench_DEV_EN --model llava_v1_5_7b --mode infer
```

**Multi-GPU distributed evaluation:**
```bash
torchrun --nproc-per-node=4 run.py --data MMBench_DEV_EN --model llava_v1_5_7b
```

**List available models and datasets:**
```bash
vlmutil mlist        # list models
vlmutil dlist        # list datasets
```

## Code Style

- Line length: **99 characters** (flake8/YAPF/isort all use 99)
- Formatter: YAPF (PEP8-based), import sorter: isort
- Pre-commit hooks enforce style:
  ```bash
  pip install pre-commit && pre-commit install
  pre-commit run --all-files
  ```
- `vlmeval/config.py` is excluded from linting (large auto-generated registry)

## Architecture

### Three Core Components

**1. Models (`vlmeval/vlm/`)**
- Each model is a class inheriting `BaseModel` (local) or `BaseAPI` (API-based)
- Required method: `generate_inner(self, message, dataset=None) -> str`
- Optional overrides: `use_custom_prompt()`, `build_prompt()`, `chat_inner()` (for multi-turn)
- All models registered in `vlmeval/config.py` via `supported_VLM` dict using `functools.partial`

**2. Datasets (`vlmeval/dataset/`)**
- Each benchmark inherits `ImageBaseDataset`, `VideoBaseDataset`, or `TextBaseDataset`
- Required class attributes: `TYPE` (e.g. `'MCQ'`, `'VQA'`, `'Y/N'`), `DATASET_URL`, `DATASET_MD5`
- Required methods: `build_prompt(line) -> list` (returns multimodal message), `evaluate(eval_file, **kwargs) -> dict`
- Required class method: `supported_datasets() -> list[str]`
- Dataset lookup via `build_dataset(name)` which iterates `DATASET_CLASSES`
- Dataset files are TSV format; images stored as base64 in `image` column

**3. Evaluation Pipeline (`run.py`, `vlmeval/inference*.py`)**
- `run.py`: Top-level orchestrator; parses args, sets up distributed env, calls inference then evaluate
- `inference.py`: Single-node image inference
- `inference_video.py`: Video benchmark inference (with frame sampling)
- `inference_mt.py`: Multi-turn conversation inference
- `inference_api.py`: API-based model inference with async/parallel support
- Results saved as `.pkl` checkpoints during inference, then exported to CSV/Excel

### Key Files

| File | Purpose |
|---|---|
| `vlmeval/config.py` | Registry of all 200+ models (`supported_VLM` dict) |
| `vlmeval/dataset/__init__.py` | Dataset registry; `IMAGE_DATASET`, `VIDEO_DATASET` lists |
| `vlmeval/vlm/__init__.py` | VLM imports |
| `vlmeval/smp/` | Shared utilities: file I/O, logging, distributed helpers |
| `vlmeval/api/` | API client implementations (OpenAI, Gemini, Claude, etc.) |

## Adding a New Model

1. Create `vlmeval/vlm/mymodel.py` with class inheriting `BaseModel` or `BaseAPI`
2. Implement `generate_inner(self, message, dataset=None) -> str`
3. Export from `vlmeval/vlm/__init__.py`
4. Register in `vlmeval/config.py` within an appropriate series dict:
   ```python
   mymodel_series = {
       'mymodel_7b': partial(vlm.MyModel, model_path='org/mymodel-7b'),
   }
   ```
   Then add to `supported_VLM`.

## Adding a New Dataset/Benchmark

1. Create `vlmeval/dataset/mybenchmark.py` inheriting `ImageBaseDataset`
2. Define `TYPE`, `DATASET_URL`, `DATASET_MD5`, `DEFAULT_JUDGE`
3. Implement `build_prompt(line)` and `evaluate(eval_file, **judge_kwargs)`
4. Implement `supported_datasets()` classmethod
5. Import in `vlmeval/dataset/__init__.py` and append to `IMAGE_DATASET` (or `VIDEO_DATASET`)

See `docs/en/Development.md` for detailed examples.

## Environment Variables

- `LMUData`: Directory for downloaded dataset files (default: `~/LMUData`)
- `CUDA_VISIBLE_DEVICES`: GPU selection
- API keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.) loaded from `.env` at startup via `load_env()`
