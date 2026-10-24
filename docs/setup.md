# Setup details

## Python environment
- Run `uv sync`. It installs Python 3.12 and the pinned dependencies from `uv.lock`.
- `torch==2.14.1` is installed from the PyTorch CUDA 13.0 index (`pyproject.toml` → `[tool.uv.sources]`). Blackwell GPUs (sm_120, e.g. the RTX 50xx series) need a wheel built with CUDA ≥ 12.8 ([PyTorch install matrix](https://pytorch.org/get-started/locally/)).
- `transformers==4.57.6` is pinned because Ouro ships remote code (`trust_remote_code=True`). The model card recommends `transformers<4.56`, but the pinned revision already contains the KV-cache fix for >=4.56 and fails on 4.54.1 (`key_cache` has no setter). The model revision is pinned in `.env` (`MODEL_REVISION`).

## Local labeler / judge LLM (llama.cpp + Qwen3.5 GGUF)
This LLM is used only by the labeling, LLM-judge, and audit phases. The primary steering statistic does not use an LLM.
1. Download a CUDA build of `llama-server` from https://github.com/ggml-org/llama.cpp/releases and unzip it into `tools/llama.cpp/`. This folder is gitignored.
2. Download a Qwen3.5 GGUF (Q4_K_M) into `cache/gguf/`. This folder is gitignored.
3. Start the server on port 8093 and set `LLM_BASE_URL` and `LLM_MODEL` in `.env`.

The exact release tag, GGUF repository, and file hash are written here once the setup phase has run.

## Disk budget (development machine: 24.7 GB free)
| Item | Size |
|---|---|
| `.venv` | ~5.5 GB |
| Ouro-1.4B (`model.safetensors`, bf16) | 2.87 GB |
| Ouro-1.4B-Thinking | 2.87 GB |
| llama.cpp CUDA binaries | ~0.6 GB |
| Qwen3.5-9B Q4_K_M GGUF | ~6 GB |
| atlas + results | < 0.6 GB |

`loopneuron.config.disk_guard()` refuses any download that would leave less than `MIN_FREE_GB` (default 3 GB) free.
