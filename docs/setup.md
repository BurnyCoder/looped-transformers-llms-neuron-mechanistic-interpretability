# Setup details

## Python environment
- Run `uv sync`. It installs Python 3.12 and the pinned dependencies from `uv.lock`.
- `torch==2.14.1` is installed from the PyTorch CUDA 13.0 index (`pyproject.toml` → `[tool.uv.sources]`). Blackwell GPUs (sm_120, e.g. the RTX 50xx series) need a wheel built with CUDA ≥ 12.8 ([PyTorch install matrix](https://pytorch.org/get-started/locally/)).
- `transformers==4.57.6` is pinned because Ouro ships remote code (`trust_remote_code=True`). The model card recommends `transformers<4.56`, but the pinned revision already contains the KV-cache fix for >=4.56 and fails on 4.54.1 (`key_cache` has no setter). The model revision is pinned in `.env` (`MODEL_REVISION`).

## Local labeler / judge LLM (llama.cpp + Qwen3.5 GGUF)
This LLM is used only by the labeling, LLM-judge, and audit phases. The primary steering statistic does not use an LLM.
1. Download llama.cpp release **b11854** (Windows CUDA 13.4 build: `llama-b11854-bin-win-cuda-13.4-x64.zip` plus `cudart-llama-bin-win-cuda-13.4-x64.zip`) from https://github.com/ggml-org/llama.cpp/releases/tag/b11854 and unzip both into `tools/llama.cpp/`. On Linux, use the matching `ubuntu-cuda` build.
2. Download `Qwen3.5-9B-Q4_K_M.gguf` (5.68 GB) from https://huggingface.co/unsloth/Qwen3.5-9B-GGUF into `cache/gguf/`:

```bash
uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3.5-9B-GGUF', 'Qwen3.5-9B-Q4_K_M.gguf', local_dir='cache/gguf')"
```

3. The pipeline starts and stops `llama-server` itself (`loopneuron.llm.LlamaServer`), with `-ngl 99 -c 16384 --port 8093 --jinja`. It disables thinking per request (`chat_template_kwargs={"enable_thinking": false}`) and writes the server output to `logs/llama_server.txt`.
   - Port 8093 is used because 8080 was occupied on the development machine. Change it with `LLM_BASE_URL` in `.env`.
   - Ouro is unloaded from the GPU before the server starts, because both do not fit in 8 GB together.

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
