# AGENTS.md — notes for coding agents working on this repo

- Entry point: `run_pipeline.py`. It calls named phases from `src/loopneuron/`. Keep phase functions thin, and keep the logic in modules.
- Model: `ByteDance/Ouro-1.4B`, revision pinned in `.env`.
  - Always load it through `loopneuron.model.load_model`. That function pins `exit_at_step=T-1` and sets bf16 + eval.
  - Never call the model with the default early-exit threshold.
- Neuron hooks: `loopneuron.hooks`. One forward pre-hook per `layers[l].mlp.down_proj`. The loop index is the per-layer call counter modulo T, and it must be reset before every forward (`reset()`).
- GPU is 8 GB. Never load Ouro and the Qwen GGUF server at the same time.
- Disk is tight (~25 GB free on the dev machine). Call `config.disk_guard()` before any download.
- Statistics: confirmatory data (`data/prompts/confirm.jsonl`) must never be used for selection or tuning. Pre-registration lives in `reports/preregistration.md`.
- Logging: use `loopneuron.logging_utils.get_logger`. LLM prompts and outputs must be logged in full.
- Tests: `uv run pytest -q`. GPU tests are marked `gpu`.
