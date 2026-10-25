"""Phase 0 (setup_check): verify the loaded model behaves like the published Ouro-1.4B before any analysis.

Checks (all written to results/<model>/setup_check.json and the log):
0. the checkpoint loads with no missing, unexpected or mismatched state-dict keys;
1. held-out FineWeb-Edu loss/perplexity for T = 1..8 loops (T>4 is extrapolation; Zhu et al. 2025,
   https://arxiv.org/abs/2510.25741, report loss improving with recurrent steps up to the trained T=4);
2. how often the model's *default* early-exit path (threshold 1.0 on a bf16 cumulative exit PDF) disagrees with the
   fixed last-loop exit we use everywhere (motivates `exit_at_step=T-1`);
3. a few sampled continuations (logged in full) to eyeball fluency.
Global context: first phase of run_pipeline.py; later phases assume these checks passed.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from loopneuron.model import sample, set_ut_steps


@torch.no_grad()
def eval_loss(model, chunks: np.ndarray, batch: int = 8) -> float:
    """Mean next-token cross-entropy (nats) over `chunks` [N,S], skipping the first position's prediction target."""
    tot, n = 0.0, 0
    for i in range(0, len(chunks), batch):
        ids = torch.from_numpy(chunks[i:i + batch].astype(np.int64)).cuda()
        logits = model(ids, use_cache=False).logits.float()
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
        tot, n = tot + loss.item(), n + ids[:, 1:].numel()
    return tot / n


@torch.no_grad()
def default_exit_disagreement(model, chunks: np.ndarray, T: int) -> float:
    """Fraction of positions whose argmax differs between the default threshold exit and the fixed last-loop exit."""
    ids = torch.from_numpy(chunks.astype(np.int64)).cuda()
    fixed = model(ids, use_cache=False).logits.argmax(-1)
    model.early_exit_step = None  # temporarily restore the model's default exit logic
    try:
        default = model(ids, use_cache=False, exit_threshold=1.0).logits.argmax(-1)
    finally:
        model.early_exit_step = T - 1
    return (fixed != default).float().mean().item()


def setup_check(model, tok, eval_chunks: np.ndarray, prompts: list[dict], T: int, out_path, log,
                max_T: int = 8, n_eval: int = 64) -> dict:
    """Run the sanity checks (environment, state dict, loss vs T, exit path, samples) and save a JSON summary."""
    props = torch.cuda.get_device_properties(0)  # GPU name / VRAM (https://pytorch.org/docs/stable/cuda.html)
    res = {"environment": {"gpu": props.name, "vram_gb": round(props.total_memory / 1e9, 2),
                           "compute_capability": f"{props.major}.{props.minor}", "torch": torch.__version__,
                           "disk_free_gb": round(shutil.disk_usage(out_path.parent.parent).free / 1e9, 1)},
           "loading_info": model.loading_info, "loss_by_T": {}}
    log.info("environment: %s", res["environment"])
    bad = {k: v for k, v in model.loading_info.items() if v}
    log.info("state dict: %s", "all keys matched" if not bad else f"PROBLEMS {bad}")
    for t in range(1, max_T + 1):
        set_ut_steps(model, t)
        loss = eval_loss(model, eval_chunks[:n_eval])
        res["loss_by_T"][t] = loss
        log.info("held-out loss T=%d: %.4f nats (ppl %.2f)", t, loss, float(np.exp(loss)))
    set_ut_steps(model, T)
    res["default_exit_disagreement"] = default_exit_disagreement(model, eval_chunks[:8], T)
    log.info("default early-exit path disagrees with fixed last-loop exit on %.4f of positions",
             res["default_exit_disagreement"])
    ids = torch.tensor([p["ids"] for p in prompts[:4]]).cuda()
    gen = sample(model, ids, max_new_tokens=48, seed=0)
    res["samples"] = [{"prompt": p["text"], "continuation": tok.decode(g)} for p, g in zip(prompts[:4], gen)]
    for s in res["samples"]:
        log.info("SAMPLE prompt=%r\n  continuation=%r", s["prompt"], s["continuation"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
