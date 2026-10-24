"""Single entry point: runs the study's named phases in order (see README "What happens when you run it").

Usage:
  uv run python run_pipeline.py                       # all phases, skipping those whose outputs exist
  uv run python run_pipeline.py --phase build_atlas   # one phase
  uv run python run_pipeline.py --dry-run             # tiny budgets through every phase (smoke test)
Global context: each phase is a thin function here that calls modules in src/loopneuron/ and writes results/<model>/.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from loopneuron.config import load_settings
from loopneuron.data import build_prompt_splits, cached_chunks, load_prompt_split, stream_texts
from loopneuron.logging_utils import get_logger
from loopneuron.model import load_model

PHASES = ["setup_check", "build_atlas"]  # extended as phases are implemented


class Ctx:
    """Lazily loaded shared state (settings, logger, model, tokenizer, data) passed to every phase."""

    def __init__(self, s, log):
        self.s, self.log, self._model = s, log, None
        self.out = s.results_dir
        self.out.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        if self._model is None:
            self.log.info("loading %s @ %s (T=%d)", self.s.model_id, self.s.model_revision, self.s.ut_steps)
            self._model, self.tok = load_model(self.s.model_id, self.s.model_revision, ut_steps=self.s.ut_steps)
        return self._model

    def chunks(self, split: str) -> np.ndarray:
        """Packed token chunks for 'atlas' or 'eval' (cached under cache/), truncated to the configured budget."""
        _ = self.model  # tokenizer comes with the model
        n = self.s.atlas_tokens if split == "atlas" else 256 * self.s.seq_len
        arr = cached_chunks(self.s.root / "cache" / f"tokens_{split}.npy",
                            texts=stream_texts(split, self.s.math_fraction, self.s.seed), tok=self.tok,
                            seq_len=self.s.seq_len, n_tokens=n)
        return arr[: n // self.s.seq_len]

    def prompts(self, name: str) -> list[dict]:
        d = self.s.root / "data" / "prompts"
        if not (d / f"{name}.jsonl").exists():
            _ = self.model
            build_prompt_splits(self.tok, d, seed=self.s.seed)
        return load_prompt_split(d, name)


def setup_check(c: Ctx) -> None:
    from loopneuron.sanity import setup_check as run

    run(c.model, c.tok, c.chunks("eval"), c.prompts("discovery"), c.s.ut_steps, c.out / "setup_check.json", c.log)


def build_atlas(c: Ctx) -> None:
    from loopneuron.atlas import build_atlas as run

    path = c.out / "atlas.pt"
    if path.exists() and torch.load(path, weights_only=False).get("done_chunks", 0) >= len(c.chunks("atlas")):
        c.log.info("atlas exists: %s", path)
        return
    run(c.model, c.chunks("atlas"), c.s.ut_steps, c.s.topk, c.s.topk_per_loop, c.s.atlas_batch, c.log, save_path=path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=PHASES, help="run only this phase")
    ap.add_argument("--dry-run", action="store_true", help="tiny budgets, separate results folder")
    args = ap.parse_args()
    # dry run: tiny budgets and a separate results folder (results/<model>-dryrun/)
    over = {"atlas_tokens": 256 * 64, "null_neurons": 5, "run_tag": "-dryrun"} if args.dry_run else {}
    s = load_settings(**over)
    log = get_logger("dryrun" if args.dry_run else (args.phase or "pipeline"), s.logs_dir)
    c = Ctx(s, log)
    for name in [args.phase] if args.phase else PHASES:
        log.info("=== phase: %s ===", name)
        globals()[name](c)


if __name__ == "__main__":
    main()
