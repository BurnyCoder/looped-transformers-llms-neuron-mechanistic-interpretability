"""Neuron atlas: streaming top-k max-activating contexts per neuron, pole and loop, plus activation statistics.

Local: Ouro runs layers 0..L-1 for loops t=0..T-1; `NeuronHooks` calls `AtlasRecorder.__call__(layer, loop, act)`
with act [B,S,D] (D=5632 SwiGLU neurons). Per call we reduce over positions (max for the positive pole, min for
the negative pole), merge into running top-k tables with `torch.topk` (https://pytorch.org/docs/stable/generated/torch.topk.html),
and update per-loop moments and threshold-exceedance counts (a coarse activation distribution per sign). Overall (loop-pooled) top-k uses the max over
loops of each (sequence, neuron), recorded with the loop and position where it occurred.
This is the max-activating-dataset-examples method used by Bills et al. 2023 and Goodfire (2026).
Global context: `build_atlas` produces results/<model>/atlas.pt consumed by screening, labeling, viewer, audit.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from loopneuron.hooks import NeuronHooks

THRESHOLDS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)  # |a| levels for exceedance counts (both signs)


def merge_topk(vals, idx, new_vals, new_idx, k):
    """Merge running top-k (vals [D,k], idx dict of [D,k]) with candidates (new_vals [D,n], new_idx dict of [D,n]).

    Returns updated (vals, idx). Ties are broken by torch.topk order; NaNs never occur (bf16 activations are finite).
    """
    cat_v = torch.cat([vals, new_vals], dim=1)  # [D, k+n]
    top_v, top_i = torch.topk(cat_v, k, dim=1)  # largest k per neuron
    out = {name: torch.gather(torch.cat([idx[name], new_idx[name]], dim=1), 1, top_i) for name in idx}
    return top_v, out


class AtlasRecorder:
    """Accumulates the atlas for all layers on the GPU. Call `start_batch(seq_ids)` before each forward."""

    def __init__(self, n_layers: int, d_mlp: int, T: int, k: int, k_loop: int, skip_pos: int = 4, device="cuda",
                 stats_every: int = 1):
        """Allocate GPU top-k tables (overall and per loop, both poles) and float64 moment accumulators."""
        self.L, self.D, self.T, self.k, self.kl, self.skip = n_layers, d_mlp, T, k, k_loop, skip_pos
        self.stats_every = stats_every  # moments/exceedances on every n-th batch only (top-k always uses all batches)
        self.n_batches = 0
        f = lambda *shape: torch.full(shape, -torch.inf, device=device)  # top-k values start at -inf
        z = lambda *shape, dt=torch.int32: torch.zeros(shape, dtype=dt, device=device)
        # overall top-k per pole (pole 0 = positive: largest a; pole 1 = negative: largest -a)
        self.top_v = f(n_layers, 2, d_mlp, k)
        self.top = {"seq": z(n_layers, 2, d_mlp, k), "pos": z(n_layers, 2, d_mlp, k), "loop": z(n_layers, 2, d_mlp, k)}
        # per-loop top-k per pole
        self.lt_v = f(n_layers, T, 2, d_mlp, k_loop)
        self.lt = {"seq": z(n_layers, T, 2, d_mlp, k_loop), "pos": z(n_layers, T, 2, d_mlp, k_loop)}
        # per-loop moments (float64 sums avoid drift over millions of tokens)
        self.sum = torch.zeros(n_layers, T, d_mlp, dtype=torch.float64, device=device)
        self.sum_abs = torch.zeros_like(self.sum)
        self.sum_sq = torch.zeros_like(self.sum)
        self.n_pos = torch.zeros_like(self.sum)  # count of a > 0
        self.exceed = torch.zeros(n_layers, T, 2, len(THRESHOLDS), d_mlp, dtype=torch.int64, device=device)
        self.n_tokens = 0  # tokens per loop per layer accumulated in moments
        self._best = None  # per-forward running max over loops: (val, pos, loop) per pole

    def start_batch(self, seq_ids: torch.Tensor) -> None:
        """Register the chunk ids of the next forward pass and decide whether this batch also feeds the moments."""
        self.seq_ids = seq_ids.to(self.top_v.device, torch.int32)  # global chunk index of each batch row
        self._best = {}
        self.do_stats = self.n_batches % self.stats_every == 0  # subsampled statistics (unbiased: whole batches)
        self.n_batches += 1

    @torch.no_grad()
    def __call__(self, layer: int, loop: int, act: torch.Tensor) -> None:
        """Hook callback: reduce [B,S,D] activations of one (layer, loop) into the running tables."""
        a = act[:, self.skip:, :]  # drop first positions (attention-sink / no-context artefacts); stays bf16
        B, S, D = a.shape
        if self.do_stats:
            flat = a.reshape(-1, D).float()
            if layer == 0 and loop == 0:
                self.n_tokens += B * S  # tokens per (layer, loop) entering the moments
            self.sum[layer, loop] += flat.sum(0).double()
            self.sum_abs[layer, loop] += flat.abs().sum(0).double()
            self.sum_sq[layer, loop] += flat.pow(2).sum(0).double()
            self.n_pos[layer, loop] += (flat > 0).sum(0).double()
            for j, thr in enumerate(THRESHOLDS):  # counts of a > thr (pole +) and a < -thr (pole -)
                self.exceed[layer, loop, 0, j] += (flat > thr).sum(0)
                self.exceed[layer, loop, 1, j] += (flat < -thr).sum(0)
        for pole in (0, 1):  # positive pole: max of a; negative pole: max of -a (= -min a)
            v, p = a.max(dim=1) if pole == 0 else a.min(dim=1)  # [B, D] best value and position per sequence
            v = v.float() if pole == 0 else -v.float()
            p = p + self.skip
            seq = self.seq_ids[:, None].expand(B, D)
            # per-loop top-k merge
            vals, idx = merge_topk(self.lt_v[layer, loop, pole], {k: self.lt[k][layer, loop, pole] for k in self.lt},
                                   v.T, {"seq": seq.T.int(), "pos": p.T.int()}, self.kl)
            self.lt_v[layer, loop, pole] = vals
            for k_ in self.lt:
                self.lt[k_][layer, loop, pole] = idx[k_]
            # running max over loops for the overall table
            key = (layer, pole)
            if key not in self._best:
                self._best[key] = (v, p, torch.full_like(p, loop))
            else:
                bv, bp, bl = self._best[key]
                better = v > bv
                self._best[key] = (torch.where(better, v, bv), torch.where(better, p, bp), torch.where(better, loop, bl))
            if loop == self.T - 1:  # all loops seen for this layer -> merge into overall top-k
                bv, bp, bl = self._best.pop(key)
                vals, idx = merge_topk(self.top_v[layer, pole], {k: self.top[k][layer, pole] for k in self.top},
                                       bv.T, {"seq": seq.T.int(), "pos": bp.T.int(), "loop": bl.T.int()}, self.k)
                self.top_v[layer, pole] = vals
                for k_ in self.top:
                    self.top[k_][layer, pole] = idx[k_]

    def summary(self) -> dict:
        """CPU dict of all tables + derived per-neuron statistics (means, stds, positive fraction, dead flag)."""
        n = max(self.n_tokens, 1)
        mean = (self.sum / n).float()
        out = {
            "top_v": self.top_v.cpu(), **{f"top_{k}": v.cpu() for k, v in self.top.items()},
            "lt_v": self.lt_v.cpu(), **{f"lt_{k}": v.cpu() for k, v in self.lt.items()},
            "mean": mean.cpu(), "mean_abs": (self.sum_abs / n).float().cpu(),
            "std": (self.sum_sq / n - (self.sum / n) ** 2).clamp_min(0).sqrt().float().cpu(),
            "frac_pos": (self.n_pos / n).float().cpu(), "exceed": self.exceed.cpu(), "thresholds": THRESHOLDS,
            "n_tokens": n,
        }
        max_abs = torch.maximum(out["top_v"][:, 0, :, 0], out["top_v"][:, 1, :, 0])  # [L, D] max |a| over poles
        out["max_abs"] = max_abs
        out["dead"] = max_abs < 1e-3
        return out


@torch.no_grad()
def build_atlas(model, chunks: np.ndarray, T: int, k: int, k_loop: int, batch: int, log, save_path=None,
                checkpoint_every: int = 200) -> dict:
    """Run all `chunks` [N,S] through the model with an AtlasRecorder attached; returns `summary()`."""
    cfg = model.config
    rec = AtlasRecorder(cfg.num_hidden_layers, cfg.intermediate_size, T, k, k_loop, stats_every=8)
    t0 = time.time()
    n_batches = (len(chunks) + batch - 1) // batch
    with NeuronHooks(model, T, recorder=rec):
        for bi in range(n_batches):
            sl = slice(bi * batch, min((bi + 1) * batch, len(chunks)))
            ids = torch.from_numpy(chunks[sl].astype(np.int64)).cuda()
            rec.start_batch(torch.arange(sl.start, sl.stop))
            model(ids, use_cache=False)  # logits unused; hooks do the work
            if (bi + 1) % 20 == 0 or bi == n_batches - 1:
                el = time.time() - t0
                tok_s = (sl.stop * chunks.shape[1]) / el
                log.info("atlas batch %d/%d | %.0f tok/s | eta %.1f min", bi + 1, n_batches, tok_s,
                         (len(chunks) - sl.stop) * chunks.shape[1] / tok_s / 60)
            if save_path is not None and (bi + 1) % checkpoint_every == 0:
                torch.save({**rec.summary(), "done_chunks": sl.stop}, save_path)
    out = rec.summary()
    out["done_chunks"] = len(chunks)
    if save_path is not None:
        torch.save(out, save_path)
    log.info("atlas done: %d tokens/loop/layer in %.1f min", out["n_tokens"], (time.time() - t0) / 60)
    return out
