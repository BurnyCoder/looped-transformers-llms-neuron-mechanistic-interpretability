"""Phase: loop-level interpretability audit (H4: does a looped neuron do the same job at every loop?).

Local:
- `loop_structure` (all neurons, no LLM): which loop holds each neuron's strongest activations, per-loop mean |a|,
  and Jaccard overlap of the top-20 context sets between loops (overlap 1 = same contexts at both loops).
- `llm_audit` (sampled neurons, local Qwen3.5): per-loop Delphi explanations from ranks 1-12 of that loop's top
  contexts; detection balanced accuracy on held-out ranks 13-20 vs random contexts, for (i) the same loop,
  (ii) cross-loop (loop-1 explanation scored on loop-4 examples and vice versa) and (iii) a shuffled explanation
  from another neuron (chance calibration). Delphi: Paulo et al. 2024, https://arxiv.org/abs/2410.13928.
Global context: phase `audit` in run_pipeline.py; results/<model>/audit.json, loop_structure.json.
"""
from __future__ import annotations

import numpy as np
import torch

from loopneuron import contexts as cx
from loopneuron.autointerp import detection_score, explain

N_EXPLAIN, N_HELD = 12, 8


def loop_structure(atlas: dict) -> dict:
    """Per-layer summaries of loop specialization over all alive neurons (positive and negative poles)."""
    T = atlas["lt_v"].shape[1]
    alive = ~atlas["dead"]
    out = {"loop_of_max_share": [], "mean_abs_by_loop": [], "jaccard_first_last": [], "jaccard_adjacent": []}
    for layer in range(atlas["lt_v"].shape[0]):
        lm = atlas["top_loop"][layer, :, :, 0][:, alive[layer]]  # loop of the single strongest context
        out["loop_of_max_share"].append([float((lm == t).float().mean()) for t in range(T)])
        out["mean_abs_by_loop"].append(atlas["mean_abs"][layer][:, alive[layer]].mean(1).tolist())
        seqs = atlas["lt_seq"][layer][:, 0][:, alive[layer]]  # [T, n_alive, k] positive pole
        def jac(a, b):  # per-neuron Jaccard of top-k sequence-id sets
            return float(np.mean([len(set(x.tolist()) & set(y.tolist())) / len(set(x.tolist()) | set(y.tolist()))
                                  for x, y in zip(a, b)]))
        idx = torch.randperm(seqs.shape[1], generator=torch.Generator().manual_seed(layer))[:500]  # subsample for speed
        out["jaccard_first_last"].append(jac(seqs[0][idx], seqs[T - 1][idx]))
        out["jaccard_adjacent"].append([jac(seqs[t][idx], seqs[t + 1][idx]) for t in range(T - 1)])
    return out


def _examples(atlas, chunks, tok, layer, neuron, loop, lo, hi, mark=True):
    ents = cx.neuron_entries(atlas, layer, 0, neuron, loop=loop)[lo:hi]
    return [(cx.window_text(tok, chunks, s, p, mark=mark), v) for s, p, v, _ in ents]


def llm_audit(llm, atlas, chunks, tok, seed, log, per_tercile: int = 50) -> list[dict]:
    rng = np.random.default_rng(seed)
    L, T = atlas["lt_v"].shape[0], atlas["lt_v"].shape[1]
    terciles = np.array_split(np.arange(L), 3)
    picks = []
    for ls in terciles:
        cand = [(int(l), int(n)) for l in ls for n in np.where(~atlas["dead"][l].numpy())[0]]
        picks += [cand[i] for i in rng.choice(len(cand), per_tercile, replace=False)]
    rand = [cx.window_text(tok, chunks, s, p) for s, p in cx.random_entries(chunks, 4000, seed + 7)]
    out, expl_by = [], {}
    for i, (layer, neuron) in enumerate(picks):
        rec = {"layer": layer, "neuron": neuron, "explanations": {}, "same_loop": {}, "cross_loop": {}}
        for t in (0, T - 1):  # first and last loop (the H4 contrast)
            ex = _examples(atlas, chunks, tok, layer, neuron, t, 0, N_EXPLAIN)
            rec["explanations"][t] = explain(llm, ex, ex[0][1], tag=f"audit L{layer}N{neuron} loop{t + 1}")
        held = {t: [x for x, _ in _examples(atlas, chunks, tok, layer, neuron, t, N_EXPLAIN, N_EXPLAIN + N_HELD, mark=False)]
                for t in (0, T - 1)}
        negs = [rand[j] for j in rng.choice(len(rand), N_HELD, replace=False)]
        for t, t2 in ((0, 0), (T - 1, T - 1)):
            rec["same_loop"][t] = detection_score(llm, rec["explanations"][t], held[t2], negs, seed + i, tag="same")
        for t, t2 in ((0, T - 1), (T - 1, 0)):
            rec["cross_loop"][t] = detection_score(llm, rec["explanations"][t], held[t2], negs, seed + i, tag="cross")
        expl_by[(layer, neuron)] = rec["explanations"][T - 1]
        out.append(rec)
        log.info("AUDIT %d/%d L%d N%d same=%s cross=%s | %s", i + 1, len(picks), layer, neuron,
                 {k: round(v["balanced_acc"], 2) for k, v in rec["same_loop"].items()},
                 {k: round(v["balanced_acc"], 2) for k, v in rec["cross_loop"].items()}, rec["explanations"][T - 1])
    keys = list(expl_by)
    for i, rec in enumerate(out):  # shuffled-explanation control: another neuron's last-loop explanation
        other = expl_by[keys[(i + 1) % len(keys)]]
        held = [x for x, _ in _examples(atlas, chunks, tok, rec["layer"], rec["neuron"], T - 1, N_EXPLAIN, N_EXPLAIN + N_HELD, mark=False)]
        negs = [rand[j] for j in rng.choice(len(rand), N_HELD, replace=False)]
        rec["shuffled"] = detection_score(llm, other, held, negs, seed + 10_000 + i, tag="shuffled")
    return out
