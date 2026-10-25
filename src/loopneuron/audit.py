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
        def jac(a, b):
            """Mean over neurons of |A ∩ B| / |A ∪ B| for their top-k chunk-id sets at two loops."""
            return float(np.mean([len(set(x.tolist()) & set(y.tolist())) / len(set(x.tolist()) | set(y.tolist()))
                                  for x, y in zip(a, b)]))
        idx = torch.randperm(seqs.shape[1], generator=torch.Generator().manual_seed(layer))[:500]  # subsample for speed
        out["jaccard_first_last"].append(jac(seqs[0][idx], seqs[T - 1][idx]))
        out["jaccard_adjacent"].append([jac(seqs[t][idx], seqs[t + 1][idx]) for t in range(T - 1)])
    return out


@torch.no_grad()
def cross_loop_correlation(model, chunks: np.ndarray, T: int, batch: int = 4) -> np.ndarray:
    """Per-neuron Pearson correlation between loops over the same tokens -> [L, D, T, T] (robust to top-k ties).

    Streaming sums of a_t, a_t^2 and a_t*a_s per neuron (float64) over all positions >= 4 of `chunks`.
    """
    from loopneuron.hooks import NeuronHooks

    L, D = model.config.num_hidden_layers, model.config.intermediate_size
    S1 = torch.zeros(L, T, D, dtype=torch.float64, device="cuda")
    S2 = torch.zeros(L, T, T, D, dtype=torch.float64, device="cuda")
    cur, n = {}, 0

    def rec(layer, loop, act):
        """Hook callback: accumulate first and cross moments of this (layer, loop) with earlier loops."""
        a = act[:, 4:, :].reshape(-1, D)
        cur[(layer, loop)] = a  # kept in bf16: all layers' earlier loops are alive until loop T-1 (~1 GB at batch 4)
        a = a.double()
        S1[layer, loop] += a.sum(0)
        for s in range(loop + 1):  # products with all earlier loops of this layer (and itself)
            S2[layer, loop, s] += (a * cur[(layer, s)].double()).sum(0)
        if loop == T - 1:
            for s in range(T):
                cur.pop((layer, s))

    with NeuronHooks(model, T, recorder=rec):
        for i in range(0, len(chunks), batch):
            ids = torch.from_numpy(chunks[i:i + batch].astype(np.int64)).cuda()
            model(ids, use_cache=False)
            n += ids.shape[0] * (ids.shape[1] - 4)
    mean = S1 / n
    cov = torch.zeros(L, T, T, D, dtype=torch.float64, device="cuda")
    for t in range(T):
        for s in range(t + 1):
            cov[:, t, s] = cov[:, s, t] = S2[:, t, s] / n - mean[:, t] * mean[:, s]
    sd = torch.stack([cov[:, t, t] for t in range(T)], 1).clamp_min(1e-20).sqrt()  # [L, T, D]
    corr = cov / (sd[:, :, None, :] * sd[:, None, :, :])
    return corr.permute(0, 3, 1, 2).float().cpu().numpy()


def _examples(atlas, chunks, tok, layer, neuron, loop, lo, hi, mark=True):
    """(window text, activation) for ranks lo..hi-1 of a neuron's positive-pole top contexts at one loop."""
    ents = cx.neuron_entries(atlas, layer, 0, neuron, loop=loop)[lo:hi]
    return [(cx.window_text(tok, chunks, s, p, mark=mark), v) for s, p, v, _ in ents]


def llm_audit(llm, atlas, chunks, tok, seed, log, per_tercile: int = 50) -> list[dict]:
    """Explain each sampled neuron at loop 1 and loop T; score same-loop, cross-loop and shuffled explanations."""
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
