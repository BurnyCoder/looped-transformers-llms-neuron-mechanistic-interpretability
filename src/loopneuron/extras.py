"""Plan-completion analyses (secondary, run after the confirmatory decision; none feeds back into it).

Local, per confirmed neuron at its pre-registered alpha unless stated:
- seed_sensitivity: the confirmatory effect re-estimated with 3 independent seed sets (k=1 each, 100 confirm prompts);
- zero_clamp: dose-response at alpha = 0 (clamp to 0 at every position and loop, i.e. ablation during generation);
- concept_token_shift: teacher-forced change of next-token probability mass on the neuron's keyword tokens
  (tokens over-represented around its top-context peaks), scored on unsteered continuations;
- self_repair: for a single-loop clamp at loop t, relative residual displacement right after the steered layer at
  loop t ("injected") vs at the final norm ("surviving"); ratio < 1 means later computation shrinks the injection;
- tsweep_fit: saturating-exponential fit y(T) = a - b*exp(-T/tau) of the T-sweep (scipy curve_fit);
- audit_wilson: Wilson 95% CIs for the share of well-explained neurons (balanced accuracy >= 0.75);
- positional_check: whether candidates' top contexts peak at a fixed position (positional neurons, Voita et al. 2023).
Global context: phase `extras` in run_pipeline.py -> results/<model>/extras.json.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from scipy.optimize import curve_fit
from statsmodels.stats.proportion import proportion_confint

from loopneuron import contexts as cx
from loopneuron import stats
from loopneuron.hooks import NeuronHooks
from loopneuron.metrics import Embedder, concept_scores, keyword_logmass
from loopneuron.steering import Cond, clamp_value, generate


def _cond(atlas, c, alpha=None, loops=None) -> Cond:
    """Steering condition for candidate c at its pre-registered alpha (or `alpha`), optionally loop-restricted."""
    a =c["alpha"] if alpha is None else alpha
    return Cond(c["layer"], c["neuron"], clamp_value(atlas, c["layer"], c["pole"], c["neuron"], a), loops)


def _summary(d: np.ndarray, seed: int = 0) -> dict:
    """Mean with bootstrap 95% CI and one-sided Wilcoxon p of per-prompt differences."""
    lo, hi = stats.bootstrap_ci(d, n_boot=4000, seed=seed)
    return {"mean": float(d.mean()), "ci95": [lo, hi], "p_wilcoxon": stats.wilcoxon_greater(d), "n": int(len(d))}


def seed_sensitivity(model, tok, atlas, cands, u, P, T, n_tokens, seeds, log) -> tuple[dict, np.ndarray]:
    """Returns ({key: [summary per seed set]}, unsteered generations of the first seed set)."""
    emb, out, first_base = Embedder(), {}, None
    for s in seeds:
        base = generate(model, P, None, T, n_tokens, s)
        first_base = base if first_base is None else first_base
        be = emb([tok.decode(g) for g in base])
        for c in cands:
            e = emb([tok.decode(g) for g in generate(model, P, _cond(atlas, c), T, n_tokens, s)])
            d = concept_scores(e, np.array(u[c["key"]])) - concept_scores(be, np.array(u[c["key"]]))
            out.setdefault(c["key"], []).append({"seed": s, **_summary(d, s)})
            log.info("SEED %s seed=%d meanDC=%.4f p=%.1e", c["key"], s, d.mean(), out[c["key"]][-1]["p_wilcoxon"])
    return out, first_base


def zero_clamp(model, tok, atlas, cands, u, P, T, n_tokens, seed, log) -> dict:
    """alpha = 0: the neuron is held at 0 during generation (all positions and loops)."""
    emb = Embedder()
    base = generate(model, P, None, T, n_tokens, seed)
    be = emb([tok.decode(g) for g in base])
    out = {}
    for c in cands:
        e = emb([tok.decode(g) for g in generate(model, P, Cond(c["layer"], c["neuron"], 0.0), T, n_tokens, seed)])
        d = concept_scores(e, np.array(u[c["key"]])) - concept_scores(be, np.array(u[c["key"]]))
        out[c["key"]] = _summary(d, seed)
        log.info("ZERO %s meanDC=%.4f p=%.2f", c["key"], d.mean(), out[c["key"]]["p_wilcoxon"])
    return out


def concept_token_shift(model, tok, atlas, chunks, cands, P, base: np.ndarray, T, seed, log) -> dict:
    """Teacher-forced Delta log p(keyword tokens) on prompt + unsteered continuation, steered minus unsteered."""
    bg, n_bg = cx.background_counts(chunks, cx.random_entries(chunks, 5000, seed + 11))
    ids = torch.cat([P, torch.from_numpy(base.astype(np.int64))], dim=1)
    clean_cache, out = {}, {}
    for c in cands:
        kw = cx.keyword_ids(chunks, cx.neuron_entries(atlas, c["layer"], c["pole"], c["neuron"]), bg, n_bg)
        key = tuple(kw)
        if key not in clean_cache:
            clean_cache[key] = keyword_logmass(model, ids, kw, P.shape[1])
        with NeuronHooks(model, T, edits=_cond(atlas, c).edits()):
            steered = keyword_logmass(model, ids, kw, P.shape[1])
        d = steered - clean_cache[key]
        out[c["key"]] = {**_summary(d, seed), "keywords": [tok.decode([t]) for t in kw]}
        log.info("TOKENS %s dlogp=%.3f p=%.1e keywords=%s", c["key"], d.mean(), out[c["key"]]["p_wilcoxon"],
                 out[c["key"]]["keywords"][:10])
    return out


@torch.no_grad()
def self_repair(model, atlas, cands, P, T, log) -> dict:
    """Injected (right after the steered layer, at the steered loop) vs surviving (final norm) relative displacement."""
    out = {}
    ids = P.cuda()
    for c in cands:
        layer = model.model.layers[c["layer"]]
        rec: list = []
        h = layer.register_forward_hook(lambda m, i, o: rec.append((o[0] if isinstance(o, tuple) else o).float()))
        try:
            clean_final = model.model(ids, use_cache=False)[0].last_hidden_state.float()
            clean_layer = list(rec)
            rows = []
            for t in range(T):
                rec.clear()
                with NeuronHooks(model, T, edits=_cond(atlas, c, loops=(t,)).edits()):
                    st_final = model.model(ids, use_cache=False)[0].last_hidden_state.float()
                rel = lambda a, b: ((a - b).norm(dim=-1) / b.norm(dim=-1)).mean().item()
                inj, surv = rel(rec[t], clean_layer[t]), rel(st_final, clean_final)
                rows.append({"loop": t + 1, "injected": inj, "surviving": surv, "ratio": surv / inj})
        finally:
            h.remove()
        out[c["key"]] = rows
        log.info("REPAIR %s ratios=%s", c["key"], [round(r["ratio"], 3) for r in rows])
    return out


def tsweep_fit(followups: dict) -> dict:
    """Fit y = a - b*exp(-T/tau) to Delta C(T) and KL(T) per neuron; returns params and R^2 (None if the fit fails)."""
    f = lambda T, a, b, tau: a - b * np.exp(-T / tau)
    out = {}
    for key, v in followups.items():
        if key == "specificity" or "t_sweep" not in v:
            continue
        T = np.array([r["T"] for r in v["t_sweep"]["rows"]], dtype=float)
        out[key] = {}
        for name, y in (("dC", np.array([r["mean_dC"] for r in v["t_sweep"]["rows"]])),
                        ("kl", np.array([r["kl"] for r in v["t_sweep"]["rows"]]))):
            try:
                p, _ = curve_fit(f, T, y, p0=(y.max(), y.max() - y.min() + 1e-3, 1.0),
                                 bounds=([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 50.0]), maxfev=20000)
                r2 = 1 - ((y - f(T, *p)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                out[key][name] = {"a": float(p[0]), "b": float(p[1]), "tau": float(p[2]), "r2": float(r2)}
            except RuntimeError:
                out[key][name] = None
    return out


def audit_wilson(audit: list, thr: float = 0.75) -> dict:
    """Share of neurons whose same-loop explanation reaches balanced accuracy >= thr, with Wilson 95% CIs."""
    out = {}
    for t in ("0", "3"):
        v = np.array([r["same_loop"][t]["balanced_acc"] for r in audit])
        v = v[np.isfinite(v)]
        k, n = int((v >= thr).sum()), len(v)
        lo, hi = proportion_confint(k, n, method="wilson")
        out[f"loop{int(t) + 1}"] = {"k": k, "n": n, "share": k / n, "ci95": [float(lo), float(hi)]}
    return out


def positional_check(atlas, chunks_len: int, units: list[dict]) -> dict:
    """For each unit: share of its top-40 peaks at the single most common position, and range of peak positions."""
    out = {}
    for c in units:
        pos = [e[1] for e in cx.neuron_entries(atlas, c["layer"], c["pole"], c["neuron"])]
        cnt = Counter(pos)
        out[f'{c["layer"]}_{c["neuron"]}_{c["pole"]}'] = {"modal_position_share": max(cnt.values()) / len(pos),
                                                         "min_pos": min(pos), "max_pos": max(pos),
                                                         "n_distinct_positions": len(cnt), "chunk_len": chunks_len}
    return out
