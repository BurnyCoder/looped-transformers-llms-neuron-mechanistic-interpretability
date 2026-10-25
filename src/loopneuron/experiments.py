"""Follow-up causal experiments on confirmed (or top) neurons: dose-response, loop locus (H2), test-time loops (H3),
zero-ablation necessity. All are secondary analyses run after the pre-registered test (labeled as such in reports).

Local:
- dose_response: Delta concept score and PPL ratio over alpha (clamp = alpha x max|a| on the pole), with the
  sandwich-norm saturation scale c* = median ||W_down a|| / ||w_i|| measured on the prompts (docs/methodology.md sec. 2).
- loop_locus (H2): clamp only at loop t; behavioral Delta C, next-token KL on the prompts, and relative displacement
  of the final hidden state. Prior residual-direction steering of Ouro found the last loop most effective
  (nesiacel 2026, https://www.lesswrong.com/posts/rzcJMpnehFBD686gj/).
- t_sweep (H3): clamp at all loops while running T = 1..8 loops (T > 4 is extrapolation), each vs its own baseline.
- ablation: set the neuron to 0 at all loops on its held-out top contexts (atlas ranks 21-40) and measure the NLL
  increase of the token after the peak, vs R random same-layer ablations (Goodfire-style random control).
Global context: run_pipeline.py phase `followups`; outputs feed figures and reports.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from loopneuron import contexts as cx
from loopneuron import stats
from loopneuron.hooks import Edit, NeuronHooks
from loopneuron.metrics import Embedder, concept_scores, continuation_nll
from loopneuron.model import set_ut_steps
from loopneuron.steering import Cond, clamp_value, generate, random_same_layer

ALPHA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, -1.0, -4.0)  # negative alpha = opposite-sign clamp (exploratory)


def _delta_c(model, tok, emb, P, cond, base_s, u, T, n_tokens, seed):
    """Per-prompt Delta concept score of one condition vs precomputed unsteered scores, plus continuation NLL."""
    gen = generate(model, P, cond, T, n_tokens, seed)
    s = concept_scores(emb([tok.decode(g) for g in gen]), u)
    nll = continuation_nll(model, P, torch.from_numpy(gen))
    return s - base_s, nll


def _row(d, nll_ratio, **kw):
    """Summary row: mean Delta C with bootstrap CI, one-sided Wilcoxon p, median PPL ratio, plus extra fields."""
    lo, hi = stats.bootstrap_ci(d, n_boot=2000)
    return {**kw, "mean_dC": float(d.mean()), "ci95": [lo, hi], "p_wilcoxon": stats.wilcoxon_greater(d),
            "ppl_ratio_median": float(np.median(nll_ratio))}


def specificity_matrix(model, tok, atlas, cands: list[dict], u: dict, P, T, n_tokens, seed, log) -> dict:
    """Post hoc: M[i, j] = mean Delta C on concept j when steering neuron i (diagonal = intended concept).

    A neuron whose effect is concept-specific raises its own concept score more than other neurons' concepts.
    """
    emb = Embedder()
    U = np.stack([np.array(u[c["key"]]) for c in cands])  # [K, dim]
    base = generate(model, P, None, T, n_tokens, seed)
    base_e = emb([tok.decode(g) for g in base])
    M = np.zeros((len(cands), len(cands)))
    for i, c in enumerate(cands):
        val = clamp_value(atlas, c["layer"], c["pole"], c["neuron"], c["alpha"])
        e = emb([tok.decode(g) for g in generate(model, P, Cond(c["layer"], c["neuron"], val), T, n_tokens, seed)])
        M[i] = ((e - base_e) @ U.T).mean(0)
        log.info("SPECIFICITY %s -> %s", c["key"], np.round(M[i], 3).tolist())
    return {"keys": [c["key"] for c in cands], "matrix": M.tolist()}


@torch.no_grad()
def saturation_scale(model, P, layer: int, neuron: int, T: int) -> dict:
    """c* per loop = median over tokens of ||W_down a|| / ||w_i|| (scale where one clamped neuron dominates the MLP)."""
    norms = {t: [] for t in range(T)}
    calls = {"n": 0}
    w = model.model.layers[layer].mlp.down_proj.weight[:, neuron].float().norm().item()

    def hook(m, args, out):
        """Forward hook on down_proj: record ||W_down a|| per token for the current loop."""
        norms[calls["n"] % T].append(out.float().norm(dim=-1).flatten().cpu())
        calls["n"] += 1

    h = model.model.layers[layer].mlp.down_proj.register_forward_hook(hook)
    model(P.cuda(), use_cache=False)
    h.remove()
    return {"w_norm": w, "c_star_by_loop": [float(torch.cat(norms[t]).median() / w) for t in range(T)]}


def dose_response(model, tok, atlas, cand, u, P, T, n_tokens, seed, log) -> dict:
    """Delta C and PPL ratio for every alpha in ALPHA_GRID, plus the measured saturation scale c*."""
    emb = Embedder()
    base = generate(model, P, None, T, n_tokens, seed)
    base_s = concept_scores(emb([tok.decode(g) for g in base]), u)
    base_nll = continuation_nll(model, P, torch.from_numpy(base))
    rows = []
    for a in ALPHA_GRID:
        pole_val = clamp_value(atlas, cand["layer"], cand["pole"], cand["neuron"], abs(a)) * np.sign(a)
        d, nll = _delta_c(model, tok, emb, P, Cond(cand["layer"], cand["neuron"], pole_val), base_s, u, T, n_tokens, seed)
        rows.append(_row(d, np.exp(nll - base_nll), alpha=a, clamp=pole_val))
        log.info("DOSE %s alpha=%s meanDC=%.4f ppl=%.2f", cand["key"], a, rows[-1]["mean_dC"], rows[-1]["ppl_ratio_median"])
    return {"rows": rows, "saturation": saturation_scale(model, P, cand["layer"], cand["neuron"], T),
            "max_act": float(atlas["top_v"][cand["layer"], cand["pole"], cand["neuron"], 0])}


@torch.no_grad()
def _mech_effect(model, P, edits, T):
    """Next-token KL(steered || clean) and relative final-state displacement on prompt positions."""
    clean = model(P.cuda(), use_cache=False, output_hidden_states=False)
    h_clean = model.model(P.cuda(), use_cache=False)[0].last_hidden_state.float()
    with NeuronHooks(model, T, edits=edits):
        st = model(P.cuda(), use_cache=False)
        h_st = model.model(P.cuda(), use_cache=False)[0].last_hidden_state.float()
    kl = F.kl_div(clean.logits.float().log_softmax(-1), st.logits.float().log_softmax(-1), log_target=True,
                  reduction="none").sum(-1).mean().item()
    disp = ((h_st - h_clean).norm(dim=-1) / h_clean.norm(dim=-1)).mean().item()
    return kl, disp


def loop_locus(model, tok, atlas, cand, u, P, T, n_tokens, seed, log) -> dict:
    """H2: behavioral and mechanistic effect of clamping at all loops vs at each single loop."""
    emb = Embedder()
    base = generate(model, P, None, T, n_tokens, seed)
    base_s = concept_scores(emb([tok.decode(g) for g in base]), u)
    base_nll = continuation_nll(model, P, torch.from_numpy(base))
    val = clamp_value(atlas, cand["layer"], cand["pole"], cand["neuron"], cand["alpha"])
    rows = []
    for loops in [None] + [(t,) for t in range(T)]:
        cond = Cond(cand["layer"], cand["neuron"], val, loops)
        d, nll = _delta_c(model, tok, emb, P, cond, base_s, u, T, n_tokens, seed)
        kl, disp = _mech_effect(model, P, cond.edits(), T)
        rows.append(_row(d, np.exp(nll - base_nll), loops="all" if loops is None else loops[0] + 1, kl=kl, displacement=disp))
        log.info("LOOP %s loops=%s meanDC=%.4f KL=%.4f disp=%.4f", cand["key"], rows[-1]["loops"], rows[-1]["mean_dC"], kl, disp)
    return {"rows": rows}


def t_sweep(model, tok, atlas, cand, u, P, T_train, n_tokens, seed, log, T_max: int = 8) -> dict:
    """H3: clamp at every loop while running T = 1..T_max loops, each compared with its own unsteered baseline."""
    emb = Embedder()
    val = clamp_value(atlas, cand["layer"], cand["pole"], cand["neuron"], cand["alpha"])
    rows = []
    try:
        for t in range(1, T_max + 1):
            set_ut_steps(model, t)
            base = generate(model, P, None, t, n_tokens, seed)
            base_s = concept_scores(emb([tok.decode(g) for g in base]), u)
            base_nll = continuation_nll(model, P, torch.from_numpy(base))
            d, nll = _delta_c(model, tok, emb, P, Cond(cand["layer"], cand["neuron"], val), base_s, u, t, n_tokens, seed)
            kl, disp = _mech_effect(model, P, Cond(cand["layer"], cand["neuron"], val).edits(), t)
            rows.append(_row(d, np.exp(nll - base_nll), T=t, kl=kl, displacement=disp))
            log.info("TSWEEP %s T=%d meanDC=%.4f KL=%.4f", cand["key"], t, rows[-1]["mean_dC"], kl)
    finally:
        set_ut_steps(model, T_train)
    return {"rows": rows}


@torch.no_grad()
def _ablation_nll_increase(model, ids, nxt_pos, layer, neurons, T) -> np.ndarray:
    """Mean NLL increase of the token after each peak when `neurons` (one per row block) are zero-ablated."""
    def nll(edits):
        """NLL of the token after each peak position under the given edits."""
        with NeuronHooks(model, T, edits=edits):
            logp = model(ids, use_cache=False).logits.float().log_softmax(-1)
        rows = torch.arange(len(ids), device=ids.device)
        return -logp[rows, nxt_pos - 1, ids[rows, nxt_pos]]

    clean = nll({})
    out = []
    for n in neurons:
        out.append((nll({layer: [Edit.single(n, 0.0, "scale")]}) - clean).mean().item())
    return np.array(out)


def ablation(model, atlas, chunks, cand, T, R, seed, log) -> dict:
    """Necessity: zero-ablate the neuron on held-out top contexts vs R random same-layer ablations (empirical p)."""
    ents =[e for e in cx.neuron_entries(atlas, cand["layer"], cand["pole"], cand["neuron"])[20:40]
            if e[1] + 1 < chunks.shape[1]]  # held-out top contexts (not used for the concept direction)
    ids = torch.from_numpy(chunks[[e[0] for e in ents]].astype(np.int64)).cuda()
    nxt = torch.tensor([e[1] + 1 for e in ents], device="cuda")
    obs = _ablation_nll_increase(model, ids, nxt, cand["layer"], [cand["neuron"]], T)[0]
    null = _ablation_nll_increase(model, ids, nxt, cand["layer"],
                                  random_same_layer(atlas, cand["layer"], cand["neuron"], R, seed), T)
    p = stats.perm_p(obs, null)
    log.info("ABLATION %s: NLL increase %.4f nats vs null mean %.4f (q95 %.4f), p=%.4f", cand["key"], obs, null.mean(),
             np.quantile(null, 0.95), p)
    return {"nll_increase": float(obs), "null": null.tolist(), "p_null": p, "n_contexts": len(ents)}
