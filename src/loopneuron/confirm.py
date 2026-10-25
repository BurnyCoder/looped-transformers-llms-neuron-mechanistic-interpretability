"""Phase 4: pre-registered confirmatory steering test on held-out prompts (see reports/preregistration.md).

Local, per candidate (layer, neuron, pole, alpha frozen in candidates.json before this runs):
- k paired samples per prompt, steered vs unsteered (same seeds) -> per-prompt Delta concept score (mean over k);
- one-sided Wilcoxon signed-rank vs 0;
- random-neuron null: R alive neurons from the same layer clamped at the same alpha x their own max on the same pole,
  generated on the first N_NULL prompts (sample 0) and scored with the CANDIDATE's concept direction; empirical p of
  the candidate's mean Delta C on the same prompts/sample (Goodfire's random same-layer control, made quantitative);
- fluency guard (median continuation PPL ratio under the unsteered model), distinct-2, robustness embedder.
Holm correction across candidates for each p-value family; decision rule from the pre-registration.
Global context: writes results/<model>/confirm.json + confirm_examples.jsonl used by reports and figures.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from loopneuron import contexts as cx
from loopneuron import stats
from loopneuron.metrics import ROBUST_EMBEDDER, Embedder, concept_direction, concept_scores, continuation_nll, distinct2
from loopneuron.screen import DISTINCT2_GUARD
from loopneuron.steering import Cond, clamp_value, generate, generate_samples, random_same_layer

N_NULL = 50


def _scores(emb, tok, gen: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Concept score of every continuation in gen [..., L] (shape preserved without the token axis)."""
    flat = gen.reshape(-1, gen.shape[-1])
    return concept_scores(emb([tok.decode(g) for g in flat]), u).reshape(gen.shape[:-1])


def confirm(model, tok, atlas, chunks, cands, u, rand_mean, prompts, T, seed, k, n_tokens, R, log, out_dir,
            examples_name: str = "confirm_examples.jsonl") -> dict:
    """Run the frozen confirmatory protocol for `cands`; returns per-candidate statistics and Holm decisions."""
    emb, emb2 = Embedder(), Embedder(ROBUST_EMBEDDER)
    rand2 = emb2([cx.window_text(tok, chunks, s, p) for s, p in cx.random_entries(chunks, 2000, seed)]).mean(0)
    P = torch.tensor([p["ids"] for p in prompts])
    log.info("confirm: %d prompts x %d samples x %d tokens, %d candidates, R=%d", len(P), k, n_tokens, len(cands), R)
    base = generate_samples(model, P, None, T, n_tokens, seed, k)  # [n, k, L]
    base_nll = np.stack([continuation_nll(model, P, torch.from_numpy(base[:, j])) for j in range(k)], 1)
    null_cache: dict = {}
    results, examples = [], []
    for c in cands:
        key = f'{c["layer"]}_{c["neuron"]}_{c["pole"]}'
        ents = cx.neuron_entries(atlas, c["layer"], c["pole"], c["neuron"], k=20)
        u1 = np.array(u[key])
        u2 = concept_direction(emb2([cx.window_text(tok, chunks, s, p) for s, p, *_ in ents]), rand2)
        cond = Cond(c["layer"], c["neuron"], clamp_value(atlas, c["layer"], c["pole"], c["neuron"], c["alpha"]))
        gen = generate_samples(model, P, cond, T, n_tokens, seed, k)
        dC = (_scores(emb, tok, gen, u1) - _scores(emb, tok, base, u1)).mean(1)  # per-prompt, averaged over k
        dC2 = (_scores(emb2, tok, gen, u2) - _scores(emb2, tok, base, u2)).mean(1)
        nll = np.stack([continuation_nll(model, P, torch.from_numpy(gen[:, j])) for j in range(k)], 1)
        ppl_ratio = np.exp((nll - base_nll).mean(1))
        d2_steered = np.array([[distinct2(g.tolist()) for g in row] for row in gen]).mean(1)
        d2 = d2_steered - np.array([[distinct2(g.tolist()) for g in row] for row in base]).mean(1)
        # random same-layer neuron null (shared generations per (layer, pole, alpha); rescored per candidate)
        nkey = (c["layer"], c["pole"], c["alpha"])
        if nkey not in null_cache:
            rs = random_same_layer(atlas, c["layer"], c["neuron"], R, seed + c["layer"])
            null_cache[nkey] = {"neurons": rs, "gens": [generate(model, P[:N_NULL], Cond(
                c["layer"], r, clamp_value(atlas, c["layer"], c["pole"], r, c["alpha"])), T, n_tokens, seed) for r in rs]}
            log.info("null generations done for layer %d pole %d alpha %s (%d neurons)", *nkey, R)
        b0 = _scores(emb, tok, base[:N_NULL, 0], u1)
        null_means = np.array([(_scores(emb, tok, g, u1) - b0).mean() for g in null_cache[nkey]["gens"]])
        obs_null_stat = float((_scores(emb, tok, gen[:N_NULL, 0], u1) - b0).mean())
        res = {"key": key, "layer": c["layer"], "neuron": c["neuron"], "pole": c["pole"], "alpha": c["alpha"],
               "clamp_value": cond.value, "label": c.get("label", ""), **stats.summarize(dC, null_means, seed),
               "p_null": stats.perm_p(obs_null_stat, null_means), "null_stat_candidate": obs_null_stat,
               "null_means": null_means.tolist(), "null_neurons": null_cache[nkey]["neurons"],
               "robust_mean_delta": float(dC2.mean()), "robust_p_wilcoxon": stats.wilcoxon_greater(dC2),
               "ppl_ratio_median": float(np.median(ppl_ratio)), "distinct2_delta_mean": float(d2.mean()),
               "distinct2_steered_median": float(np.median(d2_steered)),
               "delta_per_prompt": dC.tolist()}
        results.append(res)
        log.info("CONFIRM %s alpha=%s meanDC=%.4f CI=%s p_wil=%.2e p_null=%.4f ppl_ratio=%.2f robust=%.4f",
                 key, c["alpha"], res["mean_delta"], np.round(res["ci95"], 4), res["p_wilcoxon"], res["p_null"],
                 res["ppl_ratio_median"], res["robust_mean_delta"])
        for i in range(min(8, len(P))):
            examples.append({"key": key, "prompt": prompts[i]["text"], "unsteered": tok.decode(base[i, 0]),
                             "steered": tok.decode(gen[i, 0])})
    for fam in ("p_wilcoxon", "p_null"):
        for r, a in zip(results, stats.holm([r[fam] for r in results])):
            r[fam + "_holm"] = float(a)
    for r in results:
        r["significant"] = bool(r["p_wilcoxon_holm"] < 0.05 and r["p_null_holm"] < 0.05 and r["ppl_ratio_median"] <= 2
                                and r["distinct2_steered_median"] >= DISTINCT2_GUARD)
        log.info("DECISION %s: %s (holm p_wil=%.2e, holm p_null=%.4f, ppl=%.2f, distinct2=%.2f)", r["key"],
                 "SIGNIFICANT" if r["significant"] else "not significant", r["p_wilcoxon_holm"], r["p_null_holm"],
                 r["ppl_ratio_median"], r["distinct2_steered_median"])
    with open(out_dir / examples_name, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e) + "\n" for e in examples)
    return {"n_prompts": len(P), "k": k, "n_tokens": n_tokens, "R": R, "n_null_prompts": N_NULL, "seed": seed,
            "results": results}
