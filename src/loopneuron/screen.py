"""Phase 2: select steering candidates on DISCOVERY data only.

Local pipeline:
1. random pool of alive neurons (both poles) -> top-context windows -> bge embeddings -> coherence, concept direction;
   drop token detectors (one token is the peak in >= 50% of top contexts; cf. Voita et al. 2023 n-gram neurons);
2. quick steer the most coherent units at alpha in ALPHAS on the first 25 discovery prompts (1 sample, 32 tokens);
   score Delta concept score and Delta NLL (fluency) per unit;
3. keep the top K units whose best alpha passes the fluency guard (median PPL ratio <= 2).
4. `validate_candidates`: re-run the K candidates on 100 *other* discovery prompts to get selection-unbiased effect
   sizes for the power analysis (avoids the winner's curse of estimating d_z on the selection prompts).
Global context: outputs results/<model>/screen.json and candidates.json, consumed by preregister/confirm phases.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from loopneuron import contexts as cx
from loopneuron.metrics import Embedder, coherence, concept_direction, concept_scores, continuation_nll
from loopneuron.steering import Cond, clamp_value, generate

ALPHAS = (1.0, 4.0)
SCREEN_PROMPTS, SCREEN_TOKENS = 25, 32
PPL_GUARD = np.log(2.0)  # median Delta NLL <= log 2  <=>  median PPL ratio <= 2


def unit_profiles(atlas, chunks, tok, emb: Embedder, units, rand_mean, k: int = 20) -> list[dict]:
    """For each (layer, neuron, pole): coherence, peak-token share, concept direction u, example windows."""
    out = []
    texts, spans = [], []
    for layer, neuron, pole in units:
        ents = cx.neuron_entries(atlas, layer, pole, neuron, k=k)
        spans.append((len(texts), len(texts) + len(ents)))
        texts += [cx.window_text(tok, chunks, s, p) for s, p, *_ in ents]
    E = emb(texts)
    for (layer, neuron, pole), (a, b) in zip(units, spans):
        ents = cx.neuron_entries(atlas, layer, pole, neuron, k=k)
        out.append({"layer": layer, "neuron": neuron, "pole": pole, "coherence": coherence(E[a:b]),
                    "peak_share": cx.peak_token_share(chunks, ents), "u": concept_direction(E[a:b], rand_mean),
                    "max_act": float(atlas["top_v"][layer, pole, neuron, 0]),
                    "loop_of_max": [int(e[3]) for e in ents]})
    return out


def steer_effect(model, tok, emb, T, prompt_ids, base_emb, base_nll, u, cond, n_tokens, seed):
    """Per-prompt Delta concept score and Delta NLL of one condition vs the shared unsteered baseline."""
    gen = torch.from_numpy(generate(model, prompt_ids, cond, T, n_tokens, seed))
    e = emb([tok.decode(g) for g in gen])
    nll = continuation_nll(model, prompt_ids, gen)
    return concept_scores(e, u) - concept_scores(base_emb, u), nll - base_nll, [tok.decode(g) for g in gen[:3]]


def screen(model, tok, atlas, chunks, discovery, T, seed, log, n_pool: int = 4000, n_steer: int = 200, K: int = 10):
    emb = Embedder()
    rng = np.random.default_rng(seed)
    alive = np.argwhere(~atlas["dead"].numpy())  # [N, 2] (layer, neuron)
    pool = alive[rng.choice(len(alive), size=min(n_pool, len(alive)), replace=False)]
    units = [(int(l), int(n), pole) for l, n in pool for pole in (0, 1)]
    rand_texts = [cx.window_text(tok, chunks, s, p) for s, p in cx.random_entries(chunks, 2000, seed)]
    rand_mean = emb(rand_texts).mean(0)
    log.info("profiling %d units (%d neurons x 2 poles)", len(units), len(pool))
    prof = unit_profiles(atlas, chunks, tok, emb, units, rand_mean)
    ok = [p for p in prof if p["peak_share"] < 0.5]
    ok.sort(key=lambda p: -p["coherence"])
    sel = ok[:n_steer]
    log.info("token-detector filter kept %d/%d; steering top %d by coherence (min coherence %.3f)",
             len(ok), len(prof), len(sel), sel[-1]["coherence"])
    prompt_ids = torch.tensor([p["ids"] for p in discovery[:SCREEN_PROMPTS]])
    base = torch.from_numpy(generate(model, prompt_ids, None, T, SCREEN_TOKENS, seed))
    base_emb, base_nll = emb([tok.decode(g) for g in base]), continuation_nll(model, prompt_ids, base)
    for i, p in enumerate(sel):
        p["screen"] = {}
        for a in ALPHAS:
            cond = Cond(p["layer"], p["neuron"], clamp_value(atlas, p["layer"], p["pole"], p["neuron"], a))
            dC, dN, ex = steer_effect(model, tok, emb, T, prompt_ids, base_emb, base_nll, p["u"], cond, SCREEN_TOKENS, seed)
            p["screen"][a] = {"dC": float(dC.mean()), "dNLL_median": float(np.median(dN)), "examples": ex}
        best = max(p["screen"].items(), key=lambda kv: kv[1]["dC"] if kv[1]["dNLL_median"] <= PPL_GUARD else -9)
        p["best_alpha"], p["best_dC"] = best[0], (best[1]["dC"] if best[1]["dNLL_median"] <= PPL_GUARD else -9.0)
        log.info("screen %d/%d L%d N%d pole%d coh=%.3f | %s", i + 1, len(sel), p["layer"], p["neuron"], p["pole"],
                 p["coherence"], {a: (round(v["dC"], 4), round(v["dNLL_median"], 3)) for a, v in p["screen"].items()})
    cands = sorted([p for p in sel if p["best_dC"] > -9], key=lambda p: -p["best_dC"])[:K]
    for c in cands:
        log.info("CANDIDATE L%d N%d pole%d alpha=%s dC=%.4f coherence=%.3f | %s", c["layer"], c["neuron"], c["pole"],
                 c["best_alpha"], c["best_dC"], c["coherence"], c["screen"][c["best_alpha"]]["examples"][0])
    strip = lambda p: {k: v for k, v in p.items() if k != "u"}
    return {"pool": [strip(p) for p in prof], "steered": [strip(p) for p in sel], "candidates": [strip(c) for c in cands],
            "rand_mean": rand_mean.tolist(), "u": {f'{c["layer"]}_{c["neuron"]}_{c["pole"]}': c["u"].tolist() for c in cands}}


def validate_candidates(model, tok, atlas, cands, u, prompts, T, seed, n_tokens, log) -> list[dict]:
    """Effect sizes of the chosen (candidate, alpha) on fresh discovery prompts (input to the power analysis)."""
    from loopneuron.stats import cohen_dz

    emb = Embedder()
    prompt_ids = torch.tensor([p["ids"] for p in prompts])
    base = torch.from_numpy(generate(model, prompt_ids, None, T, n_tokens, seed))
    base_emb, base_nll = emb([tok.decode(g) for g in base]), continuation_nll(model, prompt_ids, base)
    out = []
    for c in cands:
        key = f'{c["layer"]}_{c["neuron"]}_{c["pole"]}'
        cond = Cond(c["layer"], c["neuron"], clamp_value(atlas, c["layer"], c["pole"], c["neuron"], c["best_alpha"]))
        dC, dN, ex = steer_effect(model, tok, emb, T, prompt_ids, base_emb, base_nll, np.array(u[key]), cond, n_tokens, seed)
        out.append({"key": key, "alpha": c["best_alpha"], "mean_dC": float(dC.mean()), "dz": cohen_dz(dC),
                    "dNLL_median": float(np.median(dN)), "n": len(dC)})
        log.info("VALIDATE %s alpha=%s mean dC=%.4f dz=%.3f median dNLL=%.3f", key, c["best_alpha"], dC.mean(),
                 out[-1]["dz"], np.median(dN))
    return out


def save(obj, path):
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
