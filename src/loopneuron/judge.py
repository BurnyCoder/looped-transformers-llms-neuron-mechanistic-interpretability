"""Secondary pre-registered metric: AxBench-style LLM judge of steered vs unsteered continuations.

Local: (1) with Ouro on the GPU, regenerate sample-0 continuations for the first N confirm prompts, unsteered and for
each confirmed neuron at its pre-registered alpha (same seed base as the confirmatory test, new draws); (2) with Ouro unloaded, the
local Qwen3.5 judge rates every continuation for the neuron's concept (its Qwen label) and for fluency, 0-2, using the
AxBench prompt templates (Wu et al. 2025, https://arxiv.org/abs/2501.17148). Items are judged in a shuffled order and
each call sees one continuation only, so the judge is blind to condition. Paired Wilcoxon tests compare conditions.
Global context: phase `llm_judge` in run_pipeline.py -> results/<model>/llm_judge.json.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from loopneuron import stats
from loopneuron.autointerp import judge
from loopneuron.steering import Cond, clamp_value, generate


def generate_for_judge(model, tok, atlas, cands, prompts, T, n_tokens, seed) -> dict:
    P = torch.tensor([p["ids"] for p in prompts])
    out = {"prompts": [p["text"] for p in prompts], "unsteered": [tok.decode(g) for g in generate(model, P, None, T, n_tokens, seed)]}
    for c in cands:
        cond = Cond(c["layer"], c["neuron"], clamp_value(atlas, c["layer"], c["pole"], c["neuron"], c["alpha"]))
        out[c["key"]] = [tok.decode(g) for g in generate(model, P, cond, T, n_tokens, seed)]
    return out


def judge_all(llm, texts: dict, labels: dict[str, str], seed: int, log) -> dict:
    """Concept rating of (unsteered, steered) for each neuron's concept + fluency of every continuation."""
    keys = [k for k in texts if k not in ("prompts", "unsteered")]
    items = []  # (kind, neuron_key, condition, index)
    for k in keys:
        for i in range(len(texts["prompts"])):
            items += [("concept", k, "unsteered", i), ("concept", k, k, i), ("fluency", k, k, i)]
    items += [("fluency", "-", "unsteered", i) for i in range(len(texts["prompts"]))]
    random.Random(seed).shuffle(items)  # interleave conditions so the judge's context never reveals the design
    scores: dict = {}
    for n, (kind, k, cond, i) in enumerate(items):
        sentence = texts["prompts"][i] + texts[cond][i]
        s = judge(llm, sentence, concept=labels[k] if kind == "concept" else None, tag=f"{kind} item {n}")
        scores[(kind, k, cond, i)] = np.nan if s is None else s
    out = {}
    N = len(texts["prompts"])
    flu_base = np.array([scores[("fluency", "-", "unsteered", i)] for i in range(N)])
    for k in keys:
        cb = np.array([scores[("concept", k, "unsteered", i)] for i in range(N)])
        cs = np.array([scores[("concept", k, k, i)] for i in range(N)])
        fs = np.array([scores[("fluency", k, k, i)] for i in range(N)])
        ok = np.isfinite(cb) & np.isfinite(cs)
        okf = np.isfinite(fs) & np.isfinite(flu_base)
        out[k] = {"concept_unsteered": float(np.nanmean(cb)), "concept_steered": float(np.nanmean(cs)),
                  "p_concept": stats.wilcoxon_greater(cs[ok] - cb[ok]),
                  "fluency_unsteered": float(np.nanmean(flu_base)), "fluency_steered": float(np.nanmean(fs)),
                  "fluency_delta_ci95": list(stats.bootstrap_ci(fs[okf] - flu_base[okf])) if okf.sum() > 1 else None,
                  "n": int(ok.sum()), "n_unparsed": int((~np.isfinite(cs)).sum() + (~np.isfinite(cb)).sum())}
        log.info("JUDGE %s concept %.2f -> %.2f (p=%.1e) fluency %.2f -> %.2f", k, out[k]["concept_unsteered"],
                 out[k]["concept_steered"], out[k]["p_concept"], out[k]["fluency_unsteered"], out[k]["fluency_steered"])
    return out
