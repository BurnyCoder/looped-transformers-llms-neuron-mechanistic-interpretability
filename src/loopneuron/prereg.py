"""Phase 3: freeze the confirmatory analysis (candidates, alpha, n, tests, decision rule) before seeing confirm data.

Local: the top 2K screened units are validated on fresh discovery prompts; a unit is eligible only if validation shows
mean Delta C > 0, median Delta NLL <= log 2 and median steered distinct-2 >= 0.6 (repetitive degeneration such as
"Fib Fib Fib" has low perplexity, so perplexity alone does not catch it); the K eligible units with the largest
validation d_z are kept. n (prompts) = clip(max_i required_n(d_z,i, 0.05/K, power 0.9), 100, 300)
using the validation effect sizes (not the selection-biased screening ones). Writes results/<model>/candidates.json
(machine-readable) and reports/preregistration.md (human-readable, committed to git before confirm_steering runs).
Global context: confirm_steering refuses to run unless this file is committed and unmodified.
"""
from __future__ import annotations

import json
import time

import numpy as np

from statsmodels.stats.power import TTestPower

from loopneuron.screen import DISTINCT2_GUARD
from loopneuron.stats import WILCOXON_ARE, required_n


def write_preregistration(screen: dict, s, out_dir, md_path, log) -> dict:
    val = {v["key"]: v for v in screen["validation"]}
    kept = []
    for c in screen["candidates"]:
        key = f'{c["layer"]}_{c["neuron"]}_{c["pole"]}'
        v = val[key]
        if v["mean_dC"] > 0 and v["dNLL_median"] <= np.log(2) and v["distinct2_median"] >= DISTINCT2_GUARD:
            kept.append({"key": key, "layer": c["layer"], "neuron": c["neuron"], "pole": c["pole"],
                         "alpha": c["best_alpha"], "label": c.get("label", ""), "coherence": c["coherence"],
                         "validation_dz": v["dz"], "validation_mean_dC": v["mean_dC"],
                         "validation_distinct2": v["distinct2_median"]})
    kept = sorted(kept, key=lambda c: -c["validation_dz"])[: s.n_candidates]
    K = max(1, len(kept))
    p_min = 1 / (s.null_neurons + 1)  # smallest attainable empirical p with R random neurons
    if p_min > 0.05 / K:
        raise ValueError(f"R={s.null_neurons} cannot reach Holm threshold 0.05/{K}; raise NULL_NEURONS or lower K")
    req = [required_n(c["validation_dz"], 0.05 / K, 0.9) for c in kept]
    n = int(np.clip(max(req) if req else 300, 100, 300))
    for c, r in zip(kept, req):
        c["required_n"] = r
        c["power_at_n"] = float(TTestPower().power(effect_size=c["validation_dz"], nobs=n * WILCOXON_ARE,
                                                   alpha=0.05 / K, alternative="larger"))
    pre = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "model": s.model_id, "revision": s.model_revision,
           "T": s.ut_steps, "candidates": kept, "K": len(kept), "n_prompts": n, "k": s.samples_per_prompt,
           "n_tokens": s.gen_tokens, "R": s.null_neurons, "seed": s.seed + 1000, "alpha_family": 0.05}
    (out_dir / "candidates.json").write_text(json.dumps(pre, indent=1), encoding="utf-8")
    rows = "\n".join(f'| {c["key"]} | {c["layer"]} | {c["neuron"]} | {"+" if c["pole"] == 0 else "-"} | {c["alpha"]} | '
                     f'{c["coherence"]:.3f} | {c["validation_mean_dC"]:.4f} | {c["validation_dz"]:.2f} | {c["validation_distinct2"]:.2f} | '
                     f'{c["required_n"]} | {c["power_at_n"]:.2f} | {c["label"]} |' for c in kept)
    md = f"""# Pre-registration: confirmatory neuron-steering test on {s.model_id}

Written {pre["created"]} by `run_pipeline.py --phase preregister`, **before** any confirmatory prompt was used.
Machine-readable copy: `results/{out_dir.name}/candidates.json`. Methods: [docs/methodology.md](../docs/methodology.md).

## Hypothesis H1 (primary)
At least one of the K = {len(kept)} candidate neurons below, clamped to `alpha x max|a|` on its pole at every position and
all {s.ut_steps} loops, increases the concept score of 48-token continuations relative to (a) unsteered generation and (b) the
random same-layer neuron null.

## Frozen design
- Model `{s.model_id}` @ `{s.model_revision}`, T = {s.ut_steps} loops, exit at the last loop, bf16.
- Prompts: the first **n = {n}** stems of `data/prompts/confirm.jsonl` (WikiText-103 test, 12 tokens; never used before).
- k = {s.samples_per_prompt} samples per prompt, {s.gen_tokens} new tokens, temperature 0.7, top-p 0.9, seed base {s.seed + 1000};
  steered and unsteered generations share seeds and batch layout (paired).
- Primary metric: per-prompt mean over k of Delta C, C = cosine(bge-small-en-v1.5 embedding of the continuation, the
  neuron's contrastive concept direction computed from its top-20 atlas contexts minus 2000 random atlas contexts).
- Test 1: one-sided Wilcoxon signed-rank on per-prompt Delta C (H1: > 0).
- Test 2: random-neuron null: R = {s.null_neurons} alive neurons from the candidate's layer, clamped at the same alpha x their
  own max on the same pole; statistic = mean Delta C over the first 50 prompts, sample 0, scored with the candidate's
  concept direction; p = (1 + #null >= observed)/(R + 1).
- Multiplicity: Holm across the K candidates, separately for Test 1 and Test 2, family-wise alpha = 0.05
  (smallest attainable Test-2 p = 1/(R+1) = {1 / (s.null_neurons + 1):.4f} <= 0.05/K = {0.05 / max(1, len(kept)):.4f}).
- **Decision rule:** a candidate is a "significant steering neuron" iff Holm p (Test 1) < 0.05 AND Holm p (Test 2) < 0.05
  AND median per-prompt continuation-perplexity ratio (steered / unsteered, both scored by the unsteered model) <= 2
  AND median distinct-2 of steered continuations >= {DISTINCT2_GUARD} (non-degenerate text).
- Secondary (reported, not part of the decision): all-MiniLM-L6-v2 concept score, distinct-2 change, LLM judge
  (AxBench concept/fluency 0-2), dose-response over alpha in {{0.5, 1, 2, 4, 8, 16}}, zero-ablation, loop-restricted steering.
- Falsification: if no candidate meets the decision rule, H1 is rejected for this protocol. At most one further
  pre-registered round may be run on `data/prompts/reserve.jsonl`.

## Candidates (selected and validated on discovery prompts only)
Selection: 8,000 units (4,000 random alive neurons x 2 poles) -> token-detector filter -> 200 most coherent -> quick
steer (discovery prompts 1-25, alpha in {{1, 2, 4}}) -> top {2 * s.n_candidates} by screen Delta C -> validation on
discovery prompts 26-125 -> eligible if mean Delta C > 0, median Delta NLL <= log 2, median distinct-2 >= {DISTINCT2_GUARD}
-> the K = {len(kept)} eligible units with the largest validation d_z.
Power: required n per candidate for one-sided Wilcoxon (t-test power / ARE 0.955) at alpha = 0.05/K, power 0.9, using the
validation d_z (100 fresh discovery prompts, 1 sample).

| key | layer | neuron | pole | alpha | coherence | validation mean Delta C | validation d_z | validation distinct-2 | required n | power at n | label (local Qwen3.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|
{rows}
"""
    md_path.write_text(md, encoding="utf-8")
    log.info("pre-registration written: K=%d n=%d -> %s (commit + push before confirm_steering)", len(kept), n, md_path)
    return pre
