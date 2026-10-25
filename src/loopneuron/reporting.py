"""Markdown tables rendered directly from results/<model>/*.json (no hand-copied numbers in reports).

Global context: `run_pipeline.py --phase report` writes reports/tables/*.md; narrative reports include these files'
content verbatim, so every reported number is traceable to a results file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from loopneuron import stats


def _f(x, nd=3):
    """Fixed-decimal number, or 'n/a' for None/NaN/inf."""
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def confirm_table(conf: dict, labels: dict[str, str]) -> str:
    """Markdown table of the confirmatory (or replication) results, one row per candidate."""
    rows =["| neuron (layer, index, pole) | alpha | concept (Qwen3.5 label, shortened) | mean Delta C [95% CI] | d_z | rank-biserial | "
            "p Wilcoxon (Holm) | p random-neuron null (Holm) | null mean / q95 | PPL ratio | steered distinct-2 | MiniLM Delta C | decision |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in conf["results"]:
        lab = labels.get(r["key"], "")[:90]
        rows.append(
            f'| L{r["layer"]} N{r["neuron"]} {"+" if r["pole"] == 0 else "-"} | {r["alpha"]} | {lab} | '
            f'{_f(r["mean_delta"])} [{_f(r["ci95"][0])}, {_f(r["ci95"][1])}] | {_f(r["dz"], 2)} | {_f(r["rank_biserial"], 2)} | '
            f'{r["p_wilcoxon"]:.1e} ({r["p_wilcoxon_holm"]:.1e}) | {r["p_null"]:.4f} ({r["p_null_holm"]:.4f}) | '
            f'{_f(r["null_mean"])} / {_f(r["null_q95"])} | {_f(r["ppl_ratio_median"], 2)} | {_f(r["distinct2_steered_median"], 2)} | '
            f'{_f(r["robust_mean_delta"])} | {"**significant**" if r["significant"] else "not significant"} |')
    return "\n".join(rows)


def dose_table(dose: dict) -> str:
    """Markdown dose-response table plus the measured saturation scale."""
    rows =["| alpha | clamp value | mean Delta C [95% CI] | p Wilcoxon | PPL ratio |", "|---|---|---|---|---|"]
    for r in dose["rows"]:
        rows.append(f'| {r["alpha"]} | {_f(r["clamp"], 2)} | {_f(r["mean_dC"])} [{_f(r["ci95"][0])}, {_f(r["ci95"][1])}] | '
                    f'{r["p_wilcoxon"]:.1e} | {_f(r["ppl_ratio_median"], 2)} |')
    sat = dose["saturation"]
    rows.append(f'\nSaturation scale c* per loop (median ||W_down a|| / ||w_i||): {[round(x, 2) for x in sat["c_star_by_loop"]]}; '
                f'neuron max |a| = {dose["max_act"]:.2f}, so c*/max = {np.median(sat["c_star_by_loop"]) / dose["max_act"]:.2f}.')
    return "\n".join(rows)


def locus_table(ll: dict, key: str = "loops") -> str:
    """Markdown table for loop-locus (key='loops') or T-sweep (key='T') rows."""
    rows =[f"| {key} | mean Delta C [95% CI] | p Wilcoxon | next-token KL | final-state displacement | PPL ratio |",
            "|---|---|---|---|---|---|"]
    for r in ll["rows"]:
        rows.append(f'| {r[key]} | {_f(r["mean_dC"])} [{_f(r["ci95"][0])}, {_f(r["ci95"][1])}] | {r["p_wilcoxon"]:.1e} | '
                    f'{_f(r["kl"])} | {_f(r["displacement"])} | {_f(r["ppl_ratio_median"], 2)} |')
    return "\n".join(rows)


def ablation_line(ab: dict) -> str:
    """One-sentence summary of the zero-ablation necessity test."""
    null = np.array(ab["null"])
    return (f'Zero-ablation on {ab["n_contexts"]} held-out top contexts: next-token NLL change {ab["nll_increase"]:+.4f} nats vs '
            f'random same-layer ablations mean {null.mean():.4f} (95th pct {np.quantile(null, 0.95):.4f}); empirical p = {ab["p_null"]:.4f}.')


def audit_tests(audit: list, T: int = 4) -> dict:
    """Paired two-sided Wilcoxon tests on audit.json balanced accuracies (unparsable scores, None/NaN, are dropped).

    same_minus_cross: each explanation (loop 1 and loop T) scored on its own loop minus on the other loop (H4).
    same_last_minus_shuffled: loop-T explanation minus another neuron's explanation, both on loop-T contexts (chance control).
    """
    ba = lambda d: np.nan if d.get("balanced_acc") is None else float(d["balanced_acc"])
    last = str(T - 1)  # JSON keys of the per-loop dicts are strings ("0", "3")
    diffs = {"same_minus_cross": [ba(r["same_loop"][t]) - ba(r["cross_loop"][t]) for r in audit for t in ("0", last)],
             "same_last_minus_shuffled": [ba(r["same_loop"][last]) - ba(r["shuffled"]) for r in audit]}
    out = {}
    for name, d in diffs.items():
        d = np.array(d)[np.isfinite(d)]  # a NaN on either side makes the pair unusable
        out[name] = {"mean": float(d.mean()), "n": int(len(d)), "p_two_sided": stats.wilcoxon_two_sided(d)}
    return out


def judge_holm(judge: dict) -> dict:
    """Holm-adjusted one-sided judge concept p-values across the candidate family (llm_judge.json stores raw p)."""
    return dict(zip(judge, stats.holm([v["p_concept"] for v in judge.values()]).tolist()))


def audit_judge_table(audit: list, judge: dict) -> str:
    """Markdown summary of the audit Wilcoxon tests and the Holm-adjusted LLM-judge concept p-values."""
    rows = ["| audit comparison (balanced accuracy) | mean paired difference | n pairs | p (two-sided Wilcoxon) |", "|---|---|---|---|"]
    rows += [f'| {k} | {v["mean"]:+.3f} | {v["n"]} | {v["p_two_sided"]:.2g} |' for k, v in audit_tests(audit).items()]
    rows += ["", "| neuron | judge concept unsteered -> steered | p (one-sided Wilcoxon) | p (Holm) |", "|---|---|---|---|"]
    rows += [f'| {k} | {_f(judge[k]["concept_unsteered"], 2)} -> {_f(judge[k]["concept_steered"], 2)} | '
             f'{judge[k]["p_concept"]:.2g} | {p:.2g} |' for k, p in judge_holm(judge).items()]
    return "\n".join(rows)


def write_tables(res_dir: Path, out_dir: Path) -> list[Path]:
    """Render every table whose results file exists into out_dir (reports/tables/); returns written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = lambda n: json.loads((res_dir / n).read_text(encoding="utf-8"))
    made = []
    labels = {f'{c["layer"]}_{c["neuron"]}_{c["pole"]}': c.get("label", "") for c in rd("screen.json")["candidates"]}
    if (res_dir / "confirm.json").exists():
        p = out_dir / "confirm.md"
        p.write_text(confirm_table(rd("confirm.json"), labels) + "\n", encoding="utf-8")
        made.append(p)
    if (res_dir / "followups.json").exists():
        fu = rd("followups.json")
        if "specificity" in fu:
            sp = fu["specificity"]
            M = np.array(sp["matrix"])
            hdr = "| steered (rows) / scored (columns) | " + " | ".join(sp["keys"]) + " |\n|---|" + "---|" * len(sp["keys"])
            body = "\n".join(f"| {k} | " + " | ".join(f"{v:.3f}" for v in row) + " |" for k, row in zip(sp["keys"], M))
            p = out_dir / "specificity.md"
            p.write_text(hdr + "\n" + body + "\n", encoding="utf-8")
            made.append(p)
        for key, f in fu.items():
            if key == "specificity" or "dose" not in f:
                continue
            p = out_dir / f"followups_{key}.md"
            p.write_text(f"### Dose-response\n{dose_table(f['dose'])}\n\n### Loop locus\n{locus_table(f['loop_locus'])}\n\n"
                         f"### Test-time loops\n{locus_table(f['t_sweep'], 'T')}\n\n### Ablation\n{ablation_line(f['ablation'])}\n",
                         encoding="utf-8")
            made.append(p)
    if (res_dir / "replication.json").exists():
        rep = rd("replication.json")
        p = out_dir / "replication.md"
        p.write_text(confirm_table(rep, labels) + "\n\nWeight correspondence: " + json.dumps(rep["correspondence"]) +
                     "\n\nMax activation base vs Thinking: " + json.dumps({"base": rep["base_max_act"], "thinking": rep["thinking_max_act"]}) + "\n",
                     encoding="utf-8")
        made.append(p)
    if (res_dir / "audit.json").exists() and (res_dir / "llm_judge.json").exists():
        p = out_dir / "audit_judge.md"
        p.write_text(audit_judge_table(rd("audit.json"), rd("llm_judge.json")) + "\n", encoding="utf-8")
        made.append(p)
    return made
