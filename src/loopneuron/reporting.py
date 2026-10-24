"""Markdown tables rendered directly from results/<model>/*.json (no hand-copied numbers in reports).

Global context: `run_pipeline.py --phase report` writes reports/tables/*.md; narrative reports include these files'
content verbatim, so every reported number is traceable to a results file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def confirm_table(conf: dict, labels: dict[str, str]) -> str:
    rows = ["| neuron (layer, index, pole) | alpha | concept (Qwen3.5 label, shortened) | mean Delta C [95% CI] | d_z | rank-biserial | "
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
    rows = ["| alpha | clamp value | mean Delta C [95% CI] | p Wilcoxon | PPL ratio |", "|---|---|---|---|---|"]
    for r in dose["rows"]:
        rows.append(f'| {r["alpha"]} | {_f(r["clamp"], 2)} | {_f(r["mean_dC"])} [{_f(r["ci95"][0])}, {_f(r["ci95"][1])}] | '
                    f'{r["p_wilcoxon"]:.1e} | {_f(r["ppl_ratio_median"], 2)} |')
    sat = dose["saturation"]
    rows.append(f'\nSaturation scale c* per loop (median ||W_down a|| / ||w_i||): {[round(x, 2) for x in sat["c_star_by_loop"]]}; '
                f'neuron max |a| = {dose["max_act"]:.2f}, so c*/max = {np.median(sat["c_star_by_loop"]) / dose["max_act"]:.2f}.')
    return "\n".join(rows)


def locus_table(ll: dict, key: str = "loops") -> str:
    rows = [f"| {key} | mean Delta C [95% CI] | p Wilcoxon | next-token KL | final-state displacement | PPL ratio |",
            "|---|---|---|---|---|---|"]
    for r in ll["rows"]:
        rows.append(f'| {r[key]} | {_f(r["mean_dC"])} [{_f(r["ci95"][0])}, {_f(r["ci95"][1])}] | {r["p_wilcoxon"]:.1e} | '
                    f'{_f(r["kl"])} | {_f(r["displacement"])} | {_f(r["ppl_ratio_median"], 2)} |')
    return "\n".join(rows)


def ablation_line(ab: dict) -> str:
    null = np.array(ab["null"])
    return (f'Zero-ablation on {ab["n_contexts"]} held-out top contexts: next-token NLL +{ab["nll_increase"]:.4f} nats vs '
            f'random same-layer ablations mean {null.mean():.4f} (95th pct {np.quantile(null, 0.95):.4f}); empirical p = {ab["p_null"]:.4f}.')


def write_tables(res_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = lambda n: json.loads((res_dir / n).read_text(encoding="utf-8"))
    made = []
    labels = {f'{c["layer"]}_{c["neuron"]}_{c["pole"]}': c.get("label", "") for c in rd("screen.json")["candidates"]}
    if (res_dir / "confirm.json").exists():
        p = out_dir / "confirm.md"
        p.write_text(confirm_table(rd("confirm.json"), labels) + "\n", encoding="utf-8")
        made.append(p)
    if (res_dir / "followups.json").exists():
        for key, f in rd("followups.json").items():
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
    return made
