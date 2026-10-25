"""Figures (static PNG, light mode) generated only from results/<model>/*.json.

Local: one function per figure; style follows the bundled dataviz method: validated categorical slots in fixed order
(blue #2a78d6, orange #eb6834, aqua #1baf7a -- the three slots that pass all-pairs CVD checks), neutral gray for
baselines/nulls, thin marks, recessive grid, one y-axis per panel (never dual axes), uncertainty always shown.
Matplotlib docs: https://matplotlib.org/stable/
Global context: phase `report` calls `all_figures`; README/reports/paper embed the PNGs from figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np

S1, S2, S3, GRAY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b86", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.8, "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False, "axes.axisbelow": True,
})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def loss_by_T(setup: dict, path: Path, T_train: int = 4):
    T = sorted(int(t) for t in setup["loss_by_T"])
    y = [setup["loss_by_T"][str(t)] if str(t) in setup["loss_by_T"] else setup["loss_by_T"][t] for t in T]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(T, y, color=S1, marker="o")
    ax.axvline(T_train, color=GRAY, ls="--", lw=1)
    ax.text(T_train + 0.1, max(y), f"trained T = {T_train}", color=INK2, va="top")
    ax.set(xlabel="loops T at inference", ylabel="held-out loss (nats/token)", title="Ouro-1.4B: loss vs number of loops")
    return _save(fig, path)


def loop_structure(ls: dict, path: Path):
    share = np.array(ls["loop_of_max_share"])  # [L, T]
    has_corr = "corr_first_last_median" in ls
    fig, axes = plt.subplots(1, 3 if has_corr else 2, figsize=(15 if has_corr else 10, 3.6))
    im = axes[0].imshow(share.T, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower")
    axes[0].set(xlabel="layer", ylabel="loop", yticks=range(share.shape[1]), yticklabels=range(1, share.shape[1] + 1),
                title="share of neurons peaking at loop t")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    L = range(len(ls["jaccard_first_last"]))
    adj = np.array(ls["jaccard_adjacent"])  # [L, T-1]
    for t, colr in zip(range(adj.shape[1]), (S1, S2, S3)):
        axes[1].plot(L, adj[:, t], color=colr, marker="o", ms=3, label=f"loop {t + 1} vs {t + 2}")
    axes[1].plot(L, ls["jaccard_first_last"], color=GRAY, marker="o", ms=3, label="loop 1 vs 4")
    axes[1].set(xlabel="layer", ylabel="Jaccard of top-20 contexts", ylim=(0, 1), title="top-context overlap between loops")
    axes[1].legend(fontsize=8)
    if has_corr:
        cadj = np.array(ls["corr_adjacent_median"])  # [T-1, L]
        for t, colr in zip(range(cadj.shape[0]), (S1, S2, S3)):
            axes[2].plot(L, cadj[t], color=colr, marker="o", ms=3, label=f"loop {t + 1} vs {t + 2}")
        axes[2].plot(L, ls["corr_first_last_median"], color=GRAY, marker="o", ms=3, label="loop 1 vs 4")
        axes[2].set(xlabel="layer", ylabel="median Pearson r over neurons", ylim=(-0.2, 1),
                    title="same-token activation correlation between loops")
        axes[2].legend(fontsize=8)
    return _save(fig, path)


def confirm_effects(conf: dict, path: Path):
    rs = conf["results"]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rs) + 2.0))
    for i, r in enumerate(rs):
        lo, hi = r["ci95"]
        ax.plot([lo, hi], [i, i], color=S1 if r["significant"] else GRAY, lw=2)
        ax.plot(r["mean_delta"], i, "o", color=S1 if r["significant"] else GRAY)
        ax.plot(r["null_q95"], i, "|", color=S2, ms=12, mew=2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rs)), [f'L{r["layer"]} N{r["neuron"]} {"+" if r["pole"] == 0 else "−"}  α={r["alpha"]:g}' for r in rs])
    ax.set(xlabel="mean ΔC (steered − unsteered), 95% bootstrap CI", title="Pre-registered confirmatory test")
    ax.plot([], [], "o-", color=S1, label="candidate (significant)")
    ax.plot([], [], "|", color=S2, ms=12, mew=2, label="random same-layer null, 95th pct.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    return _save(fig, path)


def null_histogram(r: dict, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(r["null_means"], bins=25, color=GRAY, edgecolor=SURF, linewidth=1)
    ax.axvline(r["null_stat_candidate"], color=S1, lw=2.5)
    ax.text(r["null_stat_candidate"], ax.get_ylim()[1] * 0.95, f' L{r["layer"]} N{r["neuron"]}\n p = {r["p_null"]:.3f}',
            color=INK, va="top")
    ax.set(xlabel="mean ΔC on 50 prompts", ylabel="random same-layer neurons",
           title=f"Candidate vs {len(r['null_means'])} random neurons (same layer, same α)")
    return _save(fig, path)


def dose(dr: dict, key: str, path: Path):
    rows = [r for r in dr["rows"] if r["alpha"] > 0]
    a = np.array([r["alpha"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    m = np.array([r["mean_dC"] for r in rows])
    ci = np.array([r["ci95"] for r in rows])
    axes[0].fill_between(a, ci[:, 0], ci[:, 1], color=S1, alpha=0.18, lw=0)
    axes[0].plot(a, m, color=S1, marker="o")
    cstar = np.median(dr["saturation"]["c_star_by_loop"]) / dr["max_act"]
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.axvline(cstar, color=S2, ls="--", lw=1.5)
    axes[0].text(cstar, axes[0].get_ylim()[1], " predicted saturation c*", color=INK2, va="top")
    axes[0].set(xlabel="α (clamp = α × max activation)", ylabel="mean ΔC", title=f"Dose-response {key}")
    axes[1].plot(a, [r["ppl_ratio_median"] for r in rows], color=S2, marker="o")
    axes[1].axhline(2, color=GRAY, ls=":", lw=1)
    axes[1].set(xlabel="α", ylabel="median perplexity ratio", title="Fluency cost (guard at 2)")
    return _save(fig, path)


def loop_locus(ll: dict, key: str, path: Path):
    rows = ll["rows"]
    labels = [str(r["loops"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    m, ci = np.array([r["mean_dC"] for r in rows]), np.array([r["ci95"] for r in rows])
    axes[0].bar(labels, m, color=[GRAY] + [S1] * (len(rows) - 1), width=0.6)
    axes[0].errorbar(labels, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt="none", ecolor=INK, lw=1.2, capsize=3)
    axes[0].set(xlabel="loop(s) where the neuron is clamped", ylabel="mean ΔC", title=f"Loop locus {key}")
    axes[1].bar(labels, [r["kl"] for r in rows], color=[GRAY] + [S3] * (len(rows) - 1), width=0.6)
    axes[1].set(xlabel="loop(s) clamped", ylabel="next-token KL (nats)", title="Effect on next-token distribution")
    for ax in axes:
        ax.grid(axis="x", visible=False)  # bars: horizontal grid only
    return _save(fig, path)


def t_sweep(ts: dict, key: str, path: Path, T_train: int = 4):
    rows = ts["rows"]
    T = [r["T"] for r in rows]
    m, ci = np.array([r["mean_dC"] for r in rows]), np.array([r["ci95"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].fill_between(T, ci[:, 0], ci[:, 1], color=S1, alpha=0.18, lw=0)
    axes[0].plot(T, m, color=S1, marker="o")
    axes[1].plot(T, [r["kl"] for r in rows], color=S3, marker="o")
    for ax in axes:
        ax.axvline(T_train, color=GRAY, ls="--", lw=1)
    axes[0].set(xlabel="loops T at inference (clamped at every loop)", ylabel="mean ΔC",
                title=f"Test-time loops {key}")
    axes[1].set(xlabel="loops T", ylabel="next-token KL (nats)", title="Mechanistic effect vs T")
    return _save(fig, path)


def specificity(sp: dict, path: Path):
    M = np.array(sp["matrix"])
    keys = [f"L{k.split('_')[0]} N{k.split('_')[1]}{'+' if k.split('_')[2] == '0' else '-'}" for k in sp["keys"]]
    fig, ax = plt.subplots(figsize=(1.0 * len(keys) + 2.5, 0.8 * len(keys) + 1.8))
    lim = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)  # diverging: blue negative, red positive, white 0
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.6 * lim else INK)  # keep labels legible on dark cells
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right")
    ax.set_yticks(range(len(keys)), keys)
    ax.set(xlabel="concept direction scored", ylabel="neuron steered", title="Steering specificity (mean ΔC)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def audit_bars(audit: list, path: Path, T: int = 4):
    def col(kind, t):
        return np.array([r[kind][str(t)]["balanced_acc"] if str(t) in r[kind] else r[kind][t]["balanced_acc"] for r in audit])
    groups = {"same loop (1)": col("same_loop", 0), "same loop (4)": col("same_loop", T - 1),
              "cross 1->4": col("cross_loop", 0), "cross 4->1": col("cross_loop", T - 1),
              "shuffled label": np.array([r["shuffled"]["balanced_acc"] for r in audit])}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = list(groups)
    for i, n in enumerate(names):  # dot + CI (not bars): the axis does not start at 0
        v = groups[n][np.isfinite(groups[n])]
        se = v.std(ddof=1) / np.sqrt(len(v))
        colr = GRAY if "shuffled" in n else (S1 if "same" in n else S2)
        ax.errorbar(i, v.mean(), yerr=1.96 * se, fmt="o", color=colr, ecolor=colr, capsize=4, lw=1.5, ms=8)
    ax.axhline(0.5, color=INK2, ls=":", lw=1)
    ax.text(len(names) - 0.5, 0.505, "chance", color=INK2, ha="right", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)), names)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="detection balanced accuracy", ylim=(0.45, 0.7),
           title=f"Explanation transfer across loops (n={len(audit)} neurons, mean ± 95% CI)")
    return _save(fig, path)


def all_figures(res_dir: Path, fig_dir: Path) -> list[Path]:
    """Render every figure whose inputs exist."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    rd = lambda n: json.loads((res_dir / n).read_text(encoding="utf-8"))
    made = []
    if (res_dir / "setup_check.json").exists():
        made.append(loss_by_T(rd("setup_check.json"), fig_dir / "loss_by_T.png"))
    if (res_dir / "loop_structure.json").exists():
        made.append(loop_structure(rd("loop_structure.json"), fig_dir / "loop_structure.png"))
    if (res_dir / "confirm.json").exists():
        conf = rd("confirm.json")
        made.append(confirm_effects(conf, fig_dir / "confirm_effects.png"))
        best = max(conf["results"], key=lambda r: r["mean_delta"])
        made.append(null_histogram(best, fig_dir / "null_histogram.png"))
    if (res_dir / "followups.json").exists():
        fu = rd("followups.json")
        if "specificity" in fu:
            made.append(specificity(fu["specificity"], fig_dir / "specificity.png"))
        for key, f in fu.items():
            if key == "specificity" or "dose" not in f:
                continue
            made.append(dose(f["dose"], key, fig_dir / f"dose_{key}.png"))
            made.append(loop_locus(f["loop_locus"], key, fig_dir / f"loop_locus_{key}.png"))
            made.append(t_sweep(f["t_sweep"], key, fig_dir / f"t_sweep_{key}.png"))
    if (res_dir / "audit.json").exists():
        made.append(audit_bars(rd("audit.json"), fig_dir / "audit.png"))
    return made
