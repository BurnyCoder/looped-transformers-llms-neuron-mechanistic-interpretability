"""Single entry point: runs the study's named phases in order (see README "What happens when you run it").

Usage:
  uv run python run_pipeline.py                       # all phases (build_atlas skips if atlas.pt is complete)
  uv run python run_pipeline.py --phase build_atlas   # one phase
  uv run python run_pipeline.py --dry-run             # tiny budgets through every phase (smoke test)
Global context: each phase is a thin function here that calls modules in src/loopneuron/ and writes results/<model>/.
Phase order matters for scientific validity: candidates are chosen on discovery prompts, the pre-registration is
committed, and only then does `confirm_steering` touch the confirmatory prompts.
"""
from __future__ import annotations

import argparse
import gc
import os
import json
import subprocess

import numpy as np
import torch
from transformers import AutoTokenizer

from loopneuron.config import load_settings
from loopneuron.data import build_prompt_splits, cached_chunks, load_prompt_split, stream_texts
from loopneuron.logging_utils import get_logger
from loopneuron.model import load_model

PHASES = ["setup_check", "build_atlas", "screen_candidates", "validate_candidates", "label_candidates", "preregister",
          "confirm_steering",
          "followups", "llm_judge", "audit", "extras", "replicate", "replicate_fresh", "report", "publish"]


class Ctx:
    """Lazily loaded shared state (settings, logger, model, tokenizer, data, atlas) passed to every phase."""

    def __init__(self, s, log):
        """Store settings and logger; heavy objects (model, tokenizer, atlas) load on first access."""
        self.s, self.log, self._model, self._tok, self._atlas = s, log, None, None, None
        self.out = s.results_dir  # results/<model-name><run_tag>/
        self.out.mkdir(parents=True, exist_ok=True)

    @property
    def tok(self):
        """Ouro tokenizer (pinned revision); available without loading the model onto the GPU."""
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.s.model_id, revision=self.s.model_revision)
        return self._tok

    @property
    def model(self):
        """Ouro on the GPU via loopneuron.model.load_model (bf16, fixed T, last-loop exit, fast KV cache)."""
        if self._model is None:
            self.log.info("loading %s @ %s (T=%d)", self.s.model_id, self.s.model_revision, self.s.ut_steps)
            self._model, self._tok = load_model(self.s.model_id, self.s.model_revision, ut_steps=self.s.ut_steps,
                                                  min_free_gb=self.s.min_free_gb)
        return self._model

    def free_model(self):
        """Release Ouro from the GPU (needed before starting the local LLM server: 8 GB VRAM)."""
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def atlas(self) -> dict:
        """The neuron atlas dict written by build_atlas (CPU tensors, ~0.5 GB)."""
        if self._atlas is None:
            self._atlas = torch.load(self.out / "atlas.pt", weights_only=False)
        return self._atlas

    def chunks(self, split: str) -> np.ndarray:
        """Packed token chunks for 'atlas' or 'eval' (cached under cache/), truncated to the configured budget."""
        n = self.s.atlas_tokens if split == "atlas" else 256 * self.s.seq_len
        arr = cached_chunks(self.s.root / "cache" / f"tokens_{split}.npy",
                            texts=stream_texts(split, self.s.math_fraction, self.s.seed), tok=self.tok,
                            seq_len=self.s.seq_len, n_tokens=n)
        return arr[: n // self.s.seq_len]

    def prompts(self, name: str) -> list[dict]:
        """Prompt split 'discovery' | 'confirm' | 'reserve' (built once from WikiText-103 test if missing)."""
        d = self.s.root / "data" / "prompts"
        if not (d / f"{name}.jsonl").exists():
            build_prompt_splits(self.tok, d, seed=self.s.seed)
        return load_prompt_split(d, name)

    def read(self, name: str):
        """Load a results JSON written by an earlier phase."""
        return json.loads((self.out / name).read_text(encoding="utf-8"))

    def write(self, name: str, obj) -> None:
        """Write a results JSON (floats from numpy are converted by default=float) and log its path."""
        (self.out / name).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
        self.log.info("wrote %s", self.out / name)

    def llm_server(self):
        """Context manager for the local Qwen3.5 llama-server; frees Ouro from the GPU first (8 GB VRAM)."""
        from loopneuron.llm import LlamaServer

        self.free_model()
        root = self.s.root
        port = int(self.s.llm_base_url.rsplit(":", 1)[1].split("/")[0])  # e.g. http://127.0.0.1:8093/v1 -> 8093
        exe = root / "tools" / "llama.cpp" / ("llama-server.exe" if os.name == "nt" else "llama-server")
        return LlamaServer(exe, root / "cache" / "gguf" / self.s.llm_gguf,
                           port, self.s.logs_dir / "llama_server.txt", log=self.log)


def setup_check(c: Ctx) -> None:
    """Phase 0: state-dict keys, held-out loss for T = 1..8, default-exit disagreement, sample continuations."""
    from loopneuron.sanity import setup_check as run

    run(c.model, c.tok, c.chunks("eval"), c.prompts("discovery"), c.s.ut_steps, c.out / "setup_check.json", c.log)


def build_atlas(c: Ctx) -> None:
    """Phase 1: per-neuron, per-pole, per-loop top contexts and statistics over the atlas tokens (skips if complete)."""
    from loopneuron.atlas import build_atlas as run

    path = c.out / "atlas.pt"
    if path.exists() and torch.load(path, weights_only=False).get("done_chunks", 0) >= len(c.chunks("atlas")):
        c.log.info("atlas exists: %s", path)
        return
    run(c.model, c.chunks("atlas"), c.s.ut_steps, c.s.topk, c.s.topk_per_loop, c.s.atlas_batch, c.log, save_path=path)


def screen_candidates(c: Ctx) -> None:
    """Discovery-only selection: coherence pool -> quick steer of the most coherent units (discovery prompts 0-24)."""
    from loopneuron.screen import screen

    res = screen(c.model, c.tok, c.atlas, c.chunks("atlas"), c.prompts("discovery"), c.s.ut_steps, c.s.seed, c.log,
                 n_pool=c.s.screen_pool, n_steer=c.s.screen_steer, K=2 * c.s.n_candidates)
    c.write("screen.json", res)


def validate_candidates(c: Ctx) -> None:
    """Top 2K screened units re-tested on discovery prompts 25-124 (selection-free effect sizes, degeneracy check)."""
    from loopneuron.screen import reselect_and_validate

    res = c.read("screen.json")
    pool, u, val = reselect_and_validate(c.model, c.tok, c.atlas, c.chunks("atlas"), res, c.prompts("discovery")[25:125],
                                         c.s.ut_steps, c.s.seed + 1, c.s.gen_tokens, 2 * c.s.n_candidates, c.log)
    res.update({"candidates": pool, "u": u, "validation": val})
    c.write("screen.json", res)


def label_candidates(c: Ctx) -> None:
    """Local Qwen explanation for each candidate from its top-20 marked contexts (documentation + LLM judge concept)."""
    from loopneuron import contexts as cx
    from loopneuron.autointerp import explain
    from loopneuron.llm import LLM

    res, chunks = c.read("screen.json"), c.chunks("atlas")
    with c.llm_server():
        llm = LLM(c.s.llm_base_url, c.s.llm_model, c.log)
        for cand in res["candidates"]:
            ents = cx.neuron_entries(c.atlas, cand["layer"], cand["pole"], cand["neuron"], k=20)
            ex = [(cx.window_text(c.tok, chunks, s, p, mark=True), v) for s, p, v, _ in ents]
            cand["label"] = explain(llm, ex, ents[0][2], tag=f'L{cand["layer"]}N{cand["neuron"]}p{cand["pole"]}')
            c.log.info("LABEL L%d N%d pole%d: %s", cand["layer"], cand["neuron"], cand["pole"], cand["label"])
    c.write("screen.json", res)


def preregister(c: Ctx) -> None:
    """Freeze candidates/alpha/n/tests in results/candidates.json and reports/preregistration.md (commit before confirm)."""
    from loopneuron.prereg import write_preregistration

    md = c.out / "preregistration.md" if c.s.run_tag else c.s.root / "reports" / "preregistration.md"  # dry runs stay local
    write_preregistration(c.read("screen.json"), c.s, c.out, md, c.log)


def _assert_preregistered(c: Ctx) -> None:
    """Refuse to run the confirmatory test unless the pre-registration is committed and unmodified."""
    root, path = str(c.s.root), "reports/preregistration.md"
    committed = subprocess.run(["git", "-C", root, "log", "-1", "--format=%H", "--", path], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", path], capture_output=True, text=True).stdout.strip()
    if not committed or dirty:
        raise RuntimeError("pre-registration must be committed (and unchanged) before confirm_steering")
    c.log.info("pre-registration committed at %s", committed)


def confirm_steering(c: Ctx) -> None:
    """Phase 4: the pre-registered confirmatory test (refuses to run unless the pre-registration is committed)."""
    from loopneuron.confirm import confirm

    if c.s.run_tag != "-dryrun":
        _assert_preregistered(c)
    pre, scr = c.read("candidates.json"), c.read("screen.json")
    prompts = c.prompts("confirm")[: pre["n_prompts"]]
    res = confirm(c.model, c.tok, c.atlas, c.chunks("atlas"), pre["candidates"], scr["u"], scr["rand_mean"], prompts,
                  c.s.ut_steps, pre["seed"], pre["k"], pre["n_tokens"], pre["R"], c.log, c.out)
    c.write("confirm.json", res)


def _focus(c: Ctx, exploratory: bool = True) -> list[dict]:
    """Neurons for follow-ups: significant candidates (or the best one if none passed, labeled post hoc), plus the
    pre-registered exploratory units (outside the confirmatory family)."""
    res = c.read("confirm.json")["results"]
    sig = [r for r in res if r["significant"]] or [max(res, key=lambda r: r["mean_delta"])]
    return sig + (c.read("candidates.json").get("exploratory", []) if exploratory else [])


def followups(c: Ctx) -> None:
    """Secondary (post hoc) causal experiments on 60 confirm prompts, k=1: specificity matrix, dose-response, loop locus
    (H2), test-time loops (H3), zero-ablation."""
    from loopneuron import experiments as ex

    scr, pre = c.read("screen.json"), c.read("candidates.json")
    P = torch.tensor([p["ids"] for p in c.prompts("confirm")[:60]])  # 60 prompts x 1 sample per condition
    focus = _focus(c)
    out = {"specificity": ex.specificity_matrix(c.model, c.tok, c.atlas, focus, scr["u"], P, c.s.ut_steps,
                                                  pre["n_tokens"], pre["seed"] + 400, c.log)}
    c.write("followups.json", out)
    for r in focus:
        cand = {**r, "key": r["key"]}
        u = np.array(scr["u"][r["key"]])
        seed = pre["seed"] + 500
        out[r["key"]] = {
            "dose": ex.dose_response(c.model, c.tok, c.atlas, cand, u, P, c.s.ut_steps, pre["n_tokens"], seed, c.log),
            "loop_locus": ex.loop_locus(c.model, c.tok, c.atlas, cand, u, P, c.s.ut_steps, pre["n_tokens"], seed, c.log),
            "t_sweep": ex.t_sweep(c.model, c.tok, c.atlas, cand, u, P, c.s.ut_steps, pre["n_tokens"], seed, c.log),
            "ablation": ex.ablation(c.model, c.atlas, c.chunks("atlas"), cand, c.s.ut_steps, pre["R"], seed, c.log),
        }
        c.write("followups.json", out)  # checkpoint after every neuron


def llm_judge(c: Ctx) -> None:
    """Secondary: blind AxBench-style LLM concept/fluency ratings of steered vs unsteered text (50 confirm prompts)."""
    from loopneuron.judge import generate_for_judge, judge_all
    from loopneuron.llm import LLM

    pre, scr = c.read("candidates.json"), c.read("screen.json")
    labels = {f'{x["layer"]}_{x["neuron"]}_{x["pole"]}': x.get("label", "") for x in scr["candidates"]}
    cands = [r for r in c.read("confirm.json")["results"] if r["significant"]]
    texts = generate_for_judge(c.model, c.tok, c.atlas, cands, c.prompts("confirm")[:50], c.s.ut_steps,
                               pre["n_tokens"], pre["seed"])  # same seed base; fresh draws (batch layout differs)
    c.write("llm_judge_texts.json", texts)
    with c.llm_server():
        res = judge_all(LLM(c.s.llm_base_url, c.s.llm_model, c.log), texts, labels, c.s.seed, c.log)
    c.write("llm_judge.json", res)


def extras(c: Ctx) -> None:
    """Plan-completion secondary analyses (seed sets, alpha=0, concept tokens, self-repair, fits, Wilson CIs, position)."""
    from loopneuron import extras as xx

    pre, scr = c.read("candidates.json"), c.read("screen.json")
    cands = [r for r in c.read("confirm.json")["results"] if r["significant"]]
    P100 = torch.tensor([p["ids"] for p in c.prompts("confirm")[:100]])
    seeds = [pre["seed"] + 3000, pre["seed"] + 4000, pre["seed"] + 5000]  # never used before
    seed_res, base = xx.seed_sensitivity(c.model, c.tok, c.atlas, cands, scr["u"], P100, c.s.ut_steps, pre["n_tokens"],
                                         seeds, c.log)
    out = {"seed_sets": seeds, "seed_sensitivity": seed_res,
           "zero_clamp": xx.zero_clamp(c.model, c.tok, c.atlas, cands, scr["u"], P100[:60], c.s.ut_steps,
                                       pre["n_tokens"], pre["seed"] + 500, c.log),
           "concept_tokens": xx.concept_token_shift(c.model, c.tok, c.atlas, c.chunks("atlas"), cands, P100, base,
                                                    c.s.ut_steps, pre["seed"], c.log),
           "self_repair": xx.self_repair(c.model, c.atlas, cands + pre.get("exploratory", []), P100[:60], c.s.ut_steps,
                                         c.log),
           "tsweep_fit": xx.tsweep_fit(c.read("followups.json")),
           "audit_wilson": xx.audit_wilson(c.read("audit.json")),
           "positional": xx.positional_check(c.atlas, c.s.seq_len, scr["candidates"])}
    c.write("extras.json", out)


def audit(c: Ctx) -> None:
    """H4: loop structure over all neurons + local-LLM per-loop explanation/detection audit on sampled neurons."""
    from loopneuron.audit import cross_loop_correlation, llm_audit, loop_structure
    from loopneuron.llm import LLM

    ls = loop_structure(c.atlas)
    corr = cross_loop_correlation(c.model, c.chunks("eval")[:128], c.s.ut_steps)  # [L, D, T, T], held-out text
    np.savez_compressed(c.out / "cross_loop_corr.npz", corr=corr.astype(np.float16))
    T = c.s.ut_steps
    ls["corr_first_last_median"] = np.nanmedian(corr[:, :, 0, T - 1], axis=1).tolist()
    ls["corr_adjacent_median"] = [np.nanmedian(corr[:, :, t, t + 1], axis=1).tolist() for t in range(T - 1)]
    ls["corr_first_last_quantiles_all"] = np.nanquantile(corr[:, :, 0, T - 1], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    c.write("loop_structure.json", ls)
    chunks = c.chunks("atlas")
    with c.llm_server():
        res = llm_audit(LLM(c.s.llm_base_url, c.s.llm_model, c.log), c.atlas, chunks, c.tok, c.s.seed, c.log,
                        per_tercile=c.s.audit_per_tercile)
    c.write("audit.json", res)


def replicate(c: Ctx) -> None:
    """Same-index neuron replication on Ouro-1.4B-Thinking with the pre-registered protocol and a fresh seed."""
    from loopneuron.confirm import confirm
    from loopneuron.replicate import max_activation, neuron_correspondence

    scr, pre = c.read("screen.json"), c.read("candidates.json")
    focus = _focus(c, exploratory=False)  # replication tests only confirmatory neurons
    c.free_model()
    # load first: load_model's disk guard runs before the Thinking checkpoint is downloaded; _weights then reads the cache
    model, _ = load_model(c.s.repl_model_id, c.s.repl_model_revision, ut_steps=c.s.ut_steps, min_free_gb=c.s.min_free_gb)
    corr = [neuron_correspondence((c.s.model_id, c.s.model_revision), (c.s.repl_model_id, c.s.repl_model_revision),
                                  r["layer"], [r["neuron"]]) for r in focus]
    chunks = c.chunks("atlas")
    maxes = {r["key"]: max_activation(model, c.atlas, chunks, r, c.s.ut_steps) for r in focus}
    prompts = c.prompts("confirm")[: pre["n_prompts"]]
    cands = [{k: r[k] for k in ("key", "layer", "neuron", "pole", "alpha")} for r in focus]
    # R=20 random neurons: the replication null is descriptive (z-margin), not a Holm-family test, so a smaller R keeps
    # the run near 30 min (28 min logged) instead of ~2.5 h; the Wilcoxon test vs unsteered is the replication criterion.
    res = confirm(model, c.tok, c.atlas, chunks, cands, scr["u"], scr["rand_mean"], prompts, c.s.ut_steps,
                  pre["seed"] + 2000, pre["k"], pre["n_tokens"], 20, c.log, c.out, "replication_examples.jsonl")
    c.write("replication.json", {"model": c.s.repl_model_id, "revision": c.s.repl_model_revision,
                                 "correspondence": corr, "thinking_max_act": maxes,
                                 "base_max_act": {r["key"]: float(c.atlas["top_v"][r["layer"], r["pole"], r["neuron"], 0])
                                                  for r in focus}, **res})


def replicate_fresh(c: Ctx) -> None:
    """Replication on Ouro-1.4B-Thinking with FRESH prompts (the unused reserve split) and fresh seeds."""
    from loopneuron.confirm import confirm

    scr, pre = c.read("screen.json"), c.read("candidates.json")
    cands = [{k: r[k] for k in ("key", "layer", "neuron", "pole", "alpha")} for r in _focus(c, exploratory=False)]
    c.free_model()
    model, _ = load_model(c.s.repl_model_id, c.s.repl_model_revision, ut_steps=c.s.ut_steps, min_free_gb=c.s.min_free_gb)
    prompts = c.prompts("reserve")[: pre["n_prompts"]]  # never used before (no second confirmatory round was needed)
    res = confirm(model, c.tok, c.atlas, c.chunks("atlas"), cands, scr["u"], scr["rand_mean"], prompts, c.s.ut_steps,
                  pre["seed"] + 6000, pre["k"], pre["n_tokens"], 20, c.log, c.out, "replication_fresh_examples.jsonl")
    c.write("replication_fresh.json", {"model": c.s.repl_model_id, "revision": c.s.repl_model_revision,
                                       "prompt_split": "reserve", **res})


def report(c: Ctx) -> None:
    """Figures from results JSON + HTML viewer pages for every screened candidate."""
    from loopneuron.plots import all_figures
    from loopneuron.viewer import index_page, neuron_page

    base = c.out if c.s.run_tag else c.s.root  # dry runs keep figures/viewer inside their results folder
    figs = all_figures(c.out, base / "figures")
    c.log.info("figures: %s", [str(f.name) for f in figs])
    from loopneuron.reporting import write_tables

    c.log.info("tables: %s", [t.name for t in write_tables(c.out, (c.out if c.s.run_tag else base / "reports") / "tables")])
    vdir = base / "docs" / "viewer"
    vdir.mkdir(parents=True, exist_ok=True)
    conf = {r["key"]: r for r in c.read("confirm.json")["results"]} if (c.out / "confirm.json").exists() else {}
    entries = []
    for cand in c.read("screen.json")["candidates"]:
        key = f'{cand["layer"]}_{cand["neuron"]}_{cand["pole"]}'
        r = conf.get(key, {})
        info = {"label (local Qwen3.5)": cand.get("label", ""), "coherence": round(cand["coherence"], 3),
                "max activation": round(cand["max_act"], 3), "screen alpha": cand["best_alpha"],
                "confirm mean dC": round(r.get("mean_delta", float("nan")), 4), "significant": r.get("significant", "n/a")}
        f = f"neuron_{key}.html"
        neuron_page(c.model, c.tok, c.atlas, c.chunks("atlas"), cand["layer"], cand["neuron"], cand["pole"],
                    c.s.ut_steps, info, vdir / f)
        entries.append((f, f'L{cand["layer"]} N{cand["neuron"]} {"+" if cand["pole"] == 0 else "-"}: {cand.get("label", "")}'))
    index_page(entries, vdir / "index.html")
    from loopneuron import contexts as cx
    from loopneuron.plots import neuron_loops

    for r in [r for r in conf.values() if r.get("significant")]:  # static per-loop heatmap of each confirmed neuron
        s, p, _, _ = cx.neuron_entries(c.atlas, r["layer"], r["pole"], r["neuron"], k=1)[0]
        acts = cx.token_activations(c.model, c.chunks("atlas"), [s], r["layer"], r["neuron"], c.s.ut_steps)[0]
        lo, hi = max(0, p - 20), min(c.s.seq_len, p + 6)
        toks = [c.tok.decode([t]) for t in c.chunks("atlas")[s, lo:hi].tolist()]
        neuron_loops(toks, acts[:, lo:hi], f'L{r["layer"]} N{r["neuron"]} {"+" if r["pole"] == 0 else "−"}: '
                     f'activation per loop on its top context', base / "figures" / f'neuron_loops_{r["key"]}.png')


HF_DATASET = "BurnyCoder/ouro-1.4b-neuron-atlas"
DATASET_CARD = """---
license: mit
tags: [interpretability, mechanistic-interpretability, looped-transformer, neurons, steering, ouro]
---
# Ouro-1.4B neuron atlas and steering results

Neuron-level atlas of [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) (revision 574fa66), a looped LM
that applies its 24 layers 4 times. Produced by
https://github.com/BurnyCoder/looped-transformers-llms-neuron-mechanistic-interpretability (see its README and reports/).

Files:
- `atlas.pt`: torch dict for all 135,168 SwiGLU neurons (24 layers x 5632) and both poles (index 0 = positive, 1 = negative).
  - `top_v`/`top_seq`/`top_pos`/`top_loop` [24, 2, 5632, 40]: top-40 contexts, maximum over positions x loops.
  - `lt_v`/`lt_seq`/`lt_pos` [24, 4, 2, 5632, 20]: top-20 contexts per loop.
  - Per-loop `mean`, `mean_abs`, `std`, `frac_pos` [24, 4, 5632].
  - `exceed` counts and the `dead` flag.
- `tokens_atlas.npy`: uint16 [31250, 256], the token chunks that `*_seq`/`*_pos` index (Ouro tokenizer;
  FineWeb-Edu sample-10BT shard 000 + 10% OpenWebMath).
- `featured_neurons.jsonl`: decoded top contexts (peak token marked <<>>) for the screened and validated candidates.
- `results/*.json`: pre-registered confirmatory test, follow-ups, audit, replication and LLM-judge results.

Main result: 5/5 pre-registered neurons steer generation toward their concept, significantly against unsteered
generation and against 100 random same-layer neurons clamped the same way (Holm-corrected), and the same neuron
indices also steer Ouro-1.4B-Thinking (replication with a descriptive 20-neuron null).

Licenses and attribution: the tables and results produced by this project are MIT-licensed. `tokens_atlas.npy` and the
decoded contexts in `featured_neurons.jsonl` derive from FineWeb-Edu (https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
and OpenWebMath (https://huggingface.co/datasets/open-web-math/open-web-math), both ODC-By 1.0, and remain subject to
those terms. Activations come from Ouro-1.4B (Apache-2.0).
"""


def publish(c: Ctx) -> None:
    """Upload atlas + results to a public HF dataset in the 'Looped-LM neurons' collection (no models are trained)."""
    from loopneuron.publish import featured_jsonl, publish as upload

    keys = [f'{x["layer"]}_{x["neuron"]}_{x["pole"]}' for x in c.read("screen.json")["candidates"]]
    feat = featured_jsonl(c.atlas, c.chunks("atlas"), c.tok, keys, c.out / "featured_neurons.jsonl")
    files = {"atlas.pt": c.out / "atlas.pt", "tokens_atlas.npy": c.s.root / "cache" / "tokens_atlas.npy",
             "featured_neurons.jsonl": feat}
    files.update({f"results/{p.name}": p for p in sorted(c.out.glob("*.json"))})
    upload(HF_DATASET, files, DATASET_CARD, c.log)


def main() -> None:
    """CLI: run all phases in PHASES order, one phase (--phase), or a tiny smoke test (--dry-run, never publishes)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=PHASES, help="run only this phase")
    ap.add_argument("--dry-run", action="store_true", help="tiny budgets, separate results folder")
    args = ap.parse_args()
    # dry run: tiny budgets and a separate results folder (results/<model>-dryrun/)
    over = {"atlas_tokens": 256 * 64, "null_neurons": 3, "run_tag": "-dryrun", "screen_pool": 60, "screen_steer": 4,
            "n_candidates": 2, "audit_per_tercile": 1} if args.dry_run else {}
    s = load_settings(**over)
    log = get_logger("dryrun" if args.dry_run else (args.phase or "pipeline"), s.logs_dir)
    c = Ctx(s, log)
    phases = [args.phase] if args.phase else [p for p in PHASES if not (args.dry_run and p == "publish")]  # never upload smoke tests
    for name in phases:
        log.info("=== phase: %s ===", name)
        globals()[name](c)


if __name__ == "__main__":
    main()
