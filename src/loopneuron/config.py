"""Run configuration loaded from `.env` (python-dotenv) plus a disk-space guard.

Global context: every phase in `run_pipeline.py` receives one frozen `Settings` object, so all
hyperparameters of a run live in `.env` (template: `.env.example`) instead of being scattered in code.
python-dotenv docs: https://saurabh-kumar.com/python-dotenv/
"""
from __future__ import annotations

import os  # environment variable access
import shutil  # shutil.disk_usage -> free bytes (https://docs.python.org/3/library/shutil.html#shutil.disk_usage)
from dataclasses import dataclass, field  # immutable settings container
from pathlib import Path  # OS-independent paths (Windows dev machine)

from dotenv import load_dotenv  # reads KEY=VALUE lines from .env into os.environ

ROOT = Path(__file__).resolve().parents[2]  # repo root: src/loopneuron/config.py -> ../../


def _env(key: str, default: str) -> str:
    """Return env var `key` or `default` (local helper so every setting has an explicit default)."""
    return os.environ.get(key, default)  # .env values override defaults


@dataclass(frozen=True)
class Settings:
    """All tunable quantities of the study (frozen so phases cannot mutate them mid-run)."""

    model_id: str  # HF repo of the looped LM under study
    model_revision: str  # pinned commit hash -> reproducible remote code + weights
    repl_model_id: str  # replication checkpoint (same architecture)
    repl_model_revision: str
    ut_steps: int  # number of loops T (Ouro's total_ut_steps)
    atlas_tokens: int  # token budget for the neuron atlas
    seq_len: int  # chunk length for atlas sequences
    atlas_batch: int  # sequences per forward pass while building the atlas
    topk: int  # top-k contexts per neuron/pole (overall)
    topk_per_loop: int  # top-k contexts per neuron/pole/loop
    math_fraction: float  # share of OpenWebMath documents in the atlas stream
    seed: int  # master seed (prompt split, sampling, random-neuron draws)
    gen_tokens: int  # new tokens per steered continuation
    samples_per_prompt: int  # k paired samples per prompt
    null_neurons: int  # R random same-layer neurons in the null distribution
    screen_pool: int  # random alive neurons profiled (both poles) during screening
    screen_steer: int  # most coherent units quick-steered during screening
    n_candidates: int  # K candidates carried into the confirmatory test
    llm_gguf: str  # GGUF file name under cache/gguf/ served by llama-server
    audit_per_tercile: int  # neurons per layer tercile in the LLM audit (H4)
    llm_base_url: str  # llama-server OpenAI-compatible endpoint
    llm_model: str  # model alias passed to llama-server
    min_free_gb: float  # disk that must stay free after any download
    root: Path = field(default=ROOT)  # repo root (outputs are written relative to it)
    run_tag: str = ""  # suffix for the results folder ("-dryrun" keeps smoke tests away from real results)

    @property
    def results_dir(self) -> Path:
        """results/<model-name>/ : per-model outputs (atlas, steering, stats)."""
        return self.root / "results" / (self.model_id.split("/")[-1] + self.run_tag)

    @property
    def logs_dir(self) -> Path:
        """logs/ : timestamped run logs (see logging_utils)."""
        return self.root / "logs"


def load_settings(**overrides) -> Settings:
    """Load `.env` (if present) and build Settings; keyword overrides win (used by --dry-run and tests)."""
    load_dotenv(ROOT / ".env")  # silently no-op when .env is absent -> defaults below apply
    s = Settings(
        model_id=_env("MODEL_ID", "ByteDance/Ouro-1.4B"),
        model_revision=_env("MODEL_REVISION", "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"),
        repl_model_id=_env("REPLICATION_MODEL_ID", "ByteDance/Ouro-1.4B-Thinking"),
        repl_model_revision=_env("REPLICATION_MODEL_REVISION", "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"),
        ut_steps=int(_env("UT_STEPS", "4")),
        atlas_tokens=int(_env("ATLAS_TOKENS", "8000000")),
        seq_len=int(_env("SEQ_LEN", "256")),
        atlas_batch=int(_env("ATLAS_BATCH", "16")),
        topk=int(_env("TOPK", "40")),
        topk_per_loop=int(_env("TOPK_PER_LOOP", "20")),
        math_fraction=float(_env("MATH_FRACTION", "0.1")),
        seed=int(_env("SEED", "1234")),
        gen_tokens=int(_env("GEN_TOKENS", "48")),
        samples_per_prompt=int(_env("SAMPLES_PER_PROMPT", "3")),
        null_neurons=int(_env("NULL_NEURONS", "100")),
        screen_pool=int(_env("SCREEN_POOL", "4000")),
        screen_steer=int(_env("SCREEN_STEER", "200")),
        n_candidates=int(_env("N_CANDIDATES", "10")),
        llm_gguf=_env("LLM_GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
        audit_per_tercile=int(_env("AUDIT_PER_TERCILE", "50")),
        llm_base_url=_env("LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
        llm_model=_env("LLM_MODEL", "qwen3.5-9b"),
        min_free_gb=float(_env("MIN_FREE_GB", "3")),
    )
    return Settings(**{**s.__dict__, **overrides})  # dataclass is frozen -> rebuild with overrides


def disk_guard(required_gb: float, min_free_gb: float, path: Path = ROOT) -> float:
    """Raise if downloading `required_gb` would leave less than `min_free_gb` free on `path`'s drive.

    Returns current free GB. The dev machine has ~25 GB free, so every download goes through this check.
    """
    free_gb = shutil.disk_usage(path).free / 1e9  # bytes -> GB (decimal, like HF file sizes)
    if free_gb - required_gb < min_free_gb:  # projected free space after the download
        raise RuntimeError(f"Disk guard: {free_gb:.1f} GB free, need {required_gb:.1f} GB + {min_free_gb:.1f} GB margin")
    return free_gb
