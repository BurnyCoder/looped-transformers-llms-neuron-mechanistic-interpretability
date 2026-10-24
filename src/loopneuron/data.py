"""Text data: streamed atlas corpus (FineWeb-Edu + OpenWebMath) and held-out steering prompts (WikiText-103 test).

Local: corpora are read with `datasets` streaming (no local copy; https://huggingface.co/docs/datasets/stream),
from fixed parquet shards so atlas text and perplexity-eval text never overlap:
  atlas: FineWeb-Edu sample/10BT/000_00000 (+ OpenWebMath train-00000), eval: FineWeb-Edu sample/10BT/013_00000.
Documents are tokenized, joined with EOS (<|endoftext|>, id 0 in Ouro's tokenizer) and packed into fixed chunks.
Global context: atlas.py consumes `token_chunks`; steering phases consume `load_prompt_split`.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import interleave_datasets, load_dataset

FINEWEB = "HuggingFaceFW/fineweb-edu"  # https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
OWM = "open-web-math/open-web-math"  # https://huggingface.co/datasets/open-web-math/open-web-math
SHARDS = {
    "atlas_web": "sample/10BT/000_00000.parquet",
    "eval_web": "sample/10BT/013_00000.parquet",
    "atlas_math": "data/train-00000-of-00114-5a023365406cb9c4.parquet",
}


def _stream(repo: str, shard: str):
    """Streaming IterableDataset of one parquet shard, text column only."""
    return load_dataset(repo, data_files=shard, split="train", streaming=True).select_columns(["text"])


def stream_texts(split: str = "atlas", math_fraction: float = 0.1, seed: int = 0) -> Iterator[str]:
    """Yield document texts. split='atlas': web/math mixture; split='eval': disjoint web shard only."""
    if split == "eval":
        ds = _stream(FINEWEB, SHARDS["eval_web"])
    elif math_fraction > 0:  # probabilistic interleave (datasets docs: interleave_datasets)
        ds = interleave_datasets([_stream(FINEWEB, SHARDS["atlas_web"]), _stream(OWM, SHARDS["atlas_math"])],
                                 probabilities=[1 - math_fraction, math_fraction], seed=seed,
                                 stopping_strategy="first_exhausted")
    else:
        ds = _stream(FINEWEB, SHARDS["atlas_web"])
    for row in ds:
        yield row["text"]


def token_chunks(texts: Iterator[str], tok, seq_len: int, n_tokens: int, eos_id: int = 0) -> np.ndarray:
    """Pack tokenized documents (EOS-separated) into a uint16 array [n_tokens // seq_len, seq_len].

    uint16 is safe because Ouro's vocab is 49,152 < 65,536; 8M tokens -> 16 MB.
    """
    n_chunks = n_tokens // seq_len
    out = np.empty((n_chunks, seq_len), dtype=np.uint16)
    buf: list[int] = []
    filled = 0
    for text in texts:
        buf.extend(tok(text, add_special_tokens=False).input_ids + [eos_id])  # doc tokens + separator
        while len(buf) >= seq_len and filled < n_chunks:
            out[filled] = buf[:seq_len]
            buf = buf[seq_len:]
            filled += 1
        if filled == n_chunks:
            return out
    return out[:filled]  # corpus exhausted before the budget (logged by caller)


def cached_chunks(path: Path, **kwargs) -> np.ndarray:
    """Load packed chunks from `path` (.npy) or build them with `token_chunks(**kwargs)` and save."""
    if path.exists():
        return np.load(path)
    arr = token_chunks(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def _clean_wikitext(line: str) -> str:
    """Undo WikiText tokenization artifacts (' @-@ ' -> '-', ' @,@ ' -> ',', ' @.@ ' -> '.', space before punct)."""
    line = re.sub(r" @(.)@ ", r"\1", line.strip())
    return re.sub(r" ([,.;:!?')])", r"\1", line).replace("( ", "(")


def build_prompt_splits(tok, out_dir: Path, n_tokens: int = 12, sizes=(("discovery", 150), ("confirm", 300),
                        ("reserve", 300)), seed: int = 0) -> dict[str, list[dict]]:
    """Make disjoint prompt sets of exactly `n_tokens` tokens from WikiText-103 *test* paragraphs.

    Each paragraph contributes at most one stem (its first n_tokens tokens); stems are shuffled with a fixed seed
    and split into discovery (selection), confirm (pre-registered test) and reserve (one pre-registered retry).
    Merity et al. 2016, https://arxiv.org/abs/1609.07843 ; dataset: https://huggingface.co/datasets/Salesforce/wikitext
    """
    rows = load_dataset("Salesforce/wikitext", data_files={"test": "wikitext-103-raw-v1/test-00000-of-00001.parquet"},
                        split="test")["text"]  # test parquet only (0.7 MB) instead of the 300 MB train split
    paras = [_clean_wikitext(r) for r in rows if len(r.split()) > 30 and not r.strip().startswith("=")]
    stems, seen = [], set()
    for p in paras:
        ids = tok(p, add_special_tokens=False).input_ids[:n_tokens]
        text = tok.decode(ids)
        if len(ids) == n_tokens and text not in seen:  # exact length -> padding-free batching
            seen.add(text)
            stems.append({"text": text, "ids": ids})
    random.Random(seed).shuffle(stems)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits, start = {}, 0
    for name, size in sizes:
        splits[name] = stems[start:start + size]
        start += size
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in splits[name])
    return splits


def load_prompt_split(out_dir: Path, name: str) -> list[dict]:
    """Read one prompt split written by `build_prompt_splits`."""
    with open(out_dir / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
