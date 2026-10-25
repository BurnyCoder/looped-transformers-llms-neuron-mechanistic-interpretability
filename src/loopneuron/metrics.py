"""Behavioral metrics for steered generations.

Local:
- Concept score (primary, LLM-free): cosine between a continuation's sentence embedding and the neuron's
  *contrastive concept direction* u = normalize(mean emb(top contexts) - mean emb(random contexts)).
  Embedder: BAAI/bge-small-en-v1.5 (https://huggingface.co/BAAI/bge-small-en-v1.5) via sentence-transformers
  (https://www.sbert.net/); robustness embedder: sentence-transformers/all-MiniLM-L6-v2.
- Fluency guard: perplexity of the continuation under the *unsteered* model given the prompt.
- distinct-2: unique bigrams / bigrams (Li et al. 2016, https://arxiv.org/abs/1510.03055) for repetition confounds.
- Concept-token mass: log of total next-token probability on the neuron's keyword tokens (teacher-forced).
Global context: screening and the confirmatory test compute these for every (prompt, sample, condition).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

PRIMARY_EMBEDDER = "BAAI/bge-small-en-v1.5"
ROBUST_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Unit-normalized sentence embeddings (cosine = dot product)."""

    def __init__(self, name: str = PRIMARY_EMBEDDER, device: str = "cuda"):
        """Load a sentence-transformers model (downloads ~0.1 GB on first use)."""
        self.name = name
        self.m = SentenceTransformer(name, device=device)

    def __call__(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed `texts` -> [N, dim] unit vectors."""
        out = self.m.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)
        torch.cuda.empty_cache()  # return the embedder's activation memory so generation's KV cache never spills (8 GB GPU)
        return out


def concept_direction(top_emb: np.ndarray, rand_mean: np.ndarray) -> np.ndarray:
    """u = normalize(mean(top_emb) - rand_mean): what distinguishes the neuron's contexts from typical text."""
    d = top_emb.mean(0) - rand_mean
    return d / (np.linalg.norm(d) + 1e-12)


def coherence(emb: np.ndarray) -> float:
    """Mean pairwise cosine among a neuron's top-context embeddings (monosemanticity proxy)."""
    n = len(emb)
    if n < 2:
        return 0.0
    sim = emb @ emb.T
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def concept_scores(emb: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Cosine of each (unit) continuation embedding with the unit concept direction."""
    return emb @ u


def distinct2(token_ids) -> float:
    """Fraction of unique bigrams in a token sequence."""
    bigrams = list(zip(token_ids[:-1], token_ids[1:]))
    return len(set(bigrams)) / max(1, len(bigrams))


@torch.no_grad()
def continuation_nll(model, prompt_ids: torch.Tensor, cont_ids: torch.Tensor, batch: int = 32) -> np.ndarray:
    """Mean NLL (nats/token) of each continuation under the (unhooked) model given its prompt -> [B]."""
    out = []
    for i in range(0, len(prompt_ids), batch):
        p, c = prompt_ids[i:i + batch].cuda(), cont_ids[i:i + batch].cuda()
        ids = torch.cat([p, c], dim=1)
        logits = model(ids, use_cache=False).logits[:, p.shape[1] - 1:-1].float()
        nll = F.cross_entropy(logits.transpose(1, 2), c, reduction="none").mean(1)
        out.append(nll.cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def keyword_logmass(model, ids: torch.Tensor, keyword_ids: list[int], start: int, batch: int = 32) -> np.ndarray:
    """Mean over continuation positions of log sum_{k in keywords} p(k | prefix), per row of `ids` -> [B].

    Teacher-forced: the same text is scored with and without steering (caller attaches hooks), so the difference is
    how much the intervention shifts next-token probability mass toward the neuron's peak-context keywords.
    """
    kw = torch.tensor(keyword_ids, device="cuda")
    out = []
    for i in range(0, len(ids), batch):
        x = ids[i:i + batch].cuda()
        logp = model(x, use_cache=False).logits[:, start - 1:-1].float().log_softmax(-1)
        out.append(torch.logsumexp(logp[..., kw], dim=-1).mean(1).cpu())
    return torch.cat(out).numpy()
