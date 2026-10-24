"""Load Ouro deterministically and sample continuations.

Local: Ouro ships remote code (`trust_remote_code=True`, pinned revision). `OuroForCausalLM.forward` picks the
output loop via `exit_at_step` (falls back to `self.early_exit_step`); its default threshold path takes a cumsum
of per-loop exit probabilities in bf16, which can exit early. We set `early_exit_step = T-1` so every forward
(including `generate`) reads the last loop. Source: modeling_ouro.py, https://huggingface.co/ByteDance/Ouro-1.4B
Global context: all phases obtain the model only through `load_model`.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_ut_steps(model, ut_steps: int) -> None:
    """Set the number of loops T (test-time recurrence) and always exit after the last loop."""
    model.config.total_ut_steps = ut_steps  # sizes the UniversalTransformerCache (layers x loops entries)
    model.model.total_ut_steps = ut_steps  # loop count used in OuroModel.forward
    model.early_exit_step = ut_steps - 1  # exit_at_step default -> hidden_states_list[T-1]


def load_model(model_id: str, revision: str, ut_steps: int = 4, device: str = "cuda", dtype=torch.bfloat16):
    """Return (model, tokenizer): bf16, eval mode, fixed T, last-loop exit, SDPA attention."""
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    set_ut_steps(model, ut_steps)
    return model, tok


@torch.no_grad()
def sample(model, input_ids: torch.Tensor, max_new_tokens: int, seed: int, temperature: float = 0.7,
           top_p: float = 0.9) -> torch.Tensor:
    """Sample exactly `max_new_tokens` new tokens per row; returns only the new tokens [B, max_new_tokens].

    `torch.manual_seed(seed)` right before `generate` gives common random numbers across conditions that share
    the same batch layout (paired design: steered vs unsteered differ only through the intervention).
    min_new_tokens=max_new_tokens forbids EOS so all continuations have equal length (HF GenerationConfig docs:
    https://huggingface.co/docs/transformers/main_classes/text_generation).
    """
    torch.manual_seed(seed)
    out = model.generate(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=True, temperature=temperature,
        top_p=top_p, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, pad_token_id=0, use_cache=True,
    )
    return out[:, input_ids.shape[1]:]
