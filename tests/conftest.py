"""Shared pytest fixtures. The `ouro` fixture loads Ouro-1.4B once per session (GPU tests only)."""
import pytest
import torch


@pytest.fixture(scope="session")
def ouro():
    """(model, tokenizer, T) for GPU tests; skipped automatically without CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from loopneuron.config import load_settings
    from loopneuron.model import load_model

    s = load_settings()
    model, tok = load_model(s.model_id, s.model_revision, ut_steps=s.ut_steps)
    return model, tok, s.ut_steps
