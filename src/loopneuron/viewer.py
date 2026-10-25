"""Static HTML "microscope" pages: per neuron, its top contexts with token-level activations at every loop.

Local: activations are recomputed with `contexts.token_activations` ([n_ctx, T, S]); each context is shown as T rows
(loop 1..T) of tokens shaded by activation (teal = positive, rust = negative, one scale per neuron), in the spirit of
Goodfire's neuron explorer (https://already-clean.goodfire.com/) and OpenAI's neuron viewer (Bills et al. 2023).
Global context: phase `report` writes docs/viewer/index.html + one page per featured neuron (candidates).
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from loopneuron import contexts as cx

CSS = """body{font-family:system-ui,Segoe UI,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:24px;max-width:1100px}
.ctx{margin:14px 0;padding:8px;border:1px solid #e6e5e1;border-radius:6px}.row{font-family:Consolas,monospace;
font-size:12.5px;white-space:pre-wrap;line-height:1.7}.lab{color:#52514e;font-size:11px;display:inline-block;width:52px}
span.t{border-radius:3px}a{color:#2a78d6}table{border-collapse:collapse}td,th{padding:3px 8px;border-bottom:1px solid #e6e5e1}"""


def _tok_html(tok, ids, acts, scale):
    """HTML spans for tokens, background opacity proportional to |activation| / scale (teal +, rust -)."""
    out = []
    for i, a in zip(ids, acts):
        s = html.escape(tok.decode([i])).replace("\n", "&#8629;")
        x = float(np.clip(a / scale, -1, 1))
        bg = f"rgba(27,175,122,{x:.2f})" if x > 0 else f"rgba(235,104,52,{-x:.2f})"  # teal + / rust -
        out.append(f'<span class="t" style="background:{bg}" title="{a:.3f}">{s}</span>')
    return "".join(out)


def neuron_page(model, tok, atlas, chunks, layer, neuron, pole, T, info: dict, path: Path, n_ctx: int = 8, left=24, right=8):
    """Write one neuron's page: metadata table and its top contexts shaded at every loop."""
    ents =cx.neuron_entries(atlas, layer, pole, neuron, k=n_ctx)
    acts = cx.token_activations(model, chunks, [e[0] for e in ents], layer, neuron, T)  # [n, T, S]
    scale = max(abs(acts).max(), 1e-6)
    blocks = []
    for j, (s, p, v, loop) in enumerate(ents):
        lo, hi = max(0, p - left), min(chunks.shape[1], p + right + 1)
        ids = chunks[s, lo:hi].tolist()
        rows = "".join(f'<div class="row"><span class="lab">loop {t + 1}</span>{_tok_html(tok, ids, acts[j, t, lo:hi], scale)}</div>'
                       for t in range(T))
        blocks.append(f'<div class="ctx"><div class="lab" style="width:auto">chunk {s}, pos {p}, peak {v:.3f} at loop {loop + 1}</div>{rows}</div>')
    meta = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in info.items())
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>L{layer} N{neuron}</title><style>{CSS}</style></head>
<body><p><a href="index.html">all neurons</a></p><h2>Ouro-1.4B layer {layer} neuron {neuron} ({'positive' if pole == 0 else 'negative'} pole)</h2>
<table>{meta}</table><p>Token shading: teal = positive, rust = negative activation; one scale per neuron (max |a| = {scale:.3f}).
Each context is shown at every loop of the shared stack.</p>{''.join(blocks)}</body></html>""", encoding="utf-8")
    return path


def index_page(entries: list[tuple[str, str]], path: Path):
    """Write the viewer index linking every neuron page (entries: (file name, title))."""
    rows ="".join(f'<li><a href="{html.escape(f)}">{html.escape(t)}</a></li>' for f, t in entries)
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Ouro neuron viewer</title><style>{CSS}</style></head>
<body><h1>Ouro-1.4B neuron viewer</h1><p>Top contexts of featured neurons with activations at each of the 4 loops.</p><ul>{rows}</ul></body></html>""",
                    encoding="utf-8")
    return path
