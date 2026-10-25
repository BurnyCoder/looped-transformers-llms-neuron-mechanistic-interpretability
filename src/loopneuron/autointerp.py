"""Automated neuron explanations, detection scoring and AxBench-style judging with the local LLM.

Local:
- `explain`: Delphi explainer prompt over top examples with the peak token marked <<like this>> and its activation
  on a 0-10 scale relative to the neuron's max (scale convention of Bills et al. 2023).
- `detection_score`: Delphi detection scorer -> balanced accuracy of the LLM deciding, from the explanation alone,
  which held-out contexts are activating vs random (Paulo et al. 2024, https://arxiv.org/abs/2410.13928).
- `judge`: AxBench concept/fluency ratings 0-2 with explanation before the score (Wu et al. 2025).
Global context: labeling of candidates, the interpretability audit (H4) and the secondary LLM-judge metric.
"""
from __future__ import annotations

import random
import re

from loopneuron import prompts as P


def explain(llm, examples: list[tuple[str, float]], max_act: float, tag: str = "") -> str:
    """examples: (marked_text, activation). Returns the one-line explanation after [EXPLANATION]:."""
    lines = [f"Example {i + 1}: {t} ({round(10 * abs(a) / max(abs(max_act), 1e-9))})" for i, (t, a) in enumerate(examples)]
    out = llm.chat([{"role": "system", "content": P.EXPLAINER_SYSTEM}, {"role": "user", "content": "\n".join(lines)}],
                   max_tokens=300, tag=f"explain {tag}")
    m = re.findall(r"\[EXPLANATION\]:\s*(.+)", out)
    return (m[-1] if m else out.strip().splitlines()[-1] if out.strip() else "").strip()


def _parse_list(text: str, n: int) -> list[int] | None:
    """First bracketed 0/1 list in a Delphi detection reply; truncated to n entries, None if too short or absent."""
    m = re.search(r"\[([01,\s]*)\]", text)
    if not m:
        return None
    vals = [int(v) for v in re.findall(r"[01]", m.group(1))]
    return vals[:n] if len(vals) >= n else None  # extra trailing entries are ignored; too few -> unusable


def detect(llm, explanation: str, texts: list[str], tag: str = "") -> list[int] | None:
    """Delphi detection call: 1/0 per example (None if the reply cannot be parsed)."""
    msgs = [{"role": "system", "content": P.DETECTION_SYSTEM}]
    for q, a in P.DETECTION_FEWSHOT:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    ex = "\n".join(f"Example {i}: {t}" for i, t in enumerate(texts))
    msgs.append({"role": "user", "content": P.DETECTION_PROMPT.format(explanation=explanation, examples=ex)})
    return _parse_list(llm.chat(msgs, max_tokens=60, tag=f"detect {tag}"), len(texts))


def detection_score(llm, explanation: str, pos: list[str], neg: list[str], seed: int, per_call: int = 10,
                    tag: str = "") -> dict:
    """Balanced accuracy over shuffled activating (pos) and random (neg) contexts; unparsable calls are skipped."""
    items = [(t, 1) for t in pos] + [(t, 0) for t in neg]
    random.Random(seed).shuffle(items)
    tp = tn = n_pos = n_neg = 0
    for i in range(0, len(items), per_call):
        chunk = items[i:i + per_call]
        pred = detect(llm, explanation, [t for t, _ in chunk], tag)
        if pred is None:
            continue
        for (_, y), p in zip(chunk, pred):
            n_pos, n_neg = n_pos + y, n_neg + (1 - y)
            tp, tn = tp + (y and p), tn + ((1 - y) and (1 - p))
    if n_pos == 0 or n_neg == 0:
        return {"balanced_acc": float("nan"), "n": n_pos + n_neg}
    return {"balanced_acc": 0.5 * (tp / n_pos + tn / n_neg), "tpr": tp / n_pos, "tnr": tn / n_neg, "n": n_pos + n_neg}


def judge(llm, sentence: str, concept: str | None = None, tag: str = "") -> int | None:
    """AxBench rating 0-2: concept relevance if `concept` is given, else fluency. None if unparsable."""
    tpl = P.JUDGE_CONCEPT.format(concept=concept, sentence=sentence) if concept else P.JUDGE_FLUENCY.format(sentence=sentence)
    out = llm.chat([{"role": "user", "content": tpl}], max_tokens=250, tag=f"judge {tag}")
    m = re.search(r"Rating:\s*\[\[(\d)\]\]", out)
    return int(m.group(1)) if m else None
