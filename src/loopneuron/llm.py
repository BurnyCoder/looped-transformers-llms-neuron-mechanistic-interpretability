"""Local LLM (Qwen3.5 GGUF) served by llama.cpp `llama-server`, called through its OpenAI-compatible API.

Local: `LlamaServer` starts/stops the server process (docs: https://github.com/ggml-org/llama.cpp/tree/master/tools/server);
`LLM.chat` sends messages with thinking disabled (`chat_template_kwargs={"enable_thinking": false}`, supported by
Qwen3.x chat templates in llama-server) and logs the full prompt and full output of every call (never truncated).
Global context: autointerp.py (explanations, detection scores, AxBench judge) is the only caller.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import requests
from openai import OpenAI


class LlamaServer:
    """Context manager running `llama-server` on the GPU (Ouro must be unloaded first: 8 GB VRAM)."""

    def __init__(self, exe: Path, gguf: Path, port: int = 8080, ctx: int = 16384, log=None):
        self.cmd = [str(exe), "-m", str(gguf), "-ngl", "99", "-c", str(ctx), "--port", str(port), "--jinja",
                    "-np", "1"]
        self.url, self.log, self.proc = f"http://127.0.0.1:{port}", log, None

    def __enter__(self):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(300):  # model load can take a while on first start
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    if self.log:
                        self.log.info("llama-server ready: %s", " ".join(self.cmd))
                    return self
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(timeout=60)


class LLM:
    """Chat client with full prompt/output logging."""

    def __init__(self, base_url: str, model: str, log):
        self.client = OpenAI(base_url=base_url, api_key="local")  # llama-server ignores the key
        self.model, self.log, self.n_calls = model, log, 0

    def chat(self, messages: list[dict], max_tokens: int = 400, temperature: float = 0.0, tag: str = "") -> str:
        self.n_calls += 1
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = r.choices[0].message.content or ""
        prompt = "\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        self.log.info("LLM CALL %d %s\n--- PROMPT ---\n%s\n--- OUTPUT ---\n%s\n--- END ---", self.n_calls, tag, prompt, text)
        return text
