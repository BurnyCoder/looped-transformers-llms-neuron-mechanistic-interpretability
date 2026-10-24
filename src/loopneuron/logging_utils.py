"""Timestamped logging to file + terminal in real time.

Global context: every phase logs through `get_logger`, producing `logs/<timestamp>_<run>.log`. LLM prompts
and outputs are logged in full by `llm.py` through the same logger (no truncation anywhere).
Python logging cookbook: https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
"""
from __future__ import annotations

import logging  # stdlib logging framework
import sys  # sys.stdout stream for the terminal handler
import time  # timestamp for the log file name
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"  # ISO-like timestamp on every line


def get_logger(run_name: str, log_dir: Path) -> logging.Logger:
    """Return logger `loopneuron.<run_name>` writing to logs/<YYYYmmdd-HHMMSS>_<run_name>.log and stdout."""
    logger = logging.getLogger(f"loopneuron.{run_name}")  # one named logger per run/phase
    if logger.handlers:  # already configured in this process -> reuse (avoid duplicate lines)
        return logger
    log_dir.mkdir(parents=True, exist_ok=True)  # logs/ may not exist on a fresh clone
    path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{run_name}.log"  # timestamped file per run
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter(_FMT))  # same format in file and terminal
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # do not double-print through the root logger
    logger.info("log file: %s", path)
    return logger
