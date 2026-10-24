"""Compile paper/paper.typ -> paper/paper.pdf with the `typst` Python package (https://pypi.org/project/typst/).

Global context: run after `run_pipeline.py --phase report` so that figures/ contains the PNGs the paper embeds.
"""
from pathlib import Path

import typst

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    typst.compile(str(HERE / "paper.typ"), output=str(HERE / "paper.pdf"), root=str(HERE.parent))
    print("wrote", HERE / "paper.pdf")
