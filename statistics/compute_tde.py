#!/usr/bin/env python3
"""Compute Tree Edit Distance between two secondary structures (dot-bracket).

Uses the RNAdistance CLI when available; otherwise falls back to the ViennaRNA
Python API (same full-tree distance as ``RNAdistance -D f``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess


def _tde_rnadistance_cli(structure1: str, structure2: str) -> int:
    process = subprocess.Popen(
        ["RNAdistance", "-D", "f"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(input=f"{structure1}\n{structure2}\n")
    if process.returncode != 0:
        raise RuntimeError(f"RNAdistance failed: {stderr.strip()}")
    return int(stdout.split(":")[1].strip())


def _tde_viennarna_python(structure1: str, structure2: str) -> int:
    try:
        import RNA
    except ImportError as exc:
        raise RuntimeError(
            "Neither RNAdistance CLI nor ViennaRNA Python package (RNA) is available."
        ) from exc

    tree1 = RNA.make_tree(RNA.expand_Full(structure1))
    tree2 = RNA.make_tree(RNA.expand_Full(structure2))
    return int(RNA.tree_edit_distance(tree1, tree2))


def compute_tde(structure1: str, structure2: str) -> int:
    """Tree edit distance between two dot-bracket structures."""
    if shutil.which("RNAdistance"):
        return _tde_rnadistance_cli(structure1, structure2)
    return _tde_viennarna_python(structure1, structure2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Tree Edit Distance between two secondary structures."
    )
    parser.add_argument("structure1", help="First dot-bracket structure")
    parser.add_argument("structure2", help="Second dot-bracket structure")
    args = parser.parse_args()
    print(compute_tde(args.structure1, args.structure2))


if __name__ == "__main__":
    main()
