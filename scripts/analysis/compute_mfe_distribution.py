#!/usr/bin/env python3
"""Compute and plot MFE distributions for Negatives and Sequence_Craft."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "FT"))
from mfe_utils import calculate_mfe  # noqa: E402

CACHE_DIR = os.path.join(PROJECT_ROOT, "generated", "mfe_cache")
PLOT_PATH = os.path.join(PROJECT_ROOT, "generated", "mfe_distribution_negatives_vs_sequence_craft.png")
WORKERS = 16
CHUNK_SIZE = 500

DATASETS = {
    "Sequence_Craft": {
        "path": "Data/Sequence_Craft/SequenceCraft_dataset.csv",
        "column": "e",
    },
    "Negatives": {
        "path": "Data/Negatives/dna_sequences.csv",
        "column": "sequence",
    },
}


def mfe_chunk(sequences):
    return [calculate_mfe(s) for s in sequences]


def load_sequences(label, cfg):
    df = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["path"]))
    return df[cfg["column"]].astype(str).str.upper().tolist()


def compute_mfe_values(label, sequences):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{label.lower()}_mfe.npy")
    if os.path.exists(cache_path):
        print(f"[{label}] loading cache: {cache_path}")
        return np.load(cache_path)

    print(f"[{label}] computing MFE for {len(sequences)} sequences...")
    chunks = [sequences[i : i + CHUNK_SIZE] for i in range(0, len(sequences), CHUNK_SIZE)]
    mfes = []
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for chunk_mfes in pool.map(mfe_chunk, chunks):
            mfes.extend(chunk_mfes)
            done += len(chunk_mfes)
            if done % 5000 == 0 or done == len(sequences):
                print(f"  [{label}] {done}/{len(sequences)}", flush=True)

    arr = np.array(mfes, dtype=np.float64)
    np.save(cache_path, arr)
    print(f"[{label}] saved cache: {cache_path}")
    return arr


def plot_distributions(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Sequence_Craft": "#2563eb", "Negatives": "#dc2626"}
    bins = np.linspace(-30, 2, 65)

    for label, mfes in results.items():
        color = colors[label]
        ax.hist(
            mfes,
            bins=bins,
            density=True,
            alpha=0.45,
            color=color,
            label=f"{label} (n={len(mfes):,})",
            edgecolor="white",
            linewidth=0.3,
        )
        mean = mfes.mean()
        median = np.median(mfes)
        ax.axvline(mean, color=color, linestyle="--", linewidth=1.5, alpha=0.9)
        ax.axvline(median, color=color, linestyle=":", linewidth=1.5, alpha=0.9)
        print(
            f"{label}: mean={mean:.3f}, median={median:.3f}, "
            f"std={mfes.std():.3f}, min={mfes.min():.3f}, max={mfes.max():.3f}"
        )

    ax.set_title("MFE distribution: Negatives vs Sequence_Craft (NUPACK, material=dna)")
    ax.set_xlabel("MFE (kcal/mol)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    stats_text = []
    for label, mfes in results.items():
        stats_text.append(
            f"{label}: μ={mfes.mean():.2f}, med={np.median(mfes):.2f} kcal/mol"
        )
    ax.text(
        0.99,
        0.97,
        "\n".join(stats_text),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#ccc"),
    )

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot: {PLOT_PATH}")


def main():
    results = {}
    for label, cfg in DATASETS.items():
        sequences = load_sequences(label, cfg)
        results[label] = compute_mfe_values(label, sequences)
    plot_distributions(results)


if __name__ == "__main__":
    main()
