#!/usr/bin/env python3
"""Compute MFE for nofilter generated sets and compare with references."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "FT"))
from mfe_utils import calculate_mfe  # noqa: E402

ROOT = os.path.dirname(__file__)
CACHE_DIR = os.path.join(ROOT, "generated", "mfe_cache")
OUT_DIR = os.path.join(ROOT, "generated")
WORKERS = 16
CHUNK_SIZE = 500
BINS = np.linspace(-30, 2, 65)

REFERENCES = {
    "Sequence_Craft": ("Data/Sequence_Craft/SequenceCraft_dataset.csv", "e"),
    "Negatives": ("Data/Negatives/dna_sequences.csv", "sequence"),
}

GENERATED = [
    "eds_pretrain_nofilter",
    "mfe_pretrain_nofilter",
    "eds_ft_nofilter",
    "mfe_ft_nofilter",
]

COLORS = {
    "Sequence_Craft": "#2563eb",
    "Negatives": "#dc2626",
    "eds_pretrain_nofilter": "#16a34a",
    "mfe_pretrain_nofilter": "#ea580c",
    "eds_ft_nofilter": "#9333ea",
    "mfe_ft_nofilter": "#0891b2",
}


def mfe_chunk(sequences):
    return [calculate_mfe(s) for s in sequences]


def cache_name(label: str) -> str:
    return os.path.join(CACHE_DIR, f"{label.lower()}_mfe.npy")


def load_sequences_csv(path: str, column: str) -> list[str]:
    df = pd.read_csv(os.path.join(ROOT, path))
    return df[column].astype(str).str.upper().tolist()


def load_generated_sequences(label: str) -> list[str]:
    path = os.path.join(ROOT, "generated", label, "generated_sequences.csv")
    df = pd.read_csv(path)
    return df["Sequence"].astype(str).str.upper().tolist()


def compute_mfe_values(label: str, sequences: list[str]) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_name(label)
    if os.path.exists(path):
        print(f"[{label}] cache hit: {path}")
        return np.load(path)

    print(f"[{label}] computing MFE for {len(sequences):,} sequences...")
    chunks = [sequences[i : i + CHUNK_SIZE] for i in range(0, len(sequences), CHUNK_SIZE)]
    mfes = []
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for chunk_mfes in pool.map(mfe_chunk, chunks):
            mfes.extend(chunk_mfes)
            done += len(chunk_mfes)
            if done % 10000 == 0 or done == len(sequences):
                print(f"  [{label}] {done:,}/{len(sequences):,}", flush=True)

    arr = np.array(mfes, dtype=np.float64)
    np.save(path, arr)
    print(f"[{label}] saved {path}")
    return arr


def summarize(label: str, mfes: np.ndarray) -> dict:
    return {
        "label": label,
        "n": len(mfes),
        "mean": float(mfes.mean()),
        "median": float(np.median(mfes)),
        "std": float(mfes.std()),
        "min": float(mfes.min()),
        "max": float(mfes.max()),
        "frac_le_-10": float((mfes <= -10).mean()),
    }


def plot_overview(data: dict[str, np.ndarray], out_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    order = ["Sequence_Craft", "Negatives"] + GENERATED
    for label in order:
        mfes = data[label]
        ax.hist(
            mfes,
            bins=BINS,
            density=True,
            alpha=0.35,
            color=COLORS[label],
            label=f"{label} (n={len(mfes):,})",
            edgecolor="white",
            linewidth=0.2,
        )
    ax.set_title("MFE distributions: references vs nofilter generated (NUPACK, material=dna)")
    ax.set_xlabel("MFE (kcal/mol)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_panels(data: dict[str, np.ndarray], out_path: str):
    ref_labels = ["Sequence_Craft", "Negatives"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, gen_label in zip(axes.ravel(), GENERATED):
        for ref in ref_labels:
            ax.hist(
                data[ref],
                bins=BINS,
                density=True,
                alpha=0.25,
                color=COLORS[ref],
                label=ref,
                edgecolor="white",
                linewidth=0.2,
            )
        ax.hist(
            data[gen_label],
            bins=BINS,
            density=True,
            alpha=0.55,
            color=COLORS[gen_label],
            label=gen_label,
            edgecolor="white",
            linewidth=0.2,
        )
        s = summarize(gen_label, data[gen_label])
        ax.set_title(
            f"{gen_label}\nμ={s['mean']:.2f}, med={s['median']:.2f}, "
            f"≤−10: {100*s['frac_le_-10']:.1f}%"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.supxlabel("MFE (kcal/mol)")
    fig.supylabel("Density")
    fig.suptitle("Each generated model vs Sequence_Craft & Negatives", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    data = {}
    stats = []

    for label, (path, col) in REFERENCES.items():
        seqs = load_sequences_csv(path, col)
        data[label] = compute_mfe_values(label, seqs)
        stats.append(summarize(label, data[label]))

    for label in GENERATED:
        seqs = load_generated_sequences(label)
        data[label] = compute_mfe_values(label, seqs)
        stats.append(summarize(label, data[label]))

    print("\n=== MFE summary ===")
    for s in stats:
        print(
            f"{s['label']}: n={s['n']:,}, mean={s['mean']:.3f}, median={s['median']:.3f}, "
            f"frac≤−10={100*s['frac_le_-10']:.2f}%"
        )

    stats_path = os.path.join(OUT_DIR, "mfe_nofilter_comparison_stats.txt")
    with open(stats_path, "w") as f:
        for s in stats:
            f.write(
                f"{s['label']}\t{s['n']}\t{s['mean']:.4f}\t{s['median']:.4f}\t"
                f"{s['std']:.4f}\t{s['min']:.4f}\t{s['max']:.4f}\t{s['frac_le_-10']:.4f}\n"
            )
    print(f"Saved: {stats_path}")

    plot_overview(data, os.path.join(OUT_DIR, "mfe_nofilter_vs_references_overview.png"))
    plot_panels(data, os.path.join(OUT_DIR, "mfe_nofilter_vs_references_panels.png"))


if __name__ == "__main__":
    main()
