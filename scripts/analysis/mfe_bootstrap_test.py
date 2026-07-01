#!/usr/bin/env python3
"""Bootstrap equal-n comparison of MFE: Sequence_Craft vs Negatives."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "generated", "mfe_cache")
OUT_DIR = os.path.join(PROJECT_ROOT, "generated")
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def load_mfe(name: str) -> np.ndarray:
    path = os.path.join(CACHE_DIR, f"{name}_mfe.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cache {path}. Run compute_mfe_distribution.py first.")
    return np.load(path)


def bootstrap_equal_n(a: np.ndarray, b: np.ndarray, n: int, n_boot: int, stat_fn):
    """Bootstrap difference stat_fn(a) - stat_fn(b) with equal sample size n."""
    observed = stat_fn(a) - stat_fn(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = RNG.choice(a, size=n, replace=True)
        sb = RNG.choice(b, size=n, replace=True)
        diffs[i] = stat_fn(sa) - stat_fn(sb)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided percentile p-value: fraction of bootstrap diffs at least as extreme as observed
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p_value = min(p_value, 1.0)
    return {
        "observed": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "boot_mean": float(diffs.mean()),
        "boot_std": float(diffs.std(ddof=1)),
        "boot_diffs": diffs,
    }


def permutation_equal_n(a: np.ndarray, b: np.ndarray, n: int, n_perm: int, stat_fn):
    """Permutation test on equal subsamples (reference)."""
    idx_a = RNG.choice(len(a), size=n, replace=False)
    idx_b = RNG.choice(len(b), size=n, replace=False)
    sa, sb = a[idx_a], b[idx_b]
    observed = stat_fn(sa) - stat_fn(sb)
    combined = np.concatenate([sa, sb])
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = RNG.permutation(combined)
        diffs[i] = stat_fn(perm[:n]) - stat_fn(perm[n:])
    p_value = np.mean(np.abs(diffs) >= np.abs(observed))
    return {"observed": float(observed), "p_value": float(p_value)}


def plot_bootstrap_results(results: dict, n: int, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (label, res) in zip(axes, results.items()):
        diffs = res["boot_diffs"]
        ax.hist(diffs, bins=60, density=True, color="#6366f1", alpha=0.7, edgecolor="white")
        ax.axvline(res["observed"], color="#dc2626", linewidth=2, label=f"observed = {res['observed']:.3f}")
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(res["ci_low"], color="#2563eb", linestyle=":", linewidth=1.5)
        ax.axvline(res["ci_high"], color="#2563eb", linestyle=":", linewidth=1.5,
                   label=f"95% CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
        ax.set_title(f"Bootstrap Δ{label} (n={n:,} per group, B={N_BOOT:,})")
        ax.set_xlabel(f"Δ{label} (Sequence_Craft − Negatives, kcal/mol)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Equal-n bootstrap: Sequence_Craft vs Negatives MFE (NUPACK, material=dna)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_equal_n_densities(sc: np.ndarray, neg: np.ndarray, n: int, out_path: str):
    """Single bootstrap draw for visual comparison at equal n."""
    sa = RNG.choice(sc, size=n, replace=True)
    sb = RNG.choice(neg, size=n, replace=True)
    bins = np.linspace(-20, 2, 50)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sa, bins=bins, density=True, alpha=0.5, color="#2563eb",
            label=f"Sequence_Craft (n={n})")
    ax.hist(sb, bins=bins, density=True, alpha=0.5, color="#dc2626",
            label=f"Negatives (n={n})")
    ax.set_xlabel("MFE (kcal/mol)")
    ax.set_ylabel("Density")
    ax.set_title("Equal-n bootstrap subsample (one replicate, for visualization)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    sc = load_mfe("sequence_craft")
    neg = load_mfe("negatives")
    n = min(len(sc), len(neg))

    print(f"Sequence_Craft: n={len(sc)}")
    print(f"Negatives:      n={len(neg)}")
    print(f"Equal bootstrap sample size: n={n}\n")

    # Full-sample (for reference — inflated power for Negatives)
    mw_full = stats.mannwhitneyu(sc, neg, alternative="two-sided")
    print("--- Full sample (reference, unequal n) ---")
    print(f"Mean diff (SC−Neg): {sc.mean() - neg.mean():.4f} kcal/mol")
    print(f"Median diff:          {np.median(sc) - np.median(neg):.4f} kcal/mol")
    print(f"Mann-Whitney U p:     {mw_full.pvalue:.2e}\n")

    stats_fns = {"mean": np.mean, "median": np.median}
    report = {
        "n_sequence_craft": len(sc),
        "n_negatives": len(neg),
        "bootstrap_n_per_group": n,
        "n_bootstrap": N_BOOT,
        "full_sample": {
            "mean_diff": float(sc.mean() - neg.mean()),
            "median_diff": float(np.median(sc) - np.median(neg)),
            "mannwhitney_p": float(mw_full.pvalue),
        },
        "bootstrap_equal_n": {},
        "permutation_equal_n": {},
    }

    boot_results = {}
    print("--- Equal-n bootstrap ---")
    for name, fn in stats_fns.items():
        res = bootstrap_equal_n(sc, neg, n, N_BOOT, fn)
        boot_results[name] = res
        report["bootstrap_equal_n"][name] = {k: v for k, v in res.items() if k != "boot_diffs"}
        sig = "significant" if res["ci_low"] > 0 or res["ci_high"] < 0 else "not significant"
        print(
            f"Δ{name}: observed={res['observed']:.4f}, "
            f"95% CI=[{res['ci_low']:.4f}, {res['ci_high']:.4f}], "
            f"p={res['p_value']:.4f} ({sig})"
        )

        perm = permutation_equal_n(sc, neg, n, N_BOOT, fn)
        report["permutation_equal_n"][name] = perm
        print(f"  permutation p ({name}): {perm['p_value']:.4f}")

    # Mann-Whitney on one equal-n subsample (reference)
    idx_sc = RNG.choice(len(sc), size=n, replace=False)
    idx_neg = RNG.choice(len(neg), size=n, replace=False)
    mw_eq = stats.mannwhitneyu(sc[idx_sc], neg[idx_neg], alternative="two-sided")
    report["equal_n_subsample_mannwhitney_p"] = float(mw_eq.pvalue)
    print(f"\nMann-Whitney (one equal-n subsample n={n}): p={mw_eq.pvalue:.4f}")

    plot_bootstrap_results(
        boot_results,
        n,
        os.path.join(OUT_DIR, "mfe_bootstrap_equal_n.png"),
    )
    plot_equal_n_densities(
        sc, neg, n,
        os.path.join(OUT_DIR, "mfe_equal_n_subsample.png"),
    )

    json_path = os.path.join(OUT_DIR, "mfe_bootstrap_test_results.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
