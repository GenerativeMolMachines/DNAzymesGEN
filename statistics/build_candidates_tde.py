#!/usr/bin/env python3
"""Select top-N generated cores by min TDE vs SequenceCraft; write Candidates.

For each generated core, TDE is computed against *all* SequenceCraft structures;
``min_tde`` is the nearest-neighbor distance. Keep the top-N cores with the
smallest ``min_tde``, then write one arms CSV per target/site.

Output layout::

  Candidates/
    eds_ft_nofilter/
      T1_site1_with_arms.csv
      T1_site2_with_arms.csv
      T2_with_arms.csv
    mfe_ft_nofilter/
      ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STATS_DIR))

from compute_tde import compute_tde  # noqa: E402

DEFAULT_REF = PROJECT_ROOT / "Data/Sequence_Craft/sequence_craft_dbs.csv"
DEFAULT_OUT = PROJECT_ROOT / "Candidates"

METHODS = {
    "eds_ft_nofilter": {
        "cores": PROJECT_ROOT
        / "generated/eds_ft_nofilter/eds_ft_nofilter_after_sec_str.csv",
        "arms": {
            "T1_site1": PROJECT_ROOT
            / "generated/eds_ft_nofilter/eds_ft_nofilter_T1_site1_with_arms.csv",
            "T1_site2": PROJECT_ROOT
            / "generated/eds_ft_nofilter/eds_ft_nofilter_T1_site2_with_arms.csv",
            "T2": PROJECT_ROOT
            / "generated/eds_ft_nofilter/eds_ft_nofilter_T2_with_arms.csv",
        },
    },
    "mfe_ft_nofilter": {
        "cores": PROJECT_ROOT
        / "generated/mfe_ft_nofilter/mfe_ft_nofilter_after_sec_str.csv",
        "arms": {
            "T1_site1": PROJECT_ROOT
            / "generated/mfe_ft_nofilter/mfe_ft_nofilter_T1_site1_with_arms.csv",
            "T1_site2": PROJECT_ROOT
            / "generated/mfe_ft_nofilter/mfe_ft_nofilter_T1_site2_with_arms.csv",
            "T2": PROJECT_ROOT
            / "generated/mfe_ft_nofilter/mfe_ft_nofilter_T2_with_arms.csv",
        },
    },
}

ARMS_COLUMNS = [
    "core_index",
    "catalytic_core",
    "target_id",
    "site_id",
    "cleavage_pos",
    "cleavage_dinucleotide",
    "left_arm",
    "right_arm",
    "left_arm_len",
    "right_arm_len",
    "left_tm",
    "right_tm",
    "left_tm_delta",
    "right_tm_delta",
    "dnazyme_5to3",
    "dnazyme_len",
]

TDE_COLUMNS = [
    "db_structure",
    "min_tde",
    "min_tde_sequence",
    "min_tde_db",
]


def _min_tde_batch(structures: list[str], ref_df: pd.DataFrame) -> pd.DataFrame:
    """For each query structure, find nearest SequenceCraft by TDE.

    Uses ViennaRNA Python trees when available (much faster for batch);
    otherwise falls back to pairwise ``compute_tde``.
    """
    ref_seqs = ref_df["sequence"].astype(str).tolist()
    ref_dbs = ref_df["dot_bracket_structure"].astype(str).tolist()

    try:
        import RNA

        ref_trees = [RNA.make_tree(RNA.expand_Full(s)) for s in ref_dbs]

        rows = []
        for i, db in enumerate(structures):
            tree = RNA.make_tree(RNA.expand_Full(db))
            dists = [int(RNA.tree_edit_distance(tree, rt)) for rt in ref_trees]
            j = min(range(len(dists)), key=dists.__getitem__)
            rows.append(
                {
                    "db_structure": db,
                    "min_tde": dists[j],
                    "min_tde_sequence": ref_seqs[j],
                    "min_tde_db": ref_dbs[j],
                }
            )
            if (i + 1) % 50 == 0 or i + 1 == len(structures):
                print(f"  TDE progress: {i + 1}/{len(structures)}")
        return pd.DataFrame(rows)
    except ImportError:
        rows = []
        for i, db in enumerate(structures):
            dists = [compute_tde(db, ref_db) for ref_db in ref_dbs]
            j = min(range(len(dists)), key=dists.__getitem__)
            rows.append(
                {
                    "db_structure": db,
                    "min_tde": dists[j],
                    "min_tde_sequence": ref_seqs[j],
                    "min_tde_db": ref_dbs[j],
                }
            )
            if (i + 1) % 20 == 0 or i + 1 == len(structures):
                print(f"  TDE progress: {i + 1}/{len(structures)}")
        return pd.DataFrame(rows)


def load_cores_with_tde(cores_csv: Path, ref_df: pd.DataFrame) -> pd.DataFrame:
    cores = pd.read_csv(cores_csv)
    if "db_structure" not in cores.columns or "sequence" not in cores.columns:
        raise KeyError(f"{cores_csv}: need columns sequence, db_structure")

    # Preserve row order as core_index (same as design_binding_arms.py).
    cores = cores.reset_index(drop=True)
    cores.insert(0, "core_index", cores.index)

    print(f"Computing min TDE for {len(cores)} cores vs {len(ref_df)} SequenceCraft...")
    tde = _min_tde_batch(cores["db_structure"].astype(str).tolist(), ref_df)
    out = cores[["core_index", "sequence"]].copy()
    out["catalytic_core"] = out["sequence"].str.upper().str.replace("U", "T")
    out = out.join(tde)
    return out


def select_top_by_tde(cores_tde: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep top_n cores with smallest min_tde (ties broken by core_index)."""
    ranked = cores_tde.sort_values(
        ["min_tde", "core_index"], ascending=[True, True]
    ).head(top_n)
    return ranked.reset_index(drop=True)


def build_method(
    method: str,
    cfg: dict,
    ref_df: pd.DataFrame,
    out_root: Path,
    top_n: int,
) -> None:
    print(f"\n===== {method} =====")
    cores_tde = load_cores_with_tde(cfg["cores"], ref_df)
    top = select_top_by_tde(cores_tde, top_n)
    print(
        f"Selected top-{len(top)} by min_tde: "
        f"values={top['min_tde'].tolist()} "
        f"core_index={top['core_index'].tolist()}"
    )

    method_dir = out_root / method
    method_dir.mkdir(parents=True, exist_ok=True)

    tde_by_index = top.set_index("core_index")
    keep_idx = set(top["core_index"].tolist())

    for site_key, arms_csv in cfg["arms"].items():
        if not arms_csv.exists():
            raise FileNotFoundError(arms_csv)
        arms = pd.read_csv(arms_csv)
        missing = [c for c in ARMS_COLUMNS if c not in arms.columns]
        if missing:
            raise KeyError(f"{arms_csv}: missing columns {missing}")

        arms_top = arms[arms["core_index"].isin(keep_idx)].copy()
        merged = arms_top.merge(
            tde_by_index[TDE_COLUMNS],
            left_on="core_index",
            right_index=True,
            how="left",
            validate="many_to_one",
        )
        # Keep the same ranking order as top-N by TDE.
        order = {idx: rank for rank, idx in enumerate(top["core_index"])}
        merged["_rank"] = merged["core_index"].map(order)
        merged = merged.sort_values("_rank").drop(columns="_rank")

        if len(merged) != len(top):
            raise RuntimeError(
                f"{method}/{site_key}: expected {len(top)} rows, got {len(merged)}"
            )
        if merged["min_tde"].isna().any():
            raise RuntimeError(f"{method}/{site_key}: missing TDE after merge")

        out_path = method_dir / f"{site_key}_with_arms.csv"
        merged[ARMS_COLUMNS + TDE_COLUMNS].to_csv(out_path, index=False)
        print(
            f"wrote {out_path}  rows={len(merged)}  "
            f"min_tde={merged['min_tde'].tolist()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select top-N generated cores by min TDE vs SequenceCraft "
            "and write Candidates CSVs with arms."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REF,
        help="SequenceCraft CSV with sequence + dot_bracket_structure",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUT,
        help="Output Candidates directory",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of lowest-min_tde cores to keep (default 10).",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(METHODS),
        choices=list(METHODS),
    )
    args = parser.parse_args()

    ref = pd.read_csv(args.reference)
    need = {"sequence", "dot_bracket_structure"}
    if not need.issubset(ref.columns):
        raise KeyError(f"{args.reference}: need columns {sorted(need)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        build_method(
            method, METHODS[method], ref, args.output_root, top_n=args.top_n
        )

    print(f"\nDone. Candidates root: {args.output_root}")


if __name__ == "__main__":
    main()
