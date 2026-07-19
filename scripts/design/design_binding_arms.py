#!/usr/bin/env python3
"""Design DNA binding arms for catalytic cores against RNA targets.

For each arm independently: compute DNA/RNA hybrid Tm and pick length in
[min_arm, max_arm] so that Tm is closest to the target (default 37 °C).

Default buffer / concentrations:
  [Na+] = 150 mM, [Mg2+] = 10 mM, Ct = 1 µM (dnac1 = dnac2 = Ct/2).

Writes one CSV per cleavage site per input dataset.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio.SeqUtils import MeltingTemp as mt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Target 1: uppercase marks cleavage dinucleotides (GU).
TARGET1_RAW = (
    "uaa acc aca ccu caG Uca cua aag guc ugu uua aag uug uuc ugG Uug auu gcu ugu u"
)

# Target 2: IDT rN notation; lowercase marks cleavage (grur → GU).
TARGET2_RAW = (
    "rCrUrGrArArUrArUrArArArCrUrUrGrUrGrGrUrArGrUrUrGrGrArGrCrUrGrgrur"
    "GrGrCrGrUrArGrGrCrArArGrArGrUrGrCrCrUrUrGrArCrG rArUrArC"
)

RNA_COMPLEMENT = str.maketrans("ACGUacgu", "UGCAtgca")
DNA_FROM_RNA = str.maketrans("ACGUacgu", "ACGTacgt")

# Human-readable file suffixes for the three cleavage sites.
SITE_FILE_SUFFIX = {
    "T1_site1": "T1_site1",
    "T1_site2": "T1_site2",
    "T2_site1": "T2",
}


@dataclass(frozen=True)
class CleavageSite:
    target_id: str
    site_id: str
    rna: str
    cleavage_pos: int  # index of first base of cleavage dinucleotide (0-based)
    cleavage_dinucleotide: str

    @property
    def flank_5p(self) -> str:
        return self.rna[: self.cleavage_pos]

    @property
    def flank_3p(self) -> str:
        return self.rna[self.cleavage_pos + 2 :]


def normalize_rna(seq: str) -> str:
    return re.sub(r"\s+", "", seq).upper().replace("T", "U")


def parse_target1(raw: str) -> list[CleavageSite]:
    compact = re.sub(r"\s+", "", raw)
    rna = compact.upper().replace("T", "U")
    sites: list[CleavageSite] = []
    i = 0
    site_n = 1
    while i < len(compact) - 1:
        if compact[i].isupper() and compact[i + 1].isupper():
            dinuc = rna[i : i + 2]
            sites.append(
                CleavageSite(
                    target_id="T1",
                    site_id=f"T1_site{site_n}",
                    rna=rna,
                    cleavage_pos=i,
                    cleavage_dinucleotide=dinuc,
                )
            )
            site_n += 1
            i += 2
        else:
            i += 1
    return sites


def parse_target2_idt(raw: str) -> list[CleavageSite]:
    compact = re.sub(r"\s+", "", raw)
    bases: list[str] = []
    lower_mask: list[bool] = []
    i = 0
    while i < len(compact):
        ch = compact[i]
        if ch in "rR" and i + 1 < len(compact) and compact[i + 1].upper() in "ACGU":
            base = compact[i + 1]
            bases.append(base.upper())
            lower_mask.append(base.islower())
            i += 2
        elif ch.upper() in "ACGU":
            bases.append(ch.upper())
            lower_mask.append(ch.islower())
            i += 1
        else:
            i += 1

    rna = "".join(bases)
    sites: list[CleavageSite] = []
    site_n = 1
    i = 0
    while i < len(bases) - 1:
        if lower_mask[i] and lower_mask[i + 1]:
            sites.append(
                CleavageSite(
                    target_id="T2",
                    site_id=f"T2_site{site_n}",
                    rna=rna,
                    cleavage_pos=i,
                    cleavage_dinucleotide=rna[i : i + 2],
                )
            )
            site_n += 1
            i += 2
        else:
            i += 1
    return sites


def default_sites() -> list[CleavageSite]:
    return parse_target1(TARGET1_RAW) + parse_target2_idt(TARGET2_RAW)


def rna_to_dna_revcomp(rna: str) -> str:
    """DNA reverse complement of an RNA oligo (5'→3' in, 5'→3' out)."""
    rna = normalize_rna(rna)
    return rna.translate(RNA_COMPLEMENT)[::-1].translate(DNA_FROM_RNA)


def tm_arm_vs_rna(
    rna_binding_region: str,
    na: float,
    mg: float,
    dnac1: float,
    dnac2: float,
) -> float:
    """Nearest-neighbor Tm (°C) for left/right arm DNA vs its RNA flank."""
    rna = normalize_rna(rna_binding_region)
    return float(
        mt.Tm_NN(
            rna,
            nn_table=mt.R_DNA_NN1,
            Na=na,
            Mg=mg,
            dnac1=dnac1,
            dnac2=dnac2,
        )
    )


def choose_arm(
    flank_rna: str,
    *,
    from_cleavage: bool,
    min_arm: int,
    max_arm: int,
    target_tm: float,
    na: float,
    mg: float,
    dnac1: float,
    dnac2: float,
) -> tuple[str, int, float]:
    """Pick arm length with DNA/RNA Tm closest to target_tm; ties → shorter.

    from_cleavage=True  → 3' flank, take prefix (nt immediately after GU)
    from_cleavage=False → 5' flank, take suffix (nt immediately before GU)
    """
    flank = normalize_rna(flank_rna)
    if len(flank) < min_arm:
        raise ValueError(
            f"Flank length {len(flank)} < min arm {min_arm}: {flank!r}"
        )

    best: tuple[float, int, str, float] | None = None  # |dTm|, len, arm, tm
    upper = min(max_arm, len(flank))
    for length in range(min_arm, upper + 1):
        rna_seg = flank[:length] if from_cleavage else flank[-length:]
        arm = rna_to_dna_revcomp(rna_seg)
        tm = tm_arm_vs_rna(
            rna_seg, na=na, mg=mg, dnac1=dnac1, dnac2=dnac2
        )
        key = (abs(tm - target_tm), length)
        if best is None or key < (best[0], best[1]):
            best = (abs(tm - target_tm), length, arm, tm)

    assert best is not None
    _, length, arm, tm = best
    return arm, length, tm


def design_for_core(
    core: str,
    site: CleavageSite,
    *,
    min_arm: int,
    max_arm: int,
    target_tm: float,
    na: float,
    mg: float,
    dnac1: float,
    dnac2: float,
) -> dict:
    # Left arm (5' of DNAzyme) vs RNA 3' of cleavage; right arm vs RNA 5'.
    left_arm, left_len, left_tm = choose_arm(
        site.flank_3p,
        from_cleavage=True,
        min_arm=min_arm,
        max_arm=max_arm,
        target_tm=target_tm,
        na=na,
        mg=mg,
        dnac1=dnac1,
        dnac2=dnac2,
    )
    right_arm, right_len, right_tm = choose_arm(
        site.flank_5p,
        from_cleavage=False,
        min_arm=min_arm,
        max_arm=max_arm,
        target_tm=target_tm,
        na=na,
        mg=mg,
        dnac1=dnac1,
        dnac2=dnac2,
    )
    dnazyme = f"{left_arm}{core}{right_arm}"
    return {
        "core_index": None,  # filled by caller
        "catalytic_core": core,
        "target_id": site.target_id,
        "site_id": site.site_id,
        "cleavage_pos": site.cleavage_pos,
        "cleavage_dinucleotide": site.cleavage_dinucleotide,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_arm_len": left_len,
        "right_arm_len": right_len,
        "left_tm": round(left_tm, 3),
        "right_tm": round(right_tm, 3),
        "left_tm_delta": round(abs(left_tm - target_tm), 3),
        "right_tm_delta": round(abs(right_tm - target_tm), 3),
        "dnazyme_5to3": dnazyme,
        "dnazyme_len": len(dnazyme),
    }


COL_ORDER = [
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


def output_path_for_site(input_csv: Path, site: CleavageSite) -> Path:
    stem = input_csv.stem.replace("_after_sec_str", "")
    suffix = SITE_FILE_SUFFIX.get(site.site_id, site.site_id)
    return input_csv.with_name(f"{stem}_{suffix}_with_arms.csv")


def process_input(
    input_csv: Path,
    sites: list[CleavageSite],
    *,
    sequence_column: str,
    min_arm: int,
    max_arm: int,
    target_tm: float,
    na: float,
    mg: float,
    dnac1: float,
    dnac2: float,
) -> list[tuple[Path, pd.DataFrame]]:
    df = pd.read_csv(input_csv)
    if sequence_column not in df.columns:
        raise KeyError(f"{input_csv}: missing column {sequence_column!r}")

    cores: list[tuple[int, str]] = []
    for idx, core in enumerate(df[sequence_column].astype(str)):
        core = core.strip().upper().replace("U", "T")
        if not core or core == "NAN":
            continue
        cores.append((idx, core))

    written: list[tuple[Path, pd.DataFrame]] = []
    for site in sites:
        rows: list[dict] = []
        for idx, core in cores:
            row = design_for_core(
                core,
                site,
                min_arm=min_arm,
                max_arm=max_arm,
                target_tm=target_tm,
                na=na,
                mg=mg,
                dnac1=dnac1,
                dnac2=dnac2,
            )
            row["core_index"] = idx
            rows.append(row)

        out = pd.DataFrame(rows)[COL_ORDER]
        out_path = output_path_for_site(input_csv, site)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        written.append((out_path, out))
    return written


def summarize(df: pd.DataFrame, label: str, target_tm: float) -> None:
    both_ok = (df["left_tm_delta"] <= 2) & (df["right_tm_delta"] <= 2)
    print(f"\n=== {label} ===")
    print(f"rows: {len(df)}")
    print(
        f"arm lengths: left={df['left_arm_len'].iloc[0]}, "
        f"right={df['right_arm_len'].iloc[0]}"
    )
    print(
        f"arm Tm vs RNA: left={df['left_tm'].iloc[0]:.2f}°C "
        f"(Δ={df['left_tm_delta'].iloc[0]:.2f}), "
        f"right={df['right_tm'].iloc[0]:.2f}°C "
        f"(Δ={df['right_tm_delta'].iloc[0]:.2f})"
    )
    print(
        f"both arms |ΔTm|≤2°C vs {target_tm}°C: "
        f"{both_ok.mean() * 100:.1f}% ({both_ok.sum()}/{len(df)})"
    )
    print(
        f"dnazyme_len: min={df['dnazyme_len'].min()} "
        f"median={df['dnazyme_len'].median():.0f} max={df['dnazyme_len'].max()}"
    )


def default_inputs() -> list[Path]:
    return [
        PROJECT_ROOT
        / "generated/eds_ft_nofilter/eds_ft_nofilter_after_sec_str.csv",
        PROJECT_ROOT
        / "generated/mfe_ft_nofilter/mfe_ft_nofilter_after_sec_str.csv",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add DNA binding arms: tune left/right lengths so each arm's "
            "DNA/RNA Tm approaches the target (default 37 °C). "
            "Writes one CSV per cleavage site."
        )
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=None,
        help="Input CSVs with catalytic cores (default: eds_ft + mfe_ft after_sec_str).",
    )
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--target-tm", type=float, default=37.0)
    parser.add_argument("--na", type=float, default=150.0, help="[Na+] or [K+] mM")
    parser.add_argument("--mg", type=float, default=10.0, help="[Mg2+] mM")
    parser.add_argument("--min-arm", type=int, default=6)
    parser.add_argument("--max-arm", type=int, default=15)
    parser.add_argument(
        "--ct-uM",
        type=float,
        default=1.0,
        help="Total oligonucleotide concentration Ct in µM (default 1).",
    )
    parser.add_argument(
        "--dnac1",
        type=float,
        default=None,
        help="Strand 1 concentration nM (default: Ct/2).",
    )
    parser.add_argument(
        "--dnac2",
        type=float,
        default=None,
        help="Strand 2 concentration nM (default: Ct/2).",
    )
    args = parser.parse_args()

    # Ct in µM → nM; BioPython uses dnac1/dnac2 with Ct ≈ dnac1+dnac2
    # for non-self-complementary duplexes (each strand = Ct/2).
    ct_nM = args.ct_uM * 1000.0
    dnac1 = args.dnac1 if args.dnac1 is not None else ct_nM / 2.0
    dnac2 = args.dnac2 if args.dnac2 is not None else ct_nM / 2.0

    inputs = args.inputs if args.inputs is not None else default_inputs()
    sites = default_sites()
    print("Cleavage sites:")
    for s in sites:
        print(
            f"  {s.site_id}: pos={s.cleavage_pos} {s.cleavage_dinucleotide} "
            f"flank5={len(s.flank_5p)} flank3={len(s.flank_3p)}"
        )
    print(
        "Arm design: independently choose left/right lengths so each "
        f"DNA/RNA Tm → {args.target_tm}°C "
        f"(Na={args.na} mM, Mg={args.mg} mM, Ct={args.ct_uM} µM, "
        f"dnac1={dnac1:g} nM, dnac2={dnac2:g} nM)."
    )

    for input_csv in inputs:
        results = process_input(
            input_csv,
            sites,
            sequence_column=args.sequence_column,
            min_arm=args.min_arm,
            max_arm=args.max_arm,
            target_tm=args.target_tm,
            na=args.na,
            mg=args.mg,
            dnac1=dnac1,
            dnac2=dnac2,
        )
        for out_path, df in results:
            summarize(df, str(out_path), args.target_tm)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
