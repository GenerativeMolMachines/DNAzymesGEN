#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
CLASSIFIER_DIR = PIPELINE_ROOT / "classifier_new_version"
PROJECT_ROOT = PIPELINE_ROOT.parent

sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(CLASSIFIER_DIR))

from classifier_utils import run_pipeline  # noqa: E402
from levenshtein_q1 import run_levenshtein_q1  # noqa: E402

DATASETS = [
    "eds_pretrain_nofilter",
    "eds_ft_nofilter",
    "mfe_pretrain_nofilter",
    "mfe_ft_nofilter",
]

DEFAULT_ACTIVE_CSV = (
    PROJECT_ROOT / "Data" / "Sequence_Craft" / "SequenceCraft_dataset.csv"
)
DEFAULT_MODEL_PATH = CLASSIFIER_DIR / "model" / "LGBM_model.pkl"
DEFAULT_MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def resolve_output_dir(dataset: str, output_root: Path) -> Path:
    out_dir = output_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_dataset_screening(
    dataset: str,
    generated_root: Path,
    output_root: Path,
    active_csv: Path,
    model_path: Path,
    model_name: str,
    sequence_column: str,
    proba_threshold: float,
    quantile: float,
    batch_size: int,
    skip_classifier: bool,
    skip_levenshtein: bool,
    save_embeddings: bool,
):
    input_csv = generated_root / dataset / "generated_sequences.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing generated dataset: {input_csv}")

    out_dir = resolve_output_dir(dataset, output_root)
    embeddings_csv = out_dir / "embeddings.csv"
    classification_csv = out_dir / "classification_results.csv"
    q1_csv = out_dir / "levenshtein_q1.csv"

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset}")
    print(f"Input:   {input_csv}")
    print(f"Output:  {out_dir}")
    print(f"{'=' * 60}")

    if not skip_classifier:
        if classification_csv.exists():
            print(f"Skipping classifier, exists: {classification_csv}")
        else:
            run_pipeline(
                input_csv_path=str(input_csv),
                first_output=str(embeddings_csv),
                second_output=str(classification_csv),
                model_name=model_name,
                model_path=str(model_path),
                sequence_column=sequence_column,
                batch_size=batch_size,
                save_embeddings=save_embeddings,
            )
    elif not classification_csv.exists():
        raise FileNotFoundError(
            f"--skip-classifier set but missing {classification_csv}"
        )

    if not skip_levenshtein:
        run_levenshtein_q1(
            classification_csv=str(classification_csv),
            active_csv=str(active_csv),
            output_csv=str(q1_csv),
            active_column="e",
            proba_threshold=proba_threshold,
            quantile=quantile,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run DNABERT+LightGBM classifier and Levenshtein Q1 filter"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Dataset folder names under generated/",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=PROJECT_ROOT / "generated",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "generated" / "screening",
    )
    parser.add_argument("--active-csv", type=Path, default=DEFAULT_ACTIVE_CSV)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--proba-threshold", type=float, default=0.95)
    parser.add_argument("--quantile", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--skip-levenshtein", action="store_true")
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        run_dataset_screening(
            dataset=dataset,
            generated_root=args.generated_root,
            output_root=args.output_root,
            active_csv=args.active_csv,
            model_path=args.model_path,
            model_name=args.model_name,
            sequence_column=args.sequence_column,
            proba_threshold=args.proba_threshold,
            quantile=args.quantile,
            batch_size=args.batch_size,
            skip_classifier=args.skip_classifier,
            skip_levenshtein=args.skip_levenshtein,
            save_embeddings=not args.no_embeddings,
        )


if __name__ == "__main__":
    main()
