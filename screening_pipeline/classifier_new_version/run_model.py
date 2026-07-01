from pathlib import Path

from classifier_utils import run_pipeline

BASE = Path(__file__).resolve().parent

if __name__ == "__main__":
    run_pipeline(
        input_csv_path=str(BASE / "data" / "X_train.csv"),
        first_output=str(BASE / "embeddings.csv"),
        second_output=str(BASE / "classification_results.csv"),
        model_name="zhihan1996/DNABERT-2-117M",
        model_path=str(BASE / "model" / "LGBM_model.pkl"),
        sequence_column="sequence",
    )

