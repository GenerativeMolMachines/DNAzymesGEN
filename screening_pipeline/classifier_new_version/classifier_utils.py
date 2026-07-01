import os

import joblib
import lightgbm as lgb
import pandas as pd
import torch
import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from constants import selected_features


def load_data(csv_path: str, sequence_column: str):
    df = pd.read_csv(csv_path)
    sequences = df[sequence_column].dropna().astype(str).unique()
    return sequences


def initialize_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # Force PyTorch attention: Triton flash-attn fails on this GPU/toolchain.
    config.attention_probs_dropout_prob = max(
        getattr(config, "attention_probs_dropout_prob", 0.0), 0.01
    )
    model = AutoModel.from_pretrained(
        model_name, config=config, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return model, tokenizer, device


def embed_sequences(sequences, model, tokenizer, device, batch_size: int = 64):
    results = []
    seq_list = list(sequences)

    with torch.no_grad():
        for start in tqdm.tqdm(
            range(0, len(seq_list), batch_size),
            desc="Embedding sequences",
            total=(len(seq_list) + batch_size - 1) // batch_size,
        ):
            batch = seq_list[start : start + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(**tokens)
            hidden_states = outputs[0]
            emb = torch.mean(hidden_states, dim=1)
            emb = emb / emb.norm(dim=1, keepdim=True)

            for sequence, vector in zip(batch, emb.cpu().tolist()):
                results.append({"sequence": sequence, "embedding": vector})

    return results


def build_final_dataframe(results: list):
    df = pd.DataFrame(results)
    emb_df = pd.DataFrame(df["embedding"].tolist())
    emb_df.columns = [f"emb_{i}" for i in emb_df.columns]
    return pd.concat([df["sequence"], emb_df], axis=1)


def load_model(model_path: str):
    return joblib.load(model_path)


def predict_on_embeddings(df, model_path: str, output_path: str):
    model = load_model(model_path)
    X = df[selected_features]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["y_pred"] = y_pred
    df["y_proba"] = y_proba

    result_df = df[["sequence", "y_pred", "y_proba"]]
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to: {os.path.abspath(output_path)}")
    return result_df


def run_pipeline(
    input_csv_path: str,
    first_output: str,
    second_output: str,
    model_name: str,
    model_path: str,
    sequence_column: str,
    batch_size: int = 64,
    save_embeddings: bool = True,
):
    sequences = load_data(input_csv_path, sequence_column)
    print(f"Loaded {len(sequences)} unique sequences from {input_csv_path}")

    model, tokenizer, device = initialize_model(model_name)
    print(f"DNABERT device: {device}")

    results = embed_sequences(
        sequences, model, tokenizer, device, batch_size=batch_size
    )
    full_df = build_final_dataframe(results)

    if save_embeddings:
        full_df.to_csv(first_output, index=False)
        print(f"Embeddings saved to: {os.path.abspath(first_output)}")

    return predict_on_embeddings(full_df, model_path, second_output)
