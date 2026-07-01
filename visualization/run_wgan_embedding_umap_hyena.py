#!/usr/bin/env python3
"""HyenaDNA embedding UMAP/PCA/t-SNE for WGAN fine-tuned approaches."""

from __future__ import annotations

import os
import re
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIG_OUT = SCRIPT_DIR / "hyena_embedding_plots"
FIG_OUT.mkdir(parents=True, exist_ok=True)
GEN = REPO_ROOT / "generated"

PATH_DISTR_TXT = GEN / "eds_ft_nofilter/generated_sequences.csv"
PATH_DISTR_VAL = GEN / "eds_ft_nofilter/eds_ft_nofilter_after_sec_str.csv"
PATH_MFE_TXT = GEN / "mfe_ft_nofilter/generated_sequences.csv"
PATH_MFE_VAL = GEN / "mfe_ft_nofilter/mfe_ft_nofilter_after_sec_str.csv"
PATH_REAL_TXT = GEN / "sequence_craft_nofilter/generated_sequences.csv"
PATH_REAL_VAL = GEN / "sequence_craft_nofilter/cs_q1_after_validation.csv"
PATH_REAL_POOL_CSV = PATH_REAL_TXT
PATH_SEQCRAFT = REPO_ROOT / "Data/Sequence_Craft/SequenceCraft_dataset.csv"
PATH_DNA_SEQUENCES = REPO_ROOT / "Data/Negatives/dna_sequences.csv"
PATH_NEW_FALSE_AFTER_VAL = GEN / "negatives/new_false_results.csv"

RANDOM_SEED = 42
N_BG = 25_000
BATCH = 16
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.08
UMAP_METRIC = "cosine"
TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000

COLOR_BG = "#98B1A8"
COLOR_SCREEN = "#E5BA41"
COLOR_SEQCRAFT = "#D1855C"
FIG_SAVE_DPI = 800
AX_LABEL_FONTSIZE = 14
AX_TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 11

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_CACHE: dict[str, dict] = {}


def _as_path(p: Path | str) -> Path:
    return p if isinstance(p, Path) else Path(p)


def parse_wgan_txt(path: Path | str) -> list[str]:
    path = _as_path(path)
    if not path.is_file():
        return []
    out = []
    pat = re.compile(r"^Sample\s+\d+:\s*([ACGTacgt`]+)")
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = pat.match(line)
            if m:
                s = m.group(1).replace("`", "").strip().upper()
                if re.fullmatch(r"[ACGT]+", s):
                    out.append(s)
            elif re.fullmatch(r"[ACGT]+", line.upper()):
                out.append(line.upper())
    return out


def _sequence_series(df: pd.DataFrame) -> pd.Series:
    for col in ("sequence", "Sequence", "e"):
        if col in df.columns:
            return df[col]
    return df.iloc[:, 0]


def _dna_from_series(series: pd.Series) -> list[str]:
    out = []
    for v in series.dropna():
        s = re.sub(r"\s+", "", str(v)).strip().upper()
        if re.fullmatch(r"[ACGT]+", s):
            out.append(s)
    return out


def load_generated_pool_csv(path: Path | str) -> list[str]:
    path = _as_path(path)
    df = pd.read_csv(path)
    return _dna_from_series(_sequence_series(df))


def load_generated_pool(path: Path | str) -> list[str]:
    path = _as_path(path)
    if not path.is_file():
        return []
    if path.suffix.lower() == ".csv":
        return load_generated_pool_csv(path)
    return parse_wgan_txt(path)


def load_screened(path: Path | str) -> list[str]:
    path = _as_path(path)
    df = pd.read_csv(path)
    return list(dict.fromkeys(_dna_from_series(_sequence_series(df))))


def load_sequencecraft(path: Path | str) -> list[str]:
    path = _as_path(path)
    df = pd.read_csv(path)
    if "e" not in df.columns:
        raise ValueError("Expected column 'e' in SequenceCraft_dataset.csv")
    seqs = []
    for s in df["e"].dropna().astype(str):
        s = re.sub(r"\s+", "", s).upper()
        if re.fullmatch(r"[ACGT]+", s):
            seqs.append(s)
    return list(dict.fromkeys(seqs))


def load_prescreen_sequences(path: Path | str) -> list[str]:
    path = _as_path(path)
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    col = "sequence" if "sequence" in df.columns else df.columns[-1]
    out = []
    for s in df[col].dropna().astype(str):
        s = s.strip().upper()
        if re.fullmatch(r"[ACGT]+", s):
            out.append(s)
    return out


def sample_background(all_generated: list[str], screened: set[str], n: int) -> list[str]:
    pool = [s for s in all_generated if s not in screened]
    if len(pool) < n:
        pool = list(all_generated)
    if len(pool) <= n:
        return pool
    return random.sample(pool, n)


def resolve_real_generation_pool() -> Path:
    env_csv = os.environ.get("WGAN_REAL_POOL_CSV", "").strip()
    if env_csv:
        ep = Path(env_csv).expanduser()
        if ep.is_file():
            print("WGAN_real: pool from WGAN_REAL_POOL_CSV:", ep)
            return ep
        print("WARN: WGAN_REAL_POOL_CSV set but file not found:", ep)
    for candidate in (PATH_REAL_POOL_CSV, PATH_REAL_TXT, PATH_DISTR_TXT):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Need WGAN_real pool, e.g. {PATH_REAL_POOL_CSV}")


def masked_mean_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    m = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * m).sum(dim=1)
    denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(last_hidden.dtype)
    return summed / denom


def ensure_attention_mask(enc: dict, tokenizer) -> torch.Tensor:
    if "attention_mask" in enc:
        return enc["attention_mask"]
    ids = enc["input_ids"]
    pad = tokenizer.pad_token_id
    if pad is None:
        return torch.ones_like(ids, dtype=torch.long, device=ids.device)
    return (ids != pad).to(dtype=torch.long)


@torch.inference_mode()
def embed_batches_hyena(model, tokenizer, sequences: list[str], max_length: int, batch_size: int) -> np.ndarray:
    model.eval()
    embs = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="HyenaDNA"):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        mask = ensure_attention_mask(enc, tokenizer)
        out = model(input_ids=enc["input_ids"], output_hidden_states=True)
        h = out.hidden_states[-1]
        vec = masked_mean_hidden(h, mask)
        embs.append(vec.float().cpu().numpy())
    return np.vstack(embs)


def load_hyena_model():
    hyena_id = "LongSafari/hyenadna-small-32k-seqlen-hf"
    tok_h = AutoTokenizer.from_pretrained(hyena_id, trust_remote_code=True)
    mod_h = AutoModelForCausalLM.from_pretrained(
        hyena_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    mod_h.eval()
    max_h = getattr(tok_h, "model_max_length", None) or 32768
    max_h = min(max_h, 8192)
    return mod_h, tok_h, max_h


def _style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=AX_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AX_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=AX_TICK_FONTSIZE)


def _scatter_layers(ax, coords: np.ndarray, labels: np.ndarray) -> None:
    bg = coords[labels == 0]
    sc = coords[labels == 1]
    cr = coords[labels == 2]
    if len(bg):
        ax.scatter(bg[:, 0], bg[:, 1], c=COLOR_BG, s=8, alpha=0.45, linewidths=0, label="Generated (subsample, not screened)")
    if len(cr):
        ax.scatter(cr[:, 0], cr[:, 1], c=COLOR_SEQCRAFT, s=22, alpha=0.9, linewidths=0, label="SequenceCraft (real DNAzymes)")
    if len(sc):
        ax.scatter(sc[:, 0], sc[:, 1], c=COLOR_SCREEN, s=18, alpha=0.95, linewidths=0, label="Passed screening pipeline")


def fit_umap_plot(coords: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_layers(ax, coords, labels)
    _style_axes(ax, "UMAP-1", "UMAP-2")
    ax.legend(loc="best", framealpha=0.92, fontsize=LEGEND_FONTSIZE)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=FIG_SAVE_DPI)
    plt.close(fig)


def fit_tsne_plot(coords: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_layers(ax, coords, labels)
    _style_axes(ax, "t-SNE-1", "t-SNE-2")
    ax.legend(loc="best", framealpha=0.92, fontsize=LEGEND_FONTSIZE)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=FIG_SAVE_DPI)
    plt.close(fig)


def fit_pca_plot(coords: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_layers(ax, coords, labels)
    _style_axes(ax, "PCA-1", "PCA-2")
    ax.legend(loc="best", framealpha=0.92, fontsize=LEGEND_FONTSIZE)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=FIG_SAVE_DPI)
    plt.close(fig)


def run_panel(
    name: str,
    path_txt: Path | str,
    path_val: Path | str,
    seqcraft: list[str],
    pack_hyena,
    *,
    require_full_generation_txt: bool = False,
) -> None:
    path_txt = _as_path(path_txt)
    path_val = _as_path(path_val)
    print(f"=== Loading {name} ===")
    screened = set(load_screened(path_val))
    sc_set = set(seqcraft)
    print("parsing generated pool (txt/csv; may take a minute)...")
    generated = load_generated_pool(path_txt)
    if require_full_generation_txt:
        if not path_txt.is_file():
            raise FileNotFoundError(f"Missing generation pool: {path_txt}")
        if len(generated) == 0:
            raise ValueError(f"No valid sequences in {path_txt}")
    print(f"generated sequences: {len(generated)}, screened unique: {len(screened)}")
    bg = sample_background(generated, screened, N_BG)
    uniq: list[str] = []
    seen = set()
    for s in bg + list(screened) + seqcraft:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    y = np.array([2 if s in sc_set else (1 if s in screened else 0) for s in uniq], dtype=np.int8)

    mod_h, tok_h, max_h = pack_hyena
    E = embed_batches_hyena(mod_h, tok_h, uniq, max_h, BATCH)
    EMBEDDING_CACHE[name] = {"E": np.asarray(E), "y": np.asarray(y)}

    reducer = umap.UMAP(
        n_neighbors=min(UMAP_N_NEIGHBORS, max(5, len(uniq) // 4)),
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED,
    )
    coords = reducer.fit_transform(E)
    out_path = FIG_OUT / f"umap_{name}_hyena.png"
    fit_umap_plot(coords, y, f"{name} — HyenaDNA", out_path)
    print("saved", out_path)


def run_pca_all() -> None:
    for name, data in EMBEDDING_CACHE.items():
        E = data["E"]
        y = data["y"]
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        coords = pca.fit_transform(E)
        out_path = FIG_OUT / f"pca_{name}_hyena.png"
        fit_pca_plot(coords, y, f"{name} — HyenaDNA (PCA)", out_path)
        print("saved", out_path, "var_ratio:", pca.explained_variance_ratio_)


def run_tsne_all() -> None:
    for name, data in EMBEDDING_CACHE.items():
        E = data["E"]
        y = data["y"]
        n = len(E)
        perplexity = min(TSNE_PERPLEXITY, max(5, (n - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=float(perplexity),
            max_iter=TSNE_MAX_ITER,
            random_state=RANDOM_SEED,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(E)
        out_path = FIG_OUT / f"tsne_{name}_hyena.png"
        fit_tsne_plot(coords, y, f"{name} — HyenaDNA (t-SNE)", out_path)
        print("saved", out_path)


def run_pca_new_false(seqcraft: list[str], pack_hyena) -> None:
    path_pre = _as_path(PATH_DNA_SEQUENCES)
    path_val = _as_path(PATH_NEW_FALSE_AFTER_VAL)
    if not path_pre.is_file():
        raise FileNotFoundError(f"Missing pre-screen pool: {path_pre}")
    if not path_val.is_file():
        raise FileNotFoundError(f"Missing negatives after screening: {path_val}")

    screened = set(load_screened(path_val))
    print("reading dna_sequences.csv (may take a while)...")
    generated_list = load_prescreen_sequences(path_pre)
    if not generated_list:
        raise ValueError(f"No sequences extracted from {path_pre}")
    print(f"pre-screen pool: {len(generated_list)}, screened unique: {len(screened)}")

    sc_set = set(seqcraft)
    bg = sample_background(generated_list, screened, N_BG)
    uniq: list[str] = []
    seen = set()
    for s in bg + list(screened) + seqcraft:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    y = np.array([2 if s in sc_set else (1 if s in screened else 0) for s in uniq], dtype=np.int8)

    mod_h, tok_h, max_h = pack_hyena
    E = embed_batches_hyena(mod_h, tok_h, uniq, max_h, BATCH)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(E)
    out_path = FIG_OUT / "pca_new_false_hyena.png"
    fit_pca_plot(coords, y, "new_false — HyenaDNA (PCA)", out_path)
    print("saved", out_path)
    print("explained variance ratio:", pca.explained_variance_ratio_)


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-12, None)


def similarity_screened_vs_seqcraft(E: np.ndarray, y: np.ndarray) -> dict:
    scr = E[y == 1]
    cr = E[y == 2]
    out: dict = {"n_screened": int(len(scr)), "n_seqcraft": int(len(cr))}
    if len(scr) == 0 or len(cr) == 0:
        out["error"] = "empty screened or SequenceCraft"
        return out

    scr_n = l2_normalize_rows(scr)
    cr_n = l2_normalize_rows(cr)
    sim = scr_n @ cr_n.T

    nn_scr_to_cr = sim.max(axis=1)
    out["screened_to_nearest_seqcraft_cosine_mean"] = float(nn_scr_to_cr.mean())
    out["screened_to_nearest_seqcraft_cosine_median"] = float(np.median(nn_scr_to_cr))
    out["screened_to_nearest_seqcraft_cosine_p10"] = float(np.percentile(nn_scr_to_cr, 10))
    out["screened_to_nearest_seqcraft_cosine_p90"] = float(np.percentile(nn_scr_to_cr, 90))

    nn_cr_to_scr = sim.max(axis=0)
    out["seqcraft_to_nearest_screened_cosine_mean"] = float(nn_cr_to_scr.mean())
    out["seqcraft_to_nearest_screened_cosine_median"] = float(np.median(nn_cr_to_scr))

    c_s = scr_n.mean(axis=0)
    c_c = cr_n.mean(axis=0)
    c_s = c_s / (np.linalg.norm(c_s) + 1e-12)
    c_c = c_c / (np.linalg.norm(c_c) + 1e-12)
    out["centroid_cosine_similarity"] = float(np.dot(c_s, c_c))

    if len(cr) >= 2:
        sim_intra_cr = cr_n @ cr_n.T
        tri = np.triu_indices(len(cr), k=1)
        intra = sim_intra_cr[tri]
        out["seqcraft_intra_cosine_mean"] = float(intra.mean())
        out["seqcraft_intra_cosine_median"] = float(np.median(intra))
    if len(scr) >= 2:
        sim_intra_s = scr_n @ scr_n.T
        tri = np.triu_indices(len(scr), k=1)
        intra_s = sim_intra_s[tri]
        out["screened_intra_cosine_mean"] = float(intra_s.mean())
        out["screened_intra_cosine_median"] = float(np.median(intra_s))

    d2 = np.sum((scr[:, None, :] - cr[None, :, :]) ** 2, axis=2)
    min_eucl_scr = np.sqrt(d2.min(axis=1))
    min_eucl_cr = np.sqrt(d2.min(axis=0))
    out["screened_to_nearest_seqcraft_l2_mean"] = float(min_eucl_scr.mean())
    out["screened_to_nearest_seqcraft_l2_median"] = float(np.median(min_eucl_scr))
    out["seqcraft_to_nearest_screened_l2_mean"] = float(min_eucl_cr.mean())
    return out


def save_similarity_metrics() -> None:
    rows = []
    for name, data in EMBEDDING_CACHE.items():
        m = similarity_screened_vs_seqcraft(data["E"], data["y"])
        m["panel"] = name
        rows.append(m)
    dfm = pd.DataFrame(rows).set_index("panel")
    out_csv = FIG_OUT / "similarity_metrics.csv"
    dfm.to_csv(out_csv)
    print("saved", out_csv)
    print(dfm.round(4))


def main() -> None:
    print("device:", device)
    print("output dir:", FIG_OUT)
    print("negatives after screening:", PATH_NEW_FALSE_AFTER_VAL)

    seqcraft = load_sequencecraft(PATH_SEQCRAFT)
    print("SequenceCraft sequences:", len(seqcraft))

    pack_hyena = load_hyena_model()

    run_panel("WGAN_distr", PATH_DISTR_TXT, PATH_DISTR_VAL, seqcraft, pack_hyena)
    run_panel("WGAN_mfe", PATH_MFE_TXT, PATH_MFE_VAL, seqcraft, pack_hyena)
    run_panel(
        "WGAN_real",
        resolve_real_generation_pool(),
        PATH_REAL_VAL,
        seqcraft,
        pack_hyena,
        require_full_generation_txt=True,
    )

    run_pca_all()
    run_tsne_all()
    run_pca_new_false(seqcraft, pack_hyena)
    save_similarity_metrics()
    print("done.")


if __name__ == "__main__":
    main()
