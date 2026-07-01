# DNAzymesGEN

Computational pipeline for DNAzyme discovery using generative models, structural priors, and machine-learning screening.

Despite their broad therapeutic and biotechnological potential, deoxyribozymes remain challenging to design rationally due to few consensus sequences, context-dependent activity, target inaccessibility, and a lack of large curated datasets for machine learning. This repository provides a multi-stage pipeline that combines:

- **Synthetic pre-training** on MFE-bounded or EDS-matched sequence distributions
- **Fine-tuning** on experimentally validated DNAzymes (Sequence Craft)
- **Large-scale sequence generation** from trained WGAN-GP checkpoints
- **Hierarchical screening** with an activity classifier and structural filters

Repository: [GenerativeMolMachines/DNAzymesGEN](https://github.com/GenerativeMolMachines/DNAzymesGEN)

---

## Repository layout

| Directory | Description |
|-----------|-------------|
| `Data/` | Training datasets: EDS, MFE, Sequence Craft, and negative controls |
| `improved_wgan_training/` | DNA sequence WGAN-GP training, fine-tuning, evaluation, and generation scripts |
| `FT/` | Fine-tuning helpers and MFE utilities used during generation |
| `checkpoints/` | Pre-trained and fine-tuned model checkpoints |
| `generated/` | Generated sequence pools, screening outputs, and analysis artifacts |
| `screening_pipeline/` | Classifier-based screening and secondary-structure filtering |
| `statistics/` | Dataset statistics scripts |
| `visualization/` | Embedding and UMAP visualization utilities |

See [`generated/WORKFLOW.md`](generated/WORKFLOW.md) for checkpoint selection rationale and the full training/generation timeline.

---

## Prerequisites

- Linux with NVIDIA GPU (CUDA 11.8 recommended)
- [Git LFS](https://git-lfs.com/) — required for checkpoints and large datasets
- [Conda](https://docs.conda.io/) (Miniconda or Mambaforge)
- [NUPACK 4](https://www.nupack.org/) — **not included** in this repo; install separately for MFE calculations and screening

### Clone with LFS

```bash
git lfs install
git clone https://github.com/GenerativeMolMachines/DNAzymesGEN.git
cd DNAzymesGEN
git lfs pull
```

---

## Environment setup

### GPU environment (TensorFlow / WGAN)

```bash
conda create -n dnazymes-gpu python=3.10 -y
conda activate dnazymes-gpu
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9 -y
pip install -r requirements-gpu.txt
```

Update `gpu_env.sh` if your conda environment path differs from the default.

### NUPACK (optional, for MFE filtering and screening)

Download and install [NUPACK 4](https://www.nupack.org/downloads) according to the vendor instructions, then ensure the Python bindings are importable:

```bash
python -c "import nupack; print(nupack.__version__)"
```

NUPACK is required for:

- MFE-filtered generation (`run_generation.sh`)
- Post-hoc MFE analysis (`compute_mfe_*.py`, `mfe_bootstrap_test.py`)
- Screening pipeline secondary-structure steps

Unfiltered generation (`run_generation_unfiltered.sh`) does **not** require NUPACK.

### Screening classifier dependencies

```bash
pip install scikit-learn lightgbm pandas numpy tqdm
```

---

## Reproducing the pipeline

All shell scripts assume the project root is `/root/dnazymes`. After cloning elsewhere, either clone to that path or edit the `PROJECT_ROOT` / absolute paths in the scripts.

### Step 1 — Pre-training

Train WGAN-GP generators on synthetic EDS or MFE datasets:

```bash
# EDS pretrain (→ checkpoints/eds/)
bash improved_wgan_training/run_pretrain.sh eds 0

# MFE pretrain (→ checkpoints/mfe/)
bash improved_wgan_training/run_pretrain.sh mfe 1
```

Or use the orchestrator:

```bash
python run_training.py --scenario eds      # EDS pretrain only
python run_training.py --scenario mfe      # MFE pretrain only
python run_training.py --scenario eds_ft   # EDS pretrain + Sequence Craft finetune
python run_training.py --scenario mfe_ft   # MFE pretrain + Sequence Craft finetune
```

**Published checkpoints** (already in `checkpoints/`):

| Model | Checkpoint | Notes |
|-------|------------|-------|
| EDS pretrain | `checkpoints/eds/model-1400` | Best available EDS snapshot |
| MFE pretrain | `checkpoints/mfe/model-3000` | Lowest train JSD on MFE data |
| EDS finetune | `checkpoints/eds_ft/model-999` | Finetuned on Sequence Craft (549 seq) |
| MFE finetune | `checkpoints/mfe_ft/model-999` | Finetuned on Sequence Craft |
| SC-only pretrain | `checkpoints/sequence_craft/model-700` | Trained from scratch on Sequence Craft |

### Step 2 — Fine-tuning

Fine-tune a pre-trained checkpoint on Sequence Craft:

```bash
bash run_finetune.sh 0 \
  checkpoints/eds_ft \
  checkpoints/eds/model-1400 \
  checkpoints/eds_ft/train.log
```

Parameters: `ITERS=1000`, `SAVE_INTERVAL=100`, learning rate `5e-5` (set inside `gan_language.py` finetune mode).

### Step 3 — Checkpoint evaluation

Compare checkpoints against Sequence Craft n-gram statistics (JSD):

```bash
bash run_evaluation.sh
# → checkpoints/evaluation_results.json
```

### Step 4 — Sequence generation

Generate 250k sequences per model (length filter only, no MFE):

```bash
bash run_generation_unfiltered.sh
```

Outputs land in `generated/*_nofilter/generated_sequences.csv`.

For MFE-filtered generation (requires NUPACK):

```bash
bash run_generation.sh
```

Single-model generation:

```bash
cd improved_wgan_training
python generate_sequences.py \
  --label my_run \
  --checkpoint ../checkpoints/eds_ft/model-999 \
  --num-sequences 250000 \
  --no-mfe-filter \
  --skip-mfe-calc
```

### Step 5 — Post-hoc MFE analysis

```bash
python compute_mfe_nofilter_comparison.py
python mfe_bootstrap_test.py
```

### Step 6 — Screening

Run the hierarchical screening pipeline on generated pools:

```bash
cd screening_pipeline
bash run_all_screening.sh
```

Or score sequences with the LightGBM classifier directly:

```bash
cd screening_pipeline/classifier_new_version
python run_model.py --input /path/to/sequences.csv
```

---

## Key scripts

| Script | Purpose |
|--------|---------|
| `improved_wgan_training/gan_language.py` | WGAN-GP pretrain / finetune for DNA sequences |
| `improved_wgan_training/generate_sequences.py` | Batch sequence generation from checkpoint |
| `improved_wgan_training/evaluate_checkpoints.py` | JSD-based checkpoint comparison |
| `FT/customed_tunning.py` | Alternative fine-tuning entry point |
| `FT/generate.py` | Alternative generation entry point |
| `run_training.py` | End-to-end training orchestrator |
| `screening_pipeline/run_screening.py` | Screening pipeline driver |

---

## Data sources

| File | Description |
|------|-------------|
| `Data/EDS/distrib_result.csv` | EDS-matched synthetic sequences |
| `Data/MFE/seq2 - seq2.csv` | MFE-bounded synthetic sequences |
| `Data/Sequence_Craft/SequenceCraft_dataset.csv` | 549 experimentally validated DNAzymes |
| `Data/Negatives/` | Negative control sequences |

---

## Citation

If you use this code or data, please cite the associated publication (TBD).

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

The DNA sequence WGAN code in `improved_wgan_training/` is adapted from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training) (image demo scripts are excluded from this repository).
