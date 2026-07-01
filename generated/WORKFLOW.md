# DNA GAN: Checkpoints, Fine-Tuning, and Generation

This document describes the workflow in the DNAzymesGEN project: which checkpoints were used, why they were selected, and how the pipeline runs from pretrain through generation.

---

## Pipeline overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Pretrain       │     │  Finetune        │     │  Generation         │
│  eds / mfe      │ ──► │  sequence_craft  │ ──► │  250k seq / model   │
│  (large CSV)    │     │  (549 seq)       │     │  no MFE filter      │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
```

| Stage | Script | Model | Data |
|-------|--------|-------|------|
| Pretrain EDS | `improved_wgan_training/run_pretrain.sh eds` | WGAN-GP | `Data/EDS/distrib_result.csv` |
| Pretrain MFE | `improved_wgan_training/run_pretrain.sh mfe` | WGAN-GP | `Data/MFE/seq2 - seq2.csv` |
| Finetune | `scripts/training/run_finetune.sh` | WGAN-GP, lr=5e-5 | `Data/Sequence_Craft/SequenceCraft_dataset.csv` |
| Pretrain SC only | `improved_wgan_training/gan_language.py` | WGAN-GP, lr=1e-4 | `Data/Sequence_Craft/SequenceCraft_dataset.csv` |
| Generation | `scripts/generation/run_generation_unfiltered.sh` | same generator | — |

Quality on Sequence Craft: `scripts/evaluation/run_evaluation.sh` → `checkpoints/evaluation_results.json`  
Metric: sum of JSD over n-grams (1–4) between generated and reference sequences — **lower is closer to Sequence Craft**.

---

## 1. Pretrain: starting checkpoints for finetune

### EDS — `checkpoints/eds/model-1400`

| Parameter | Value |
|-----------|-------|
| Training iterations | 3600 (`--iters 3600`, `--save-interval 200`) |
| **Used for finetune** | **model-1400** (iter 1400) |
| Restored in finetune | `Restored checkpoint: .../eds/model-1400` (`eds_ft/train.log`) |

**Why model-1400:**

- The only pretrain checkpoint saved under `checkpoints/eds/` is `model-1400` (the `checkpoint` file points to it alone). Finetune uses the latest available checkpoint by default (`scripts/training/run_training.py` → `find_latest_checkpoint`).
- Training log (`eds/train.log`): total JSD (js1+…+js4) at iter 1399 ≈ **0.0235** — model has converged, n-gram metrics are stable.
- Absolute minimum JSD in the log is at iter **2099** (≈ 0.0176), but that checkpoint was **not saved** to disk; by iter 3599 JSD rises again (≈ 0.029), suggesting possible overfitting on the EDS dataset.
- **Conclusion:** finetune uses the only available pretrain snapshot — model-1400 — before clear metric degradation at later iterations.

### MFE — `checkpoints/mfe/model-3000`

| Parameter | Value |
|-----------|-------|
| Training iterations | 3600 |
| **Used for finetune** | **model-3000** (iter 3000) |
| Restored in finetune | `Restored checkpoint: .../mfe/model-3000` (`mfe_ft/train.log`) |

**Why model-3000:**

- Only saved pretrain checkpoint in `checkpoints/mfe/`.
- Training log (`mfe/train.log`): **best** total JSD on the pretrain dataset at iter **2999** (≈ **0.0095**, saved as model-3000).
- At later iterations (3299+) JSD worsens — model-3000 matches the train-metric optimum.

---

## 2. Finetune on Sequence_Craft

| Parameter | Value |
|-----------|-------|
| Dataset | Sequence_Craft, 549 sequences |
| Iterations | 1000 (`ITERS=1000`) |
| Save interval | every 100 iterations |
| Learning rate | 5e-5 (2× lower than pretrain) |
| EDS finetune start | `eds/model-1400` → `checkpoints/eds_ft/` |
| MFE finetune start | `mfe/model-3000` → `checkpoints/mfe_ft/` |

During finetune, the log records the same JSD (js1–js4) against the Sequence Craft n-gram model — the main convergence signal on the target domain.

---

## 3. Finetune checkpoints used for generation

### EDS finetune — `checkpoints/eds_ft/model-999`

| Criterion | model-999 |
|-----------|-----------|
| Train JSD (sum js1–js4, iter 999) | **0.0326** — best among saved finetune iterations |
| Eval on Sequence_Craft (`evaluation_results.json`) | js_sum_mean = **0.0375** |
| vs pretrain (model-1400) | pretrain js_sum = 0.0434 → **finetune improved** closeness to Sequence_Craft |

**Selection:** the final checkpoint (iter 999) gives **minimum JSD in the finetune log** and the **best offline evaluation** among EDS models. Only `model-999` is saved on disk.

### MFE finetune — `checkpoints/mfe_ft/model-999`

| Criterion | model-800 | model-999 |
|-----------|-----------|-----------|
| Train JSD (sum, finetune log) | **0.0399** (iter 799) | 0.0437 (iter 999) |
| Eval js_sum_mean (Sequence_Craft) | 0.0402 | **0.0387** |

**Why model-999 for generation:**

- By **train log**, iter ~800 is better, but intermediate checkpoints (model-800) are **not saved** — only `model-999` exists in `checkpoints/mfe_ft/`.
- By **offline evaluation** (`scripts/evaluation/run_evaluation.sh`, 5 sampling rounds) **model-999 beats model-800** (0.0387 vs 0.0402).
- MFE pretrain alone poorly matches Sequence Craft (js_sum ≈ 0.298); finetune model-999 reduces this to ≈ 0.039.

**Conclusion:** **model-999** was used for generation — the only available finetune artifact and best on Sequence Craft eval.

> Note: `scripts/evaluation/run_evaluation.sh` labels model-800 as "best" by train metric and model-999 as "final"; production generation uses **final (999)** based on eval and disk availability.

---

## 3b. Pretrain on Sequence_Craft only (no EDS/MFE)

Alternative scenario: train WGAN **from scratch** on 549 experimental DNAzymes only, without synthetic pretrain or finetune.

| Parameter | Value |
|-----------|-------|
| Script | `improved_wgan_training/gan_language.py --dataset sequence_craft --mode pretrain` |
| Dataset | Sequence_Craft, 549 sequences |
| Iterations | 1000 planned, **stopped at 700** |
| Save interval | every 100 iterations |
| Learning rate | 1e-4 |
| GPU | 1 (Tesla P40) |
| Log | `checkpoints/sequence_craft/train.log` |

### Checkpoint selection — `checkpoints/sequence_craft/model-700`

| Criterion | model-400 | model-500 | model-700 |
|-----------|-----------|-----------|-----------|
| js1 | 0.0004 | 0.0032 | **0.0001** |
| js2 | 0.0039 | 0.0079 | 0.0041 |
| js3 | 0.0092 | 0.0142 | **0.0087** |
| js4 | 0.0229 | 0.0283 | **0.0206** |
| **Σ js1–4** | 0.0364 | 0.0535 | **0.0335** |

**Why model-700:**

- Best total JSD (js1+…+js4 = **0.0335**) among all saved checkpoints — comparable to EDS+FT finetune (≈ 0.033).
- At iter 500 metrics regressed (Σ = 0.0535), then improved again by iter 699.
- Training stopped manually at iter 700; intermediate checkpoints (0–600) were removed.

### Generation

| Model | Checkpoint | Output | Filters |
|-------|------------|--------|---------|
| sequence_craft_nofilter | `sequence_craft/model-700` | `generated/sequence_craft_nofilter/` | 20 ≤ len < 100 |

Volume: **250,000** sequences, no MFE filter (same as other nofilter sets).

---

## 4. Generation (current configuration)

Uses **`scripts/generation/run_generation_unfiltered.sh`** — no MFE filtering.

| Model | Checkpoint | Output | Filters |
|-------|------------|--------|---------|
| eds_pretrain_nofilter | `eds/model-1400` | `generated/eds_pretrain_nofilter/` | 20 ≤ len < 100 |
| mfe_pretrain_nofilter | `mfe/model-3000` | `generated/mfe_pretrain_nofilter/` | 20 ≤ len < 100 |
| eds_ft_nofilter | `eds_ft/model-999` | `generated/eds_ft_nofilter/` | 20 ≤ len < 100 |
| mfe_ft_nofilter | `mfe_ft/model-999` | `generated/mfe_ft_nofilter/` | 20 ≤ len < 100 |
| sequence_craft_nofilter | `sequence_craft/model-700` | `generated/sequence_craft_nofilter/` | 20 ≤ len < 100 |

Volume: **250,000** sequences per model. MFE is not computed during generation (for speed); post-hoc MFE via `scripts/analysis/compute_mfe_nofilter_comparison.py`, cache in `generated/mfe_cache/`.

Earlier runs used MFE filtering (MFE ≤ −10) via `scripts/generation/run_generation.sh`; those sets were removed in favor of nofilter outputs.

---

## 5. MFE summary (post-hoc, NUPACK)

| Dataset | Mean MFE | Median | Fraction MFE ≤ −10 |
|---------|----------|--------|---------------------|
| Sequence_Craft | −3.86 | −3.58 | 0.5% |
| Negatives | −3.36 | −3.08 | 0.6% |
| eds_pretrain_nofilter | −4.15 | −3.86 | 1.9% |
| mfe_pretrain_nofilter | −8.21 | −7.98 | 25.8% |
| eds_ft_nofilter | −3.31 | −2.98 | 1.1% |
| mfe_ft_nofilter | −2.95 | −2.66 | 0.4% |

Finetune shifts MFE and length distributions toward Sequence_Craft; MFE pretrain remains more "structured" than the reference.

---

## 6. Key files

| File | Purpose |
|------|---------|
| `improved_wgan_training/gan_language.py` | WGAN train / finetune |
| `improved_wgan_training/run_pretrain.sh` | pretrain eds or mfe |
| `scripts/training/run_finetune.sh` | finetune on sequence_craft |
| `scripts/evaluation/run_evaluation.sh` | JSD checkpoint evaluation |
| `scripts/generation/run_generation_unfiltered.sh` | generation without MFE |
| `checkpoints/evaluation_results.json` | numeric checkpoint comparison |
| `generated/mfe_nofilter_vs_references_*.png` | MFE plots |
| `scripts/analysis/mfe_bootstrap_test.py` | bootstrap MFE test: Sequence_Craft vs Negatives |

---

## 7. Timeline (approximate)

1. **Pretrain** EDS (→ model-1400) and MFE (→ model-3000) on large CSVs.
2. **Evaluation** of pretrain + finetune candidates on Sequence Craft (`evaluation_results.json`).
3. **Finetune** eds_ft and mfe_ft (1000 iter, starting from pretrain checkpoints).
4. **MFE-filtered generation** — low accept rate on ft models; aborted.
5. **Unfiltered generation** — 4 × 250k seq, length filter only.
6. **Post-hoc MFE** and comparison with Sequence Craft / Negatives.
7. **SC-only pretrain** — WGAN from scratch on Sequence Craft (1000 iter planned, best model-700, Σ JSD = 0.0335).
8. **sequence_craft_nofilter generation** — 250k seq from `sequence_craft/model-700`.

---

*Last updated: 2026-06-23. JSD and MFE metrics match artifacts in `checkpoints/` and `generated/`.*
