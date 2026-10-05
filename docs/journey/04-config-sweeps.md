# 04 — Configuration Sweeps (Block, Batch, Dropout, Vocab, Capacity)

> **Status:** mixed findings (later re-analyzed in step 05) · **Effect:** identified capacity & dropout neutrality; flagged metric sensitivity · **Date:** 2026-09-07
>
> ← [03-scaled-residual-init.md](03-scaled-residual-init.md) · [timeline](README.md) · [next](05-eval-coverage-artifact.md) →

## Question
How do the hyperparameter toggles that require no code changes (`block_size`, `batch_size`, `dropout`, `vocab_size=8000`, model depth/width) perform under both SGD and AdamW?

## Reasoning
Before writing additional custom training code, we swept the existing configuration space to determine whether the baseline defaults were suboptimal.

## Approach
Executed structured sweeps via `experiments/drivers/phase4-config-sweeps.sh` evaluating:
1. `block_size`: 32, 64, 128 (baseline)
2. `batch_size`: 32, 64 (baseline), 128, 256
3. `dropout`: 0.0, 0.05, 0.1 (baseline), 0.2
4. `vocab_size`: 8,000 vs 16,000 (baseline)
5. Capacity: depth (4, 6, 8 layers) and width (`d_model` 256, 512, 768)

## Results

### 1. Dropout Sweep (AdamW + Scaled Residual Init)
| Dropout | Val Loss | Verdict |
|:---:|:---:|---|
| **0.00** | **1.20367** | Negligible gain (−0.0006) |
| 0.05 | 1.20392 | ~Flat |
| 0.10 (default) | 1.20428 | Reference |
| 0.20 | 1.21850 | Harmful (underfitting) |
*Decision:* Kept at 0.1 (or 0.0) — the model is in an underfitting regime.

### 2. Vocab Size Reduction (`vocab_size=8000`)
| Vocab Size | Val Loss | Verdict |
|:---:|:---:|---|
| 8,000 | 1.29822 | Worse by **+0.094** |
| 16,000 | 1.20428 | Reference |
*Finding:* Smaller vocabulary worsens character-level validation loss significantly.

### 3. Model Capacity (Depth & Width)
| Config (`n_layer`, `d_model`) | Params | Val Loss | Verdict |
|:---:|:---:|:---:|---|
| 4L / 512d | 18.9M | 1.23120 | Under-capacity |
| **6L / 512d (default)** | **25.2M** | **1.20428** | **Best balance** |
| 8L / 512d | 31.5M | 1.21105 | Slower convergence in 1000 steps |
| 6L / 768d | 49.3M | 1.20810 | Over-parameterized for 7 epochs |

### 4. Raw Batch Size & Block Size Sweeps
| Parameter | Setting | SGD Raw Val | AdamW Raw Val |
|---|:---:|:---:|:---:|
| `batch_size` | 32 | 1.75801 | — |
| `batch_size` | 64 (default) | 1.75193 | 1.20428 |
| `batch_size` | 128 | 1.67167 | 1.14971 |
| `batch_size` | 256 | — | 1.08912 |
| `block_size` | 32 | 1.72970 | 1.28564 |
| `block_size` | 64 | 1.76772 | 1.25748 |
| `block_size` | 128 (default) | 1.75193 | 1.20428 |

## What we learned
- Capacity: The default 6L / 512d architecture is well-tuned for a 1,000-step budget. Adding layers or width increased parameter count without improving validation loss within 7 epochs.
- Vocab: Dropping to 8,000 hurt performance dramatically.
- **The Red Flag:** The apparent massive improvement with `batch_size=128` (1.149) and `batch_size=256` (1.089) contradicted basic learning dynamics (fewer gradient steps should not dramatically beat more steps on identical data). This led directly to the critical diagnostic in Step 05.

## Open threads
- Investigate why larger `batch_size` and `block_size` appear to drop validation loss so aggressively ([05-eval-coverage-artifact.md](05-eval-coverage-artifact.md)).

## Sources
- Driver script: `experiments/drivers/phase4-config-sweeps.sh`
- Source note: `docs/_source/notes/2026-09-07-phase4-config-sweeps-results.md`
