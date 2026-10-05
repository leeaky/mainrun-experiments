# 12 — Tuning the Muon Recipe & Architectural Exploration

> **Status:** kept Phase 10 headline; ruled out remaining levers · **Effect:** confirmed 1.16566 as final optimal recipe · **Date:** 2026-09-07
>
> ← [11-muon-and-ema.md](11-muon-and-ema.md) · [timeline](README.md) · [decisions](../decisions-log.md) →

## Question
Can we further improve the 1.16566 Muon + EMA benchmark by tuning the AdamW sub-optimizer parameters, adjusting WSD decay shapes, or introducing modern architectural components (SwiGLU, QK-Norm)?

## Reasoning
With Muon confirmed as the primary optimizer, we conducted an exhaustive exploration of secondary parameters:
1. AdamW parameters for the embedding/norm groups (`lr`, `betas`, `weight_decay`).
2. Architecture upgrades from modern LLM literature:
   - **SwiGLU MLP:** Replaces GELU MLP with parameter-matched gated linear units ($h \approx \frac{8}{3} d_{\text{model}}$).
   - **QK-Norm:** RMSNorm applied to query and key vectors before dot-product attention to stabilize logit growth.
3. Schedule variants: 1-sqrt inverse decay tail vs linear decay.

## Approach
Executed structured sweeps via `experiments/drivers/phase11-session-b.py` and `small-batch.py`:
- AdamW sub-group tuning: `lr` $\in [0.0006, 0.0009, 0.0012, 0.0018]$, `wd` $\in [0.0, 0.01, 0.02, 0.04]$, `beta2` $\in [0.95, 0.98, 0.99]$.
- SwiGLU MLP vs GELU MLP.
- QK-Norm enabled vs disabled.
- WSD decay shape: `linear` vs `invsqrt`.
- Smaller batch sizes (16, 24, 32) under coverage adjustment.

## Results

### 1. Architectural Additions (Under Muon + WSD)
| Variant | No-EMA Val Loss | With EMA (0.997) | Verdict |
|---|:---:|:---:|---|
| **Baseline GELU MLP (default)** | **1.17184** | **1.16566** | **Kept** |
| SwiGLU MLP (param-matched) | 1.17482 | 1.16812 | Slightly worse (+0.0025) |
| QK-Norm (`q_norm`, `k_norm`) | 1.17640 | 1.16950 | Worse (+0.0038) |
| SwiGLU + QK-Norm | 1.17910 | 1.17205 | Cumulative degradation |

### 2. AdamW Sub-Group Tuning
- Best raw (no-EMA) settings: `adamw_lr=0.0009`, `weight_decay=0.02`, `beta2=0.99` reached **1.16854** without EMA (vs 1.17184 default).
- However, when combined with optimal EMA (`decay=0.997`), all tuned configurations converged to **~1.1656–1.1658**. The EMA smoothing effectively subsumed the fine-grained AdamW learning rate differences.

### 3. WSD Schedule Tail Shape
| Decay Shape | Final Val Loss | Verdict |
|---|:---:|---|
| Linear decay (default) | **1.16566** | Optimal |
| Inverse-sqrt (`invsqrt`) | 1.16720 | Drops too quickly, stalls near floor |

### 4. Small-Batch Investigation (Coverage Adjusted)
Testing `batch_size` 16, 24, 32 on the Muon recipe confirmed that once coverage differences are normalized, `batch_size=64` remains the optimal compromise between gradient quality and step count.

## What we learned
- **SwiGLU and QK-Norm did not help:** On a 1,000-step training run of a 25M parameter model, the added parameter overhead of 3 gated matrices and additional normalization layers slowed convergence. Standard GELU MLP and unnormalized dot products were superior.
- **Recipe Convergence:** The Phase 10 headline recipe is robust and optimal. Further tuning of secondary parameters yields diminishing returns (<0.0003).

## Final Winning Recipe
```json
{
  "optimizer": "muon",
  "muon_lr": 0.02,
  "lr": 0.0012,
  "weight_decay": 0.01,
  "shuffle_batches": true,
  "lr_schedule": "wsd",
  "ema_decay": 0.997
}
```
- **Validation Loss:** **1.16566** (Baseline: **1.75193**, Net Improvement: **Δ −0.58627**).
- Bit-exact log trace: [`results/logs/recipe-muon-ema.log`](../../results/logs/recipe-muon-ema.log).

## Sources
- Code: [`mainrun/config.py`](../../mainrun/config.py), [`mainrun/train.py`](../../mainrun/train.py), [`mainrun/muon.py`](../../mainrun/muon.py)
- Driver scripts: `experiments/drivers/phase11-session-b.py`, `small-batch.py`
- Source note: `docs/_source/notes/2026-09-07-phase11-session-b-results.md`
- Comprehensive overview: [../SUMMARY.md](../SUMMARY.md), [../coverage-map.md](../coverage-map.md)
