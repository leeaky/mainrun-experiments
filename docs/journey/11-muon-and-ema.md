# 11 — The Muon Hybrid Optimizer & Weight EMA

> **Status:** kept · **Effect:** val loss 1.19206 → **1.16566** (total **Δ −0.586** vs baseline) · **Date:** 2026-09-07
>
> ← [10-vocab-upsweep.md](10-vocab-upsweep.md) · [timeline](README.md) · [next](12-tuning-muon-recipe.md) →

## Question
Can we break through the ~1.192 AdamW optimization plateau by implementing the Muon (Momentum Orthogonalized by Newton-Schulz) optimizer for 2D hidden weights combined with an evaluation-time Exponential Moving Average (EMA) of weights?

## Reasoning
1. **Muon Optimizer:** Ported from Keller Jordan's *modded-nanogpt*. Standard AdamW scales updates using coordinate-wise second moments, which can be sensitive to outlier directions and ill-conditioned feature spectra. Muon computes standard momentum on 2D weight matrices, then applies 5 steps of a quintic Newton-Schulz iteration:
   $$X_{k+1} = a X_k + (b (X_k X_k^T) + c (X_k X_k^T)^2) X_k$$
   This projects the update onto the nearest orthogonal matrix, ensuring every singular direction is updated with uniform energy.
2. **Hybrid Partitioning:** Muon is designed strictly for 2D internal weight matrices. 1D parameters (LayerNorms, biases) and embeddings (token embedding tied to the LM head) must remain on AdamW.
3. **Eval-time Weight EMA:** In short, highly stochastic training runs, final weights oscillate around the loss valley. Maintaining a shadow EMA of weights $\theta_{\text{EMA}} = \alpha \theta_{\text{EMA}} + (1-\alpha) \theta$ and swapping them in *strictly during evaluation* yields smoother, better-generalizing predictions without altering training dynamics.

## Approach
1. Implemented [`muon.py`](../../mainrun/muon.py) with bit-exact fp32 Newton-Schulz iteration and `CombinedOptimizer` to coordinate Muon and AdamW parameter groups under a unified schedule.
2. Implemented [`ema.py`](../../mainrun/ema.py) with a clean context manager `with ema.swapped_in(model): evaluate()`.
3. Verified mathematically via unit tests: [`tests/test_muon.py`](../../tests/test_muon.py) and [`tests/test_ema.py`](../../tests/test_ema.py).
4. Swept `muon_lr` across `[0.01, 0.02, 0.03, 0.04]` and `ema_decay` across `[0.990, 0.995, 0.997, 0.998, 0.999]`.

## Results

### 1. Muon vs AdamW Head-to-Head (without EMA)
| Configuration | Val Loss | Δ vs AdamW Combo |
|---|:---:|:---:|
| AdamW Recipe (from Step 08) | 1.19206 | reference |
| Muon (`muon_lr=0.02`, AdamW `lr=1.2e-3`, no EMA) | **1.17362** | **−0.01844** |

Muon alone provided an immediate **−0.018** breakthrough!

### 2. EMA Decay Sweep
| EMA Decay | Val Loss | Notes |
|:---:|:---:|---|
| 0.000 (disabled) | 1.17362 | Reference raw weights |
| 0.990 | 1.17180 | Useful smoothing |
| 0.995 | 1.16870 | Strong gain |
| **0.997** | **1.16566** | **Optimal sweet spot (Δ −0.00796)** |
| 0.998 | 1.16610 | Beginning to lag |
| 0.999 | 1.17540 | Severe lag (retains initial weight bias) |

*(Note on history: Initial exploratory runs with an un-decoupled LayerNorm decay yielded 1.16539; after cleaning the parameter split so LayerNorm gains/biases had zero weight decay, the bit-exact reproduced result is **1.16566**, as preserved in `results/logs/recipe-muon-ema.log`).*

### Interaction with Scaled Residual Init
Interestingly, with Muon active, toggling `scaled_residual_init=true` slightly hurt performance (+0.002) because Newton-Schulz update orthogonalization inherently normalizes projection scale. We therefore disabled `scaled_residual_init` for the final Muon recipe.

## What we learned
- Muon provides a substantial, statistically robust improvement over AdamW for transformer hidden layers.
- EMA is an exceptionally cost-effective evaluation tool: zero training overhead, zero change to step dynamics, and delivers ~0.008 val loss reduction.
- Decays $>0.998$ are harmful on short ~1,000-step runs because the initial weights fail to wash out.

## Open threads
- Run Phase 11 / Session B to fine-tune the combined recipe and test architectural modernizations ([12-tuning-muon-recipe.md](12-tuning-muon-recipe.md)).

## Sources
- Code: [`mainrun/muon.py`](../../mainrun/muon.py), [`mainrun/ema.py`](../../mainrun/ema.py), [`mainrun/train.py`](../../mainrun/train.py)
- Tests: [`tests/test_muon.py`](../../tests/test_muon.py), [`tests/test_ema.py`](../../tests/test_ema.py)
- Reference log: [`results/logs/recipe-muon-ema.log`](../../results/logs/recipe-muon-ema.log)
- Source note: `docs/_source/notes/2026-09-07-phase10-session-a-results.md`
