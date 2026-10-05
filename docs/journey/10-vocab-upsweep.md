# 10 — Vocabulary Size Upward Sweep & Precheck Diagnostic

> **Status:** kept baseline vocab_size=16000 · **Effect:** confirmed 16,000 is optimal in both directions · **Date:** 2026-09-07
>
> ← [09-dashboard-abandoned.md](09-dashboard-abandoned.md) · [timeline](README.md) · [next](11-muon-and-ema.md) →

## Question
Step 04 demonstrated that dropping `vocab_size` to 8,000 degraded validation loss (+0.094). Does increasing `vocab_size` above 16,000 (e.g. to 20k, 24k, 28k, 32k) improve validation loss, and can we predict this without full training runs?

## Reasoning
Larger vocabularies compress text more efficiently (higher characters per token), shortening sequence lengths and allowing the model to attend to wider contexts in fewer tokens. However, larger vocabularies also:
1. Increase parameter count in the embedding and LM head ($V \times d_{\text{model}}$).
2. Result in rare tokens that are updated only a handful of times across 7 epochs (gradient starvation).
We built a standalone, CPU-runnable diagnostic tool ([`vocab_precheck.py`](../../mainrun/vocab_precheck.py)) to map the compression curve and count undertrained tokens *before* launching GPU training runs.

## Approach
1. **Tokenizer Precheck Tool:** [`mainrun/vocab_precheck.py`](../../mainrun/vocab_precheck.py) measures:
   - Compression efficiency (val characters per token).
   - Diminishing returns knee point via Kneedle algorithm.
   - `frac_undertrained`: Fraction of vocabulary tokens appearing fewer than 5 times in the training corpus.
2. **GPU Validation Sweep:** Ran full training runs using `experiments/drivers/vocab-upsweep.sh` across `[16000, 20000, 24000, 28000]` on the AdamW recipe.

## Results

### Standalone Precheck Metrics
| Requested Vocab | Actual Vocab | Val Chars / Token | Diminishing Return Δ | Tokens with <5 occ. | % Undertrained |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 4,000 | 4,000 | 3.65 | — | 0 | 0.0% |
| 8,000 | 8,000 | 4.04 | +0.39 | 0 | 0.0% |
| **16,000** | **16,000** | **4.34** | **+0.30** | **74** | **0.5%** |
| 20,000 | 20,000 | 4.41 | +0.07 | 342 | 1.7% |
| 24,000 | 24,000 | 4.46 | +0.05 | 988 | 4.1% |
| 28,000 | 28,000 | 4.49 | +0.03 | 2,120 | 7.6% |
| 32,000 | 32,000 | 4.52 | +0.03 | 3,842 | 12.0% |

The precheck clearly demonstrated that compression gains flatten severely past 16,000, while the percentage of undertrained tokens climbs rapidly.

### Full Training Run Validation Loss
| Vocab Size | Headline Val Loss | Δ vs 16k |
|:---:|:---:|:---:|
| **16,000** | **1.19206** | **Reference** |
| 20,000 | 1.20015 | +0.00809 (worse) |
| 24,000 | 1.20842 | +0.01636 (worse) |
| 28,000 | 1.21588 | +0.02382 (worse) |

## What we learned
- The baseline choice of `vocab_size=16000` is a sweet spot:
  - Sweeping downwards (8k) degraded loss by +0.094 due to poor compression.
  - Sweeping upwards (20k–28k) monotonically degraded loss because rare tokens suffered from gradient starvation within the 7-epoch limit.
- The standalone precheck diagnostic accurately predicted this behavior in seconds on CPU without wasting GPU hours.

## Open threads
- Proceed to the core innovation: the Muon optimizer and evaluation-time weight EMA ([11-muon-and-ema.md](11-muon-and-ema.md)).

## Sources
- Code: [`mainrun/vocab_precheck.py`](../../mainrun/vocab_precheck.py), [`tests/test_vocab_precheck.py`](../../tests/test_vocab_precheck.py)
- Driver script: `experiments/drivers/vocab-upsweep.sh`
- Plots & Data: [`results/vocab_precheck/`](../../results/vocab_precheck/)
- Source note: `docs/_source/notes/2026-09-07-exp-4.8-vocab-upsweep.md`
