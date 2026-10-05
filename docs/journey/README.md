# The Experimentation Journey

A step-by-step record of our learning journey optimizing a small GPT-2 model on Hacker News headlines within 7 epochs.

Each document merges the original hypothesis and reasoning with the exact empirical results and decisions taken.

## Chronological Steps

| Step | Title | Status | Effect (Δ Val Loss) | Core Takeaway |
|:---:|---|:---:|:---:|---|
| **00** | [**Training Experimentation Design**](00-design.md) | Baseline Roadmap | — | First-principles ranking of 14 training levers; optimizer predicted as highest ROI. |
| **01** | [**Baseline Reproduction & Config Plumbing**](01-baseline-reproduction.md) | Kept | **0.0 noise floor** | Verified bit-exact GPU determinism (`CUBLAS_WORKSPACE_CONFIG=:4096:8`); baseline locked at **1.75193**. |
| **02** | [**Optimizer: SGD → AdamW**](02-adamw.md) | Kept | **−0.504** | Single biggest breakthrough: 1.75193 → **1.24794** with peak `lr=1.2e-3` and warmup-cosine schedule. |
| **03** | [**Scaled Residual Initialization**](03-scaled-residual-init.md) | Kept (AdamW) | **−0.044** | $1/\sqrt{2 N_{\text{layer}}}$ residual projection scaling preserved signal flow, dropping loss to **1.20428**. |
| **04** | [**Configuration Sweeps**](04-config-sweeps.md) | Mixed Findings | — | Swept depth, width, dropout, vocab 8k, batch and block size; identified suspicious batch_size anomalies. |
| **05** | [**The Evaluation Metric Coverage Artifact**](05-eval-coverage-artifact.md) | **Critical Discovery** | Diagnosis | Uncovered `evaluate()` remainder-drop bug: larger batch/block sizes artificially dropped reported loss. Protected combo from contamination. |
| **06** | [**Tokenizer Scope & Leakage**](06-tokenizer-scope.md) | Kept Leakage | +0.07 to +0.12 | Measured impact of training tokenizer on val split (`train_and_val`); kept to match baseline rules. |
| **07** | [**Batch Shuffling**](07-batch-shuffling.md) | Kept | **−0.010** | Shuffling chunk order between epochs improved exploration under AdamW, reaching **1.19472**. |
| **08** | [**Engineering Toggles & Combo**](08-toggles-and-combo.md) | Kept | **−0.003** | SDPA bit-exact speedup; WSD schedule (60% stable) consolidated AdamW recipe at **1.19206**. |
| **09** | [**Streamlit Dashboard**](09-dashboard-abandoned.md) | Abandoned | 0.0 | Planned real-time monitoring UI abandoned to keep focus purely on ML training dynamics. |
| **10** | [**Vocab Upward Sweep & Precheck**](10-vocab-upsweep.md) | Kept 16k | Diagnostic tool | Precheck tool mapped compression vs gradient starvation; confirmed 16,000 is optimal in both directions. |
| **11** | [**The Muon Hybrid Optimizer & Weight EMA**](11-muon-and-ema.md) | **Core Breakthrough** | **−0.026** | Muon orthogonalization on 2D weights (−0.018) + eval-time weight EMA (−0.008) drove val loss to **1.16566**. |
| **12** | [**Tuning Muon & Architecture**](12-tuning-muon-recipe.md) | Complete | Validation | SwiGLU and QK-norm proved unhelpful; confirmed **1.16566** as the final optimal recipe. |

## The Overall Trajectory

```
Baseline (SGD, 6e-3)                      1.75193
  │
  ├── Step 02: AdamW (1.2e-3, cosine)      1.24794   (Δ -0.504)
  │
  ├── Step 03: Scaled Residual Init        1.20428   (Δ -0.044)
  │
  ├── Step 07: Batch Shuffling             1.19472   (Δ -0.010)
  │
  ├── Step 08: WSD Schedule                1.19206   (Δ -0.003)
  │
  ├── Step 11: Muon Optimizer              1.17362   (Δ -0.018)
  │
  └── Step 11: Eval-time Weight EMA        1.16566   (Δ -0.008)
```
**Total Validation Loss Reduction:** **−0.58627** (from 1.75193 down to 1.16566).
