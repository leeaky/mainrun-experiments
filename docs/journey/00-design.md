# 00 — Training Experimentation Design

> **Status:** baseline roadmap · **Effect:** design roadmap for 14 levers · **Date:** 2026-09-05
>
> ← start · [timeline](README.md) · [next](01-baseline-reproduction.md) →

## Question
How should we systematically explore improvements to the `mainrun` GPT-2 training run within the 7-epoch constraint, and which levers are most likely to yield significant reductions in validation loss?

## Reasoning
This was a personal learning exercise on Maincode's standardized assessment harness: training a small GPT-2 model on a Hacker News headline dataset to minimize validation loss within 7 epochs.

Our approach was rooted in an intuitive first-principles audit:
1. Deconstruct the training pipeline as dictated by the original repository.
2. Compare the baseline configuration to standard modern LLM pre-training practices.
3. Identify core bottlenecks:
   - **Short run duration (7 epochs / ~1,000 steps):** Learning must be rapid and sample-efficient.
   - **Suboptimal default optimizer:** The baseline used vanilla SGD without momentum (`lr=6e-3`, `weight_decay=0.0`), which converges notoriously slowly compared to adaptive or second-order optimizers.
4. Rank training levers by expected return on investment (ROI) to guide structured experimentation.

## Approach
We categorized all adjustable components into prioritized experimental areas, adhering to strict rules (fixed seed `1337`, 7 epochs, 10% validation split, untouched dataset, and frozen `evaluate()` function):

| Rank | Lever / Component | Area | Hypothesis / Expected Impact |
|:---:|---|---|---|
| 1 | **Optimizer (SGD → AdamW / Muon)** | Optimization | **Highest ROI.** SGD struggles in short horizons. Adaptive or spectral updates will drive the steepest convergence. |
| 2 | **Learning Rate & Schedule** | Optimization | Warmup + Cosine / WSD decay to ensure stable warmup and aggressive late-stage convergence. |
| 3 | **Initialization (Scaled Residual Init)** | Architecture | Scale residual projection weights by $1/\sqrt{2 N_{\text{layer}}}$ to stabilize variance growth across residual layers. |
| 4 | **Weight Decay & Regularization** | Optimization | Decoupled weight decay on 2D weight matrices; evaluate dropout impact under tight data budget. |
| 5 | **Sequence & Batch Layout** | Data Pipeline | Examine `block_size` vs headline length distributions, batch size tradeoffs, and batch chunk shuffling. |
| 6 | **Vocabulary Size & Tokenizer Scope** | Data Pipeline | Evaluate compression vs token count tradeoffs, and inspect the `train_and_val` tokenizer leakage. |
| 7 | **Architecture Refinements** | Architecture | Test modern alternatives: SwiGLU MLPs, QK-Norm, Rotary Position Embeddings (RoPE). |
| 8 | **Execution & Eval Enhancements** | Engineering | SDPA / FlashAttention for speed; eval-time Exponential Moving Average (EMA) of weights. |

Experimental discipline was locked down from day one:
- One variable changed at a time (OVAT).
- Deterministic execution (`CUBLAS_WORKSPACE_CONFIG=:4096:8` + `torch.use_deterministic_algorithms(True, warn_only=True)`) to eliminate run-to-run noise on GPU.
- Every experiment documented with explicit hypothesis, parameters, and delta against baseline.

## Results
The design provided the structured framework for ~190 pod training runs. The baseline starting point was:
- **Optimizer:** SGD (`lr=6e-3`, `weight_decay=0.0`)
- **Schedule:** CosineAnnealingLR
- **Architecture:** 6-layer GPT-2 (`d_model=512`, `n_head=8`, `block_size=128`, `vocab_size=16000`, learned absolute pos emb)
- **Baseline Val Loss:** **1.75193** (over 1,001 steps on the current dataset)

## What we learned
Pre-ranking components prevented haphazard parameter tinkering. It correctly predicted that optimization (first AdamW, later Muon + EMA) would deliver >90% of all net validation gains, while architecture changes (SwiGLU, QK-norm) and regularizers yielded negligible or negative returns on this short horizon.

## Open threads
- Establish bit-exact baseline reproducibility and build non-invasive configuration plumbing ([01-baseline-reproduction.md](01-baseline-reproduction.md)).
- Test the highest-ranked lever: replacing SGD with AdamW ([02-adamw.md](02-adamw.md)).

## Sources
- Source design doc: `docs/_source/specs/2026-09-05-training-experimentation-design.md`
- Context: `docs/_source/leeaky_README.md`
- Decisions log: [../decisions-log.md](../decisions-log.md)
