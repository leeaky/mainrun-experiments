# mainrun-experiments

A systematic study in optimizing small LLM training dynamics under severe compute and data constraints (7 epochs, 100k Hacker News headlines).

> **Context & Disclaimer:**  
> This was undertaken as a personal learning exercise based on Maincode's LLM pre-training challenge. The goal was not recruitment submission, but building intuitive, empirical understanding of LLM training levers:
> 1. What training parameters actually move validation loss in a short training run?
> 2. How much does the choice of optimizer dictate sample efficiency?
> 3. How do you design a disciplined, zero-noise experimental pipeline?
>
> All training sweeps were run on a rented RunPod Secure Cloud NVIDIA RTX 4090 ($4 total compute spend).

---

## Headline Results

Validation loss minimized from **1.75193** down to **1.16566** (**Δ −0.58627** across 7 epochs, ~1,000 steps).

| Milestone | Configuration | Val Loss | Δ vs Baseline | Notes |
|---|---|:---:|:---:|---|
| **0. Shipped Baseline** | Vanilla SGD (`lr=6e-3`, `wd=0.0`), CosineAnnealing | 1.75193 | — | Slow convergence; un-momentumized |
| **1. AdamW Anchor** | AdamW (`lr=1.2e-3`, `wd=0.01`), Scaled Residual Init, Shuffled Batches, WSD Schedule | 1.19206 | **−0.55987** | Over 85% of total gains came from switching to adaptive optimizer |
| **2. Final Winning Recipe** | **Muon Hybrid** (`muon_lr=0.02`, `adamw_lr=1.2e-3`, `wd=0.01`), Shuffled Batches, WSD Schedule, **Weight EMA (0.997)** | **1.16566** | **−0.58627** | Quintic Newton-Schulz orthogonalization on 2D weights + eval-time EMA |

Every result is bit-exact reproducible on an RTX 4090 via `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms`.

---

## The Winning Recipe

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

To run this recipe:
```bash
MAINRUN_CONFIG=configs/recipe-muon-ema.json python mainrun/train.py
```

---

## Step-by-Step Learning Journey

The full exploration history is documented chronologically in [`docs/journey/`](docs/journey/README.md). Each document pairs the pre-experiment hypothesis and reasoning with the exact empirical results and decisions taken:

- [**00 — Training Experimentation Design**](docs/journey/00-design.md): First-principles ranking of 14 training levers.
- [**01 — Baseline Reproduction & Config Plumbing**](docs/journey/01-baseline-reproduction.md): Locking down the bit-exact 0.0 noise floor on GPU.
- [**02 — Optimizer: SGD → AdamW**](docs/journey/02-adamw.md): The single largest lever (1.75193 → 1.24794).
- [**03 — Scaled Residual Initialization**](docs/journey/03-scaled-residual-init.md): Stabilizing activation variance ($1/\sqrt{2 N_{\text{layer}}}$).
- [**04 — Configuration Sweeps**](docs/journey/04-config-sweeps.md): Model capacity, dropout, vocab 8k, batch and block sizes.
- [**05 — The Evaluation Metric Coverage Artifact**](docs/journey/05-eval-coverage-artifact.md): **Critical discovery** of the evaluation batch-drop bug.
- [**06 — Tokenizer Scope & Leakage**](docs/journey/06-tokenizer-scope.md): Measuring the impact of `train_and_val` tokenizer leakage (+0.07 to +0.12).
- [**07 — Batch Shuffling**](docs/journey/07-batch-shuffling.md): Shuffling sequence chunks across epochs (−0.010).
- [**08 — Engineering Toggles & AdamW Combo**](docs/journey/08-toggles-and-combo.md): SDPA, AMP, WSD schedule consolidating at 1.19206.
- [**09 — Streamlit Dashboard (Abandoned)**](docs/journey/09-dashboard-abandoned.md): Why real-time UI tooling was dropped.
- [**10 — Vocabulary Size Upward Sweep & Precheck**](docs/journey/10-vocab-upsweep.md): Precheck diagnostic mapping compression vs rare-token starvation.
- [**11 — The Muon Hybrid Optimizer & Weight EMA**](docs/journey/11-muon-and-ema.md): The core breakthrough driving val loss to 1.16566.
- [**12 — Tuning Muon & Architecture**](docs/journey/12-tuning-muon-recipe.md): Testing SwiGLU, QK-Norm, and confirming the optimal recipe.

Detailed reference guides:
- [**Summary & Scoreboard**](docs/SUMMARY.md): Consolidated overview of every experiment.
- [**Decisions Log**](docs/decisions-log.md): Exhaustive log of every architectural and engineering choice.
- [**Coverage Map**](docs/coverage-map.md): Exploration audit mapping tested components against the full pre-training pipeline.
- [**Evaluation Metric Artifact**](docs/evaluate-val-coverage-artifact.md): Technical breakdown of the `evaluate()` remainder-drop issue.

---

## Repository Structure

```
mainrun-experiments/
├── configs/                          # Experiment configurations
│   ├── baseline.json                 # Baseline defaults
│   ├── recipe-adamw.json             # Phase 8 AdamW anchor (1.19206)
│   └── recipe-muon-ema.json          # Final winning Muon+EMA recipe (1.16566)
├── mainrun/                          # Core training and model source code
│   ├── train.py                      # Main training script
│   ├── config.py                     # Hyperparameters dataclass and validation
│   ├── muon.py                       # Muon optimizer & Newton-Schulz iteration
│   ├── ema.py                        # Eval-time Exponential Moving Average
│   └── vocab_precheck.py             # Tokenizer compression & starvation diagnostic
├── tests/                            # Unit tests
│   ├── conftest.py                   # Test path configuration
│   ├── test_config.py                # Hyperparameter validation tests
│   ├── test_muon.py                  # Newton-Schulz & optimizer tests
│   ├── test_ema.py                   # Weight EMA tests
│   ├── test_arch.py                  # Architecture unit tests (SwiGLU, QK-Norm)
│   ├── test_sdpa_equiv.py            # SDPA numerical equivalence tests
│   └── test_vocab_precheck.py        # Tokenizer precheck tests
├── results/                          # Reference outputs and logs
│   ├── logs/                         # Structured JSON event traces
│   │   ├── baseline-1001.log         # Reference trace: baseline (1.75193)
│   │   ├── recipe-adamw.log          # Reference trace: AdamW recipe (1.19206)
│   │   └── recipe-muon-ema.log       # Reference trace: Muon+EMA recipe (1.16566)
│   └── vocab_precheck/               # Vocabulary sweep curves and charts
├── experiments/                      # Sweep orchestration records
│   ├── analyze.py                    # Metric extraction and log analysis tool
│   └── drivers/                      # Pod sweep runner scripts (historical records)
├── infra/                            # Infrastructure and container setup
│   └── runpod/                       # RunPod adapter Dockerfile and verify scripts
└── docs/                             # Documentation and learning journey
    ├── journey/                      # 13 chronological step-by-step experiment files
    ├── SUMMARY.md                    # Overall results summary
    ├── decisions-log.md              # Detailed rationale for every decision
    └── coverage-map.md               # Pipeline coverage audit
```

---

## Key Lessons Learned

1. **The Optimizer is the Decisive Lever:** In short training runs (~1,000 steps), the choice of optimizer overwhelmingly dominates all other parameters. Switching from vanilla SGD to AdamW yielded Δ −0.504; switching to Muon yielded an additional Δ −0.018.
2. **Weight EMA is an Invaluable Eval Transform:** Applying an exponential moving average of model weights strictly at evaluation time yielded a free Δ −0.008 reduction without altering training dynamics or requiring extra gradient steps.
3. **Always Audit Your Evaluation Metric:** The apparent massive improvement with large batch sizes (`batch_size=256`) was entirely an artifact of `evaluate()` dropping trailing tokens while dividing by a constant total length. Verifying evaluation math prevented a deeply flawed configuration choice.
4. **Architectural Complexity Often Costs Speed:** In a tight 1,000-step budget, complex modern architectures like SwiGLU and QK-Norm increased parameter count and slowed convergence compared to well-initialized standard GELU MLPs.
