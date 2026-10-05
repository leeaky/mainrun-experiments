# 08 — Engineering Toggles & AdamW Recipe Consolidation

> **Status:** kept · **Effect:** consolidated Phase 1–8 AdamW recipe to **1.19206** (Δ −0.560 vs baseline) · **Date:** 2026-09-07
>
> ← [07-batch-shuffling.md](07-batch-shuffling.md) · [timeline](README.md) · [next](09-dashboard-abandoned.md) →

## Question
Can we consolidate all verified gains into a unified AdamW recipe, and do speed/stability toggles (SDPA, AMP, Gradient Clipping, WSD schedule) improve results?

## Reasoning
Phases 1 through 6 demonstrated strong individual improvements. In this phase, we tested interaction effects, explored low-overhead engineering features, and finalized an anchor checkpoint before investigating new optimizers.

## Approach
Evaluated the following toggles on top of `{optimizer=adamw, lr=1.2e-3, wd=0.01, scaled_residual_init=true, shuffle_batches=true}`:
1. **Scaled Dot Product Attention (`attn_impl="sdpa"`):** FlashAttention / cuDNN kernel vs manual matrix multiplication.
2. **Automatic Mixed Precision (`use_amp=true`):** PyTorch `autocast(dtype=torch.bfloat16)`.
3. **Gradient Clip Norm:** 0.5, 1.0 (default), 2.0.
4. **LR Schedule:** Cosine vs Warmup-Stable-Decay (`lr_schedule="wsd"`, 60% stable, linear tail).

## Results

### Toggles Evaluation
| Toggle | Val Loss | Effect / Decision |
|---|:---:|---|
| Baseline Combo (Cosine, Manual Attn, fp32) | 1.19472 | Reference |
| + `attn_impl="sdpa"` | 1.19472 | **Bit-exact equivalence** on evaluation, 15% faster training |
| + `use_amp=true` (bfloat16) | 1.19476 | Minor numerical difference (+0.00004); left off to protect bit-exact determinism |
| + Gradient Clip 0.5 | 1.19510 | Slightly restrictive |
| + Gradient Clip 2.0 | 1.19480 | ~Flat |
| + `lr_schedule="wsd"` (stable=0.6, linear decay) | **1.19206** | **Improvement of −0.00266** |

### Consolidated Phase 8 Recipe
```json
{
  "optimizer": "adamw",
  "lr": 0.0012,
  "weight_decay": 0.01,
  "scaled_residual_init": true,
  "shuffle_batches": true,
  "lr_schedule": "wsd"
}
```
**Final Validation Loss:** **1.19206** (Baseline: 1.75193, Total **Δ −0.55987**).
Log trace preserved in [`results/logs/recipe-adamw.log`](../../results/logs/recipe-adamw.log).

## What we learned
- WSD (Warmup-Stable-Decay) outperforms standard cosine decay because holding maximum learning rate across 60% of steps allows more exploration before the final annealing phase.
- SDPA is bit-exact with manual attention in eval mode, proving the manual attention implementation was mathematically sound.
- With AdamW thoroughly tuned, the model reached a performance plateau around ~1.192. Further major breakthroughs required fundamentally rethinking the optimization algorithm.

## Open threads
- Document the abandoned Streamlit monitoring dashboard ([09-dashboard-abandoned.md](09-dashboard-abandoned.md)).
- Investigate vocabulary size scaling upwards ([10-vocab-upsweep.md](10-vocab-upsweep.md)).
- Explore the Muon optimizer and EMA ([11-muon-and-ema.md](11-muon-and-ema.md)).

## Sources
- Driver script: `experiments/drivers/phase78-toggles.sh`
- Reference log: [`results/logs/recipe-adamw.log`](../../results/logs/recipe-adamw.log)
- Source note: `docs/_source/notes/2026-09-07-phase78-toggles-and-combo.md`
