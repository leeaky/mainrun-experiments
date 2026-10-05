# 07 — Batch Shuffling

> **Status:** kept · **Effect:** val loss 1.20428 → 1.19472 (**Δ −0.0096**) · **Date:** 2026-09-07
>
> ← [06-tokenizer-scope.md](06-tokenizer-scope.md) · [timeline](README.md) · [next](08-toggles-and-combo.md) →

## Question
Does shuffling the sequence chunks across batches each epoch improve training convergence compared to the baseline's sequential pointer walk?

## Reasoning
The baseline `train.py` streamed tokens sequentially using a simple index pointer:
`get_batch(train_ids, ptr, block_size, batch_size, device)`
Once `ptr` reached the end of the dataset, it wrapped back to 0. Consequently, every epoch saw the exact same batch compositions in the exact same sequence.
Shuffling chunk order each epoch introduces stochasticity, disrupts spurious inter-batch correlations, and ensures gradient updates explore different directions across epochs.

## Approach
Implemented chunk-based shuffling in `train.py`:
```python
if args.shuffle_batches:
    n_chunks = len(train_ids) // (args.block_size + 1)
    train_chunks = train_ids[:n_chunks * (args.block_size + 1)].view(n_chunks, args.block_size + 1)
    batches = n_chunks // args.batch_size
    ...
    # In epoch loop:
    order = list(range(n_chunks))
    random.Random(args.seed + epoch).shuffle(order)
```

## Results

| Optimizer & Recipe | Sequential Pointer | Shuffled Batches | Δ |
|---|:---:|:---:|:---:|
| Baseline (SGD, `lr=6e-3`) | 1.75193 | 1.75210 | +0.00017 (neutral) |
| AdamW + Scaled Residual Init | 1.20428 | **1.19472** | **−0.00956** |

## What we learned
- Shuffling had zero effect under SGD (which was already making tiny, noisy updates), but yielded a clean, reproducible **−0.01** drop under AdamW.
- Breaking fixed batch trajectories helps adaptive momentum optimizers escape local saddles.
- Shuffling chunks rather than raw individual tokens preserved local text coherence within each 128-token sequence.

## Open threads
- Test execution toggles (SDPA, AMP, Grad Clip) and schedule alternatives (WSD) to consolidate the AdamW recipe ([08-toggles-and-combo.md](08-toggles-and-combo.md)).

## Sources
- Code: [`mainrun/train.py`](../../mainrun/train.py)
- Source note: `docs/_source/notes/2026-09-07-exp-4.6-batch-shuffling.md`
