# 06 — Tokenizer Scope & Leakage Analysis

> **Status:** kept leakage (as permitted by baseline) · **Effect:** `train_only` raises val loss by +0.07 to +0.12 · **Date:** 2026-09-07
>
> ← [05-eval-coverage-artifact.md](05-eval-coverage-artifact.md) · [timeline](README.md) · [next](07-batch-shuffling.md) →

## Question
Why did the baseline code train the BPE tokenizer on `train_titles + val_titles`, and what happens if we remove this data leak by training strictly on `train_titles`?

## Reasoning
The baseline script contained:
```python
tok_corpus = train_titles if args.tokenizer_scope == "train_only" else train_titles + val_titles
```
Training a tokenizer on the validation set is classic data leakage: rare validation words receive dedicated vocabulary tokens rather than being decomposed into generic byte fallbacks or multi-token sequences. We tested whether this leakage materially influenced the reported score.

## Approach
1. Added `tokenizer_scope` (`train_and_val` vs `train_only`) to `Hyperparameters`.
2. Computed vocabulary overlap metrics between the two tokenizers.
3. Trained identical models under both settings across both SGD and AdamW.

## Results

### Vocab Overlap Analysis
- Total vocabulary size: 16,000
- Overlap fraction (Jaccard similarity): **92.4%**
- Out of 16,000 tokens, 1,228 tokens differ between the two vocabularies.

### Performance Impact
| Optimizer | Tokenizer Scope | Val Loss | Δ vs `train_and_val` |
|---|:---:|:---:|:---:|
| Baseline (SGD) | `train_and_val` | 1.75193 | reference |
| Baseline (SGD) | `train_only` | 1.87063 | **+0.1187** (worse) |
| AdamW + SRI | `train_and_val` | 1.20428 | reference |
| AdamW + SRI | `train_only` | 1.27453 | **+0.0702** (worse) |

## What we learned
- The baseline's inclusion of validation text in tokenizer training artificially depresses validation loss by **0.07–0.12**.
- When `train_only` is used, the validation set encounters more out-of-vocabulary splits into shorter subwords, increasing cross-entropy per character.
- Since the baseline code defaults to `train_and_val` and challenge rules do not prohibit it, we retained `tokenizer_scope="train_and_val"` for all official submissions to remain comparable to the baseline, while documenting the exact magnitude of the leak.

## Open threads
- Test batch shuffling to improve epoch-to-epoch diversity ([07-batch-shuffling.md](07-batch-shuffling.md)).

## Sources
- Code: [`mainrun/train.py`](../../mainrun/train.py)
- Source note: `docs/_source/notes/2026-09-07-exp-4.2-tokenizer-scope.md`
