# 05 — The Evaluation Metric Coverage Artifact

> **Status:** critical discovery · **Effect:** explained false batch_size gains; prevented contaminated combo · **Date:** 2026-09-07
>
> ← [04-config-sweeps.md](04-config-sweeps.md) · [timeline](README.md) · [next](06-tokenizer-scope.md) →

## Question
Why did increasing `batch_size` to 128 and 256 yield apparently miraculous drops in validation loss?

## Reasoning
In Step 04, `batch_size=256` reported a val loss of 1.089 (compared to 1.204 at batch 64). At `batch_size=256`, 7 epochs execute only 245 total gradient steps. Achieving lower loss with one-fourth the gradient updates on identical data violates optimization fundamentals. We closely audited the evaluation pipeline in `train.py`.

## Approach
We analyzed the implementation of `iter_full_split` and `evaluate()`:
```python
def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        ...
        yield x, y

def evaluate():
    ...
    losses += loss.item() # sum of CE over evaluated batches
    return losses / len(val_text) # CONSTANT total character length
```

### The Mechanism
1. `iter_full_split` steps through validation tokens in chunks of size `span = block_size * batch_size + 1`.
2. It yields `floor(len(val_ids) / span)` batches. Any trailing remainder tokens (up to `span - 1` tokens) are **completely discarded**.
3. However, `evaluate()` divides the accumulated loss by `len(val_text)`—the **fixed, full character length** of the entire validation split (565,446 characters)!
4. Therefore:
   $$\text{Reported Loss} \approx \text{True Loss} \times \left(\frac{\text{Tokens Actually Evaluated}}{\text{Total Tokens}}\right)$$
Larger `batch_size` or `block_size` yields a larger `span`, dropping a much larger chunk of validation data while dividing by the same constant denominator!

## Results

### Validation Set Coverage Analysis (Val Split = 130,300 tokens / 565,446 chars)
| `batch_size` | `block_size` | Span | Val Tokens Evaluated | Coverage % | Raw Val Loss | Coverage-Adjusted Val Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 | 128 | 4,097 | 126,976 | 97.4% | 1.75801 | 1.805 |
| **64 (default)** | **128** | **8,193** | **122,880** | **94.3%** | **1.75193** | **1.858** |
| 128 | 128 | 16,385 | 114,688 | 88.0% | 1.67167 | 1.900 |
| 256 | 128 | 32,769 | 98,304 | 75.4% | 1.08912 | **1.444** (severely underfitted) |

### Corrected Interpretation
- Under SGD: Smaller `batch_size` (32) is actually *better* than 64 (adj. 1.805 vs 1.858) because more steps were taken.
- Under AdamW: `batch_size=64` remains optimal (adj. 1.277). `batch_size=256` is by far the worst (adj. 1.444).
- Clean comparisons: Any experiment holding `block_size=128` and `batch_size=64` constant shares the identical span (8,193) and coverage (94.3%), meaning comparisons within that group are **100% valid**.

## What we learned
- Always verify metrics when a hyperparameter sweep exhibits an unexpectedly dramatic gain.
- Because `evaluate()` is protected under the challenge rules, we cannot rewrite its logic to evaluate fractional batches.
- **Rule:** We kept `batch_size=64` and `block_size=128` locked in the recipe to keep all comparisons on an identical, uncontaminated footing.

## Open threads
- Examine data leakage in tokenizer training ([06-tokenizer-scope.md](06-tokenizer-scope.md)).

## Sources
- Source note: `docs/_source/notes/2026-09-07-evaluate-val-coverage-artifact.md`
- Code: `mainrun/train.py` (`evaluate`, `iter_full_split`)
