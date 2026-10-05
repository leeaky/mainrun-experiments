# `evaluate()` drops a batch_size- / block_size-dependent tail of the val set

Date: 2026-09-07. Discovered while running the §4.4 batch_size sweep.

## The bug (in the frozen `evaluate()` / `iter_full_split`)

`iter_full_split(split_ids, block_size, batch_size, device)` walks the val
token stream in strides of `span = block_size * batch_size + 1` and yields
`floor(len(val_ids) / span)` full batches. **The remainder — up to `span`
tokens — is never evaluated.**

`evaluate()` sums cross-entropy over only those covered tokens, then divides
by `len(val_text)` — the **full** character count of the concatenated val
string, which does not change with `span`.

So the reported val loss ≈ (true per-char loss) × (fraction of val tokens
actually evaluated). Larger `span` → smaller fraction → **artificially lower
reported loss.**

Measured (val = 10 000 titles = 130 300 tokens / 565 446 chars,
block_size = 128):

| batch_size | span | val tokens evaluated | coverage |
|---|---|---|---|
| 32  | 4 097  | 126 976 | 97.4% |
| **64 (baseline)** | 8 193  | 122 880 | **94.3%** |
| 128 | 16 385 | 114 688 | 88.0% |
| 256 | 32 769 | 98 304  | 75.4% |
| 512 | 65 537 | 65 536  | 50.3% (also OOMs on a 24 GB card) |

And with batch_size held at 64, `block_size`:

| block_size | span | coverage |
|---|---|---|
| 32  | 2 049 | 99.0% |
| 64  | 4 097 | 97.4% |
| 128 (baseline) | 8 193 | 94.3% |

`evaluate()` is protected (spec §1 / §8) — this can't be fixed. It just has
to be worked around in analysis.

## Which experiments are affected

**Contaminated — cross-value comparison invalid at face value:**

- **§4.4 batch_size** — heavily. The naive sweep looked like a huge win
  (`batch_size=128` −0.08 SGD, −0.05 AdamW; `256` −0.16). Coverage-adjusted
  (reported ÷ coverage-fraction, assuming a homogeneous val set):

  | | SGD reported → adj. | AdamW+sri reported → adj. |
  |---|---|---|
  | batch 32 | 1.75801 → 1.805 | — |
  | batch 64 | 1.75193 → 1.858 | 1.20428 → 1.277 |
  | batch 128 | 1.67167 → 1.900 | 1.14971 → 1.307 |
  | batch 256 | — | 1.08912 → 1.444 |

  Corrected: **smaller batch is mildly better under SGD** (more gradient
  steps at a fixed data budget), **~flat under AdamW**, and `batch_size=256`
  is clearly worse (only 245 steps). The apparent improvement was the
  artifact plus, at 256, real under-training.

- **§4.3 block_size** — mildly (and here the artifact was *hiding* a real
  effect). Coverage-adjusted:

  | | SGD reported → adj. | AdamW reported → adj. |
  |---|---|---|
  | block 32 | 1.72970 → 1.747 | 1.28564 → 1.299 |
  | block 64 | 1.76772 → 1.815 | 1.25748 → 1.291 |
  | block 128 | 1.75193 → 1.858 | 1.24794 → 1.323 |

  Corrected: **smaller `block_size` genuinely helps** — monotone under SGD
  (the raw "block64 worse than block128" was the artifact), mild under
  AdamW. This *supports* §4.3's title-length hypothesis, which the raw
  numbers had contradicted. Still confounded with the §5.6 total-steps
  lever (smaller block → more steps), so treat as suggestive, not proven.

**Clean — same `span` (8193), same 94.3% coverage, comparisons valid:**
§4.1 (AdamW), §4.5 (scaled residual init), §4.7 (dropout), §4.8 (vocab_size
— `span` unaffected; the tiny `val_ids`-length change from more tokens is
<0.5% of coverage), §4.9 (capacity — `span` unaffected).

## Consequences

1. **The combo recipe stays at `{optimizer:adamw, lr:1.2e-3,
   weight_decay:0.01, scaled_residual_init:true}` → 1.20428.** Do **not**
   add `batch_size=128/256` — that "gain" is mostly fake and the rest is
   under-training.
2. Any future combo that changes `block_size` or `batch_size` must be
   compared **coverage-adjusted**, or by re-evaluating all candidates at a
   single fixed `span` (e.g. always `evaluate` with `batch_size=64`,
   independent of the training `batch_size` — but that needs a code change
   to `evaluate`'s call site, which is protected).
3. Best defensible reading of §4.3/§4.4: keep `batch_size=64`; smaller
   `block_size` (32–64) is worth a proper, coverage-corrected combo run
   later.
