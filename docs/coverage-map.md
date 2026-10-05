# Coverage map — what we tested, by pipeline stage

Every experiment mapped to the component of the `mainrun` training pipeline
it touches, so the exploration coverage is visible. Verdicts and detail:
`2026-09-07-SUMMARY.md` + the per-experiment notes. Rationale for each
choice: `2026-09-07-decisions-log.md`.

Legend: **swept** = multiple values compared; **sampled** = one or two
representative variants tried; **exposed** = made a config field but not
varied; ⚠️ = in-scope (`train.py` edit, nothing protected) but untested.

---

## 1. Data acquisition & split — `get_titles()`

Load HN titles → shuffle by seed → 90/10 split.

| lever | coverage | verdict |
|---|---|---|
| — | none possible | `num_titles` (100k), `seed` (1337), `val_frac` (0.10), the dataset — all **protected**. Zero degrees of freedom. |

## 2. Tokenizer — byte-level BPE

| lever | coverage | verdict |
|---|---|---|
| `vocab_size` (8k/16k/20k/24k/28k, all 3 optimizers) | **swept** | **16000** — worse in both directions (§4.8 + upsweep) |
| `tokenizer_scope` (train vs train+val corpus) | **swept** | keep **train+val** (leaky); `train_only` is +0.07–0.12 (§4.2) |
| `vocab_precheck.py` diagnostic (compression knee, `frac_undertrained`) | built | informed the sweep range |
| pre-tokenizer regex, BPE dropout, special-token set, char/byte tokenizer | ⚠️ untested | low EV; regex is `ByteLevel` fixed |

## 3. Data feed & batching

| lever | coverage | verdict |
|---|---|---|
| `block_size` (32/64/128, + product-held `block64·batch128`) | **swept** | **128** — shorter is a wash once confounds removed (§4.3, Ph11) |
| `batch_size` (16/24/32/64/128/256) | **swept** | **64** — smaller monotonically worse under Muon (§4.4 + follow-up) |
| `shuffle_batches` (on/off) | **swept** | **on** — −0.01 under AdamW/Muon (§4.6) |
| cross-`<eos>` attention masking (docs bleed together now) | ⚠️ untested | — |
| length bucketing, curriculum order, overlapping windows | ⚠️ untested | overlapping windows rejected in design (decision F) |

## 4. Embeddings — `token_emb`, learned `pos_emb`, tied head

| lever | coverage | verdict |
|---|---|---|
| `d_model` (384/512/640, capacity sweep) | **swept** | **512** (§4.9) |
| embedding → AdamW side of the Muon hybrid | design choice | correct — Muon on lookups is harmful (Ph10) |
| `weight_decay_embeddings` toggle | **exposed / tested** | keep decaying; off = +0 (Ph11) |
| **positional-encoding scheme** (RoPE / ALiBi / NoPE / sinusoidal) | ⚠️ **untested** | **biggest untouched architectural axis**; ~−0.01 at best, blunted by 12-token titles |
| untying the head; embedding scaling (×√d_model); factorized embeddings | ⚠️ untested | untie adds 8.2M params → §4.9 says no |

## 5. Attention — `CausalSelfAttention` (manual QKᵀ→softmax→matmul)

| lever | coverage | verdict |
|---|---|---|
| `attn_impl` (manual vs SDPA) | **swept** | **manual** — equivalent (1.9e-7); SDPA +0.0016 from dropout RNG (§4.11) |
| `n_head` (6/8/10, head_dim held 64, capacity sweep) | **swept** | **8** (§4.9) |
| QK-norm (RMSNorm on q,k) | **sampled** | **rejected** — +0.010 (Ph11) |
| `qkv`/`proj` → Muon params | design choice | in (Ph10) |
| split fused `qkv` into 3 for orthogonalization | ⚠️ untested | deferred; low EV after SwiGLU/QK-norm both hurt |
| GQA/MQA, `bias=False` on qkv/proj, logit softcap, scale ≠ `1/√d`, `head_dim` ≠ 64, local attention | ⚠️ untested | GQA reduces capacity — wrong direction |

## 6. MLP — `Linear(d,4d) → GELU → Linear(4d,d) → Dropout`

| lever | coverage | verdict |
|---|---|---|
| SwiGLU vs GELU (param-matched) | **sampled** | **GELU** — SwiGLU +0.006 (Ph11) |
| width (via `d_model`, capacity sweep) | **swept** | baseline (§4.9) |
| MLP matrices → Muon params | design choice | in |
| expansion ratio ≠ 4×, other activations (ReLU², GEGLU), `bias=False` | ⚠️ untested | more params → §4.9 says no |

## 7. Normalization — `nn.LayerNorm`, pre-LN

| lever | coverage | verdict |
|---|---|---|
| QK-norm (adds RMSNorm on q,k) | **sampled** | rejected (Ph11) |
| stop weight-decaying LN gains/biases on the Muon branch | fixed (Ph11 A1) | correct; ~0 effect |
| **RMSNorm vs LayerNorm** for the block norms | ⚠️ **untested** | ~0 expected (speed/param optimization, not accuracy) |
| **LN placement** (pre / post / sandwich / peri-LN) | ⚠️ untested | pre-LN is already the right choice; post-LN is worse |
| LN `bias=False`, LN epsilon | ⚠️ untested | — |

## 8. Residual structure & init

| lever | coverage | verdict |
|---|---|---|
| `scaled_residual_init` (depth-scaled std on the 2 residual projections) | **swept** | **the one lever that reversed** — in under AdamW (−0.05), out under Muon (+0.002) (§4.5, Ph10) |
| `n_layer` (4/6/8, capacity sweep) | **swept** | **6** (§4.9) |
| **base init std** (0.02) & scheme (truncated-normal / orthogonal) | ⚠️ **untested** | ~±0.005; §4.5 already probed the impactful init lever |
| **aspect ratio** (depth vs width at fixed params) | ⚠️ **untested** | capacity sweep co-varied both — isolating it is genuinely open, ~coin-flip ±0.01 |
| LayerScale / ReZero, stochastic depth | ⚠️ untested | — |

## 9. Forward pass & loss — `head(x)` → `F.cross_entropy`

| lever | coverage | verdict |
|---|---|---|
| the `evaluate()` coverage artifact | discovered | worked around — hold `block·batch` fixed (`evaluate-val-coverage-artifact.md`) |
| char-vs-token denominator | documented | metric is nats/char, not perplexity |
| label smoothing, z-loss, loss weighting, logit softcap | ⚠️ untested | training loss is editable (only `evaluate()` is frozen); §4.7 says underfitting → regularizers cost |

## 10. Optimizer — **exhaustively explored**

| lever | coverage | verdict |
|---|---|---|
| SGD → AdamW | **swept** | **−0.50** (§4.1) |
| AdamW → **Muon hybrid** | **swept** | **−0.018** (Ph10) |
| matrix `lr` bracket | **swept** | 1.2e-3 |
| embed-group `lr` / `β1,β2` / `weight_decay` | **swept** | 0.9e-3 / 0.9,0.99 / 0.02 — but redundant with EMA (Ph11) |
| `ndim≥2` param grouping | decision G | standard nanoGPT rule |
| `muon_lr` / `muon_momentum` / `muon_ns_steps` | **swept** | 0.02 / 0.95 / 5 |
| `grad_clip_norm` | **exposed** | kept 1.0, not swept |
| Lion / Sophia / AdEMAMix / SOAP / schedule-free; per-layer LR; AdamW `eps` | ⚠️ untested | Muon was the better optimizer bet |

## 11. LR schedule

| lever | coverage | verdict |
|---|---|---|
| cosine → WSD | **swept** | **−0.0025** (§4.12) |
| `warmup_frac` | **swept** | **0.06** |
| `stable_frac` (WSD) | **swept** | 0.5–0.6, wash (Ph11) |
| `wsd_decay_shape` (linear vs 1−√) | **swept** | invsqrt −0.0005, marginal (Ph11) |
| `lr_floor_frac` | **exposed** | field exists, default 0.1, **never swept** |
| warmup shape, restarts, per-group schedules | ⚠️ untested | — |

## 12. Precision & training dynamics

| lever | coverage | verdict |
|---|---|---|
| `use_amp` (bf16 autocast) | **swept** | loss-neutral, **off** — breaks determinism (§4.12) |
| determinism flags | set | noise floor **0.0** |
| gradient accumulation, `torch.compile`, tf32 | ⚠️ untested | grad-accum → fewer steps, low EV |

## 13. Evaluation-time transform

| lever | coverage | verdict |
|---|---|---|
| **weight EMA at eval** | **swept** | **−0.0065**, decay 0.997, hard cliff after ~0.997 (Ph10) |
| checkpoint averaging; report best-step vs final-step | ⚠️ untested | checkpoint-avg ≈ EMA |

## 14. Regularization (cross-cutting)

| lever | coverage | verdict |
|---|---|---|
| `dropout` (0.0 / 0.05 / 0.1) | **swept** | **0.1** — near-null, underfitting regime (§4.7) |
| weight decay | via optimizer | — |
| label smoothing, stochastic depth, separate embedding/attention dropout | ⚠️ untested | underfitting → any regularizer costs |

---

## Coverage verdict

**Thoroughly swept — where the exercise's degrees of freedom live and
where the −0.586 came from:** optimizer, LR schedule, tokenizer/vocab, data
batching, model capacity, the eval-time EMA.

**Sampled, not swept — model-architecture internals:** positional encoding,
norm type/placement, attention mechanism, MLP structure, init scheme. We
tested a representative diagonal (SDPA, QK-norm, SwiGLU,
scaled-residual-init) and **every one hurt or washed.** Combined with §4.9
proving the model is **budget-limited, not architecture-limited**, this is
strong evidence the rest wouldn't move it — but it is not exhaustive.

**The three ⚠️ items with any residual chance of a small (~−0.01) win:**
RoPE/NoPE positional encoding, RMSNorm, aspect-ratio (depth vs width at
fixed params). Each is a legitimate `train.py` edit; the prior after
Session B is low.

**Genuinely locked, zero exploration possible:** the dataset, the split,
`epochs`, `seed`, `evaluate()`.
