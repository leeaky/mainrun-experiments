# mainrun experimentation — summary (2026-09-07)

Branch `training-experiments` off `main`. All work done this session;
nothing merged (per instruction — merge at the end).

## Result

| | val loss | notes |
|---|---|---|
| baseline (`task train`, no override) | **1.75193** | reproduced bit-for-bit on RTX 4090, deterministic |
| best AdamW recipe (Phases 1–9) | 1.19206 | −0.560, deterministic |
| **best recipe (Muon + EMA)** | **1.16566** | **−0.586**, deterministic |
| best recipe *without* EMA (Phase 11 `R_A`) | 1.16854 | −0.567 |

(`1.16566` = what current `main` reproduces. Phase 10 first reported
`1.16539`; the Session B fix that stopped weight-decaying LayerNorm
gains/biases on the Muon branch shifted it +0.00027 — more correct,
marginally worse. Committed reference trace: `results/logs/recipe-muon-ema.log`.)

**Best recipe** (a `MAINRUN_CONFIG` JSON) — branch `phase10-muon-ema`:
```json
{"optimizer": "muon", "muon_lr": 0.02, "lr": 0.0012, "weight_decay": 0.01,
 "shuffle_batches": true, "lr_schedule": "wsd", "ema_decay": 0.997}
```
Note `scaled_residual_init` is **off** here — it helps under AdamW but hurts
under Muon.

**Phase 11 / Session B did not improve on this.** It tuned the AdamW-group
hyperparameters (`lr 0.0009`, `weight_decay 0.02`, `β2 0.99` → −0.0033 on
the *no-EMA* base) but that gain is redundant with EMA — end-of-training
tuning and eval-time EMA are substitutes. `R_A` (the tuned no-EMA recipe:
`lr 0.0009, weight_decay 0.02, beta2 0.99`, no `ema_decay`) is the best
option if you don't want EMA. SwiGLU (+0.006), QK-norm (+0.010),
`vocab_size>16000`, `block_size=64`, `ns_steps=6` all ruled out.
`wsd_decay_shape:"invsqrt"` is a real but marginal −0.0005.
(`2026-09-07-phase11-session-b-results.md`)

Contribution of each lever (measured where it lands in the stack):

| lever | ~Δ | § |
|---|---|---|
| SGD → AdamW (lr 1.2e-3 from a bracket, decoupled weight decay via nanoGPT `ndim≥2` param groups, linear warmup → cosine-to-floor) | **−0.50** | 4.1 |
| AdamW → **Muon** (hybrid: 2D hidden matrices on Muon, embed/head/norm/bias on AdamW; `muon_lr=0.02`) | **−0.018** | Ph10 |
| scaled residual init (`0.02/√(2·n_layer)` on the two per-block residual projections) — **AdamW only** | −0.05 (AdamW) / **+0.002 (Muon)** | 4.5 |
| eval-time weight EMA (`decay=0.997`, shadow swapped in around `evaluate()`) | −0.0065 | Ph10 |
| batch shuffling (per-epoch chunk shuffle, independent RNG) | −0.01 | 4.6 |
| WSD LR schedule (warmup-stable-decay) | −0.0025 | 4.12 |

Everything else swept is neutral, negative, or a metric artifact (below).

## What was built

**Config plumbing** (`mainrun/config.py`, new): `Hyperparameters` moved here +
`load_config()` reads `$MAINRUN_CONFIG` JSON, rejects unknown keys, the 5
protected keys, wrong types, and out-of-set choices. 24 unit tests. Every
new field defaults to baseline behaviour — empty config reproduces
`1.7519307076847304` exactly.

**Logging** (`mainrun/train.py`): added `dataset_info.max_steps` +
`avg_tokens_per_val_title`, `validation_step.train_loss`, and
`optimizer_info` / `training_summary` / `tokenizer_overlap_info` events.
Determinism flags (`CUBLAS_WORKSPACE_CONFIG`,
`use_deterministic_algorithms`) — **GPU noise floor is 0.0**, no
nondeterministic-op warnings.

**Toggles** (all in `config.py` + `train.py`, all default = baseline):
`optimizer`, `beta1/beta2`, `warmup_frac`, `lr_floor_frac`,
`scaled_residual_init`, `tokenizer_scope`, `shuffle_batches`, `attn_impl`,
`grad_clip_norm`, `use_amp`, `lr_schedule` + `stable_frac`.

**Dashboard** (`tools/`): a Streamlit config-form + run-launcher + history
overlay was built to spec §6, then **removed 2026-09-07** — the sweeps were
driven by pod scripts, the dashboard was never used, and a UI that looks
like a results dashboard but isn't one caused more confusion than value.
The results record is these notes.

## The metric artifact (important)

`evaluate()` / `iter_full_split` (protected, unchangeable) stride the val
set by `span = block_size·batch_size + 1` and **drop the `len(val_ids) %
span` remainder** while dividing by the full char count. Baseline covers
94.3% of val tokens; `batch_size=128` → 88.0%, `256` → 75.4%. **Larger
`block_size`/`batch_size` ⇒ artificially lower reported loss.**

- **§4.4 batch_size**: the apparent big win was entirely this + real
  under-training. `batch_size` stays at 64.
- **§4.3 block_size**: the artifact hid a real (small) SGD-only benefit of
  smaller windows; under the AdamW combo, coverage-corrected, `block_size`
  is a wash — stays at 128.
- All other experiments hold `block_size·batch_size` at baseline and are
  clean.

Full detail: `2026-09-07-evaluate-val-coverage-artifact.md`.

## Per-experiment verdicts

| § | experiment | verdict |
|---|---|---|
| 4.1 | SGD → AdamW | **−0.50, the dominant lever.** lr=1.2e-3 (bracketed). |
| 4.2 | tokenizer scope | `train_only` is **+0.07–0.12 worse** — the val-set leak into the BPE vocab (overlap 0.929) is worth a real slice of the metric. Keep `train_and_val`; disclose the leak. |
| 4.3 | block_size | Confounded by the coverage artifact. **Settled Ph11:** `block64·batch128` with the product held at 8192 (coverage- and step-clean) is a **dead wash** (+0.00004) under the Muon combo. Keep 128. |
| 4.4 | batch_size | Coverage artifact + under-training. **Settled Ph11:** `batch_size ∈ {16,24,32}` on the Muon+EMA recipe, coverage-adjusted — **monotonically worse** (bs32 +0.025, bs16 +0.16), the *opposite* of the SGD-era hint. Muon+AdamW isn't step-starved; it's data-limited. Keep 64. |
| 4.5 | scaled residual init | **−0.05 under AdamW**, deterministic. Contradicts §5.7 (helps SGD too) and "smoother early loss" (it doesn't). **Reverses to +0.002 under Muon** (Ph10) — off in the Muon recipe. |
| 4.6 | batch shuffling | Null under SGD, **−0.01 under AdamW**, still −0.0014 under Muon — exactly §5.7's prediction. Keep. |
| 4.7 | dropout | Near-null (underfitting regime). 0.05 marginally best single-variable, but **hurts in the combo** — keep 0.1. |
| 4.8 | vocab_size | 8000 is **worse** (+0.02 to +0.12). Upward sweep 16k/20k/24k/28k under the final recipe: **none beats 16000** (+0.008 to +0.073). Compression rises but `frac_undertrained` predicts the degradation; leaky/non-leaky tokenizer bases agree within ~1%. Keep 16000 — confirmed both directions (`2026-09-07-exp-4.8-vocab-upsweep.md`). |
| 4.9 | capacity | Baseline 27M is the sweet spot under AdamW — **budget-limited, not capacity-limited**. Keep 6/8/512. |
| 4.10 | best-checkpoint tracking | Done — `training_summary` event (metric only, no weights). |
| 4.11 | manual → SDPA | Numerically equivalent (1.9e-7, eval). +0.0016 in-combo from the dropout-path difference. Keep `manual`. |
| 4.12 | amp / grad_clip / wsd | amp loss-neutral; **wsd −0.0025** (kept, still helps under Muon); grad_clip exposed, not swept. |
| Ph10 | **Muon hybrid optimizer** | **−0.018**, deterministic (fp32 Newton–Schulz). `muon_lr` sharp optimum at 0.02; AdamW group (embed/norm) keeps `lr=0.0012`; momentum 0.95. `sri` drops out. (`2026-09-07-phase10-session-a-results.md`) |
| Ph10 | **eval-time weight EMA** | **−0.0065** on top of Muon (−0.0104 on AdamW). `decay=0.997`, hard cliff after ~0.997 (init contamination). Pure eval transform. |
| Ph11 | AdamW-group tuning under Muon | `lr 0.0009` / `wd 0.02` / `β2 0.99` → **−0.0033 on the no-EMA base**, **~0 with EMA** (substitutes). `warmup_frac` unchanged. |
| Ph11 | SwiGLU MLP (param-matched) | **+0.006** — worse. Budget-limited; architecture has no room. |
| Ph11 | QK-norm | **+0.010** — worse. |
| Ph11 | wsd `invsqrt` decay / `ns_steps=6` / `stable_frac` | invsqrt real but −0.0005; ns6 +0.0009; stable_frac a wash (0.5 best by 0.0001). |

## Files

- Spec (revised + review): `docs/specs/2026-09-05-training-experimentation-design.md`
- Plans: `docs/journey/phase{1..6,9}-*.md`
- Notes: `docs/journey/*.md` (this file, the artifact
  note, and one per experiment)
- Code: `mainrun/config.py`, `mainrun/train.py`, `mainrun/muon.py`, `mainrun/ema.py`
- **Reference logs**: `results/logs/{baseline-1001,recipe-adamw,recipe-muon-ema}.log`
  — full bit-exact-reproducible traces of the three anchor configs, current
  `main`. See `results/logs/README.md`.
- **Sweep drivers**: `mainrun/experiments/` — the pod orchestration scripts,
  one per phase, each mapped to its results note. Plus `analyze.py` and the
  `vocab_precheck.py` diagnostic.
- RunPod infra: `docs/notes/mainrun-runpod-setup` context —
  template `9rwvm97wd9`, Secure Cloud RTX 4090 (Community is unusable with
  this custom image).

## Status — complete

The experiment surface is exhausted. Session B found no lever left in the
AdamW-group hyperparameters or the cheap architecture tweaks; the
smaller-`batch_size` follow-up closed the last "more optimizer steps"
thread (monotonically worse). Further gains would need more steps or more
data — both locked by the exercise.

Full stage-by-stage exploration audit: `2026-09-07-coverage-map.md`.
Short version: optimizer / schedule / tokenizer / batching / capacity /
eval-EMA thoroughly swept; model-architecture internals (positional
encoding, norm type, attention mechanism, MLP structure, init) sampled —
every variant tried hurt or washed, and §4.9 shows the model is
budget-limited not architecture-limited. The 3 untested items with any
residual chance of ~−0.01: RoPE/NoPE, RMSNorm, aspect-ratio.

**Done:**
- All phases merged into `main` (fast-forward, 53 commits), feature
  branches deleted.
- Pushed to a **private** repo: `github.com/leeaky/mainrun-experiments`
  (`origin`); `MaincodeHQ/mainrun` kept as `upstream`.

**Dropped (deliberately, not pending):**
- **The Streamlit dashboard** (spec §6, Phase 9) — removed from the repo
  2026-09-07. Built but never used; the pod driver scripts did the sweeps,
  and a UI that looked like a results dashboard but wasn't one caused
  confusion. `git log -- tools/` recovers it.
- **Split the fused `qkv`** for Muon — low EV now that SwiGLU and QK-norm
  both hurt.

**Not committed** (left in the working tree as the user had them):
`.gitignore` edit, `mainrun/vocab_precheck.py`, `mainrun/test_vocab_precheck.py`,
`.devcontainer/Dockerfile.runpod`, `scripts/runpod-verify.sh`.
