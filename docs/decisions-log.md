# Decisions log — mainrun experimentation

One place for every choice made in this exercise and why. Results live in
`2026-09-07-SUMMARY.md`; the per-experiment detail lives in the other
`2026-09-07-*.md` notes. This file is the "why did we do it that way"
record.

Last updated: 2026-09-07 (through Phase 11 / Session B — complete).

---

## 0. Fixed constraints (not our choices — the exercise's rules)

| constraint | source |
|---|---|
| `epochs=7`, `seed=1337`, dataset + 90/10 split, `evaluate()` — unchanged | assessment rules; kept because a fixed small-data budget is what makes the comparisons meaningful |
| every real run goes through `task train` **inside the devcontainer** (Ubuntu 22.04, Py 3.10.12, pinned `requirements.txt`) | fidelity to the graded environment |
| this is **personal learning, not a submission** — nothing uploaded | user |

Protected `MAINRUN_CONFIG` keys (loader raises on override): `epochs`,
`seed`, `num_titles`, `val_frac`, `log_file`.

---

## 1. Infrastructure

### 1.1 RunPod **Secure Cloud** RTX 4090, not Community
Community Cloud ($0.34/hr) is half the price but **unusable with this
custom image** — `torch.cuda.init()` → "CUDA unknown error (999)" on 4/4
Community hosts despite `nvidia-smi` working. Root cause: broken
nvidia-container-toolkit on unmanaged operators; nothing bakeable into the
image fixes it (tried `NVIDIA_VISIBLE_DEVICES=all` etc. in a `:runpod`
image tag — still failed). Secure Cloud RTX 4090 ($0.74/hr) works first
try. **Decision: eat the 2× cost for a working GPU.**

### 1.2 Devcontainer image **as** the pod image (Model B), not devcontainer-CLI-on-pod (Model A)
RunPod pods have no Docker daemon, so `devcontainer up` can't run. The
devcontainer CLI only does `docker run` + workspace mount + root user —
all of which the pod provides natively. **Decision: run `task train`
directly in the pod shell against a fresh `git clone`; image stays 100%
pristine (`leeaky/mainrun-devcontainer:latest`, built from
`.devcontainer/Dockerfile` unchanged).**

### 1.3 No network volume
A volume would persist the HF dataset cache (~137 MB) across pods, saving
~10 s/boot. But it **bills while the pod is stopped** and the exercise is
not on a schedule. **Decision: no volume; re-clone + re-download each
session (~15 s). Delete every pod at end of session.**

### 1.4 Code transfer by **scp**, not git patches
Local `main` is one commit ahead of `origin/main` (`18e9368` "Mainrun auto
checkpoint") and that commit **rewrote every file's line endings** (CRLF).
So `git diff main..branch` patches don't apply cleanly to a fresh clone.
**Decision: scp the final file contents straight into the pod's clone
(Python doesn't care about line endings).**

### 1.5 SSH via the pod's **direct** endpoint (Secure Cloud only)
Secure Cloud pods get a real public IP + a mapped port 22 →
`ssh root@<ip> -p <port>` and `scp -P <port>` both work non-interactively.
Community pods only expose `ssh.runpod.io`, an **interactive-PTY-only
proxy** where `ssh host 'cmd'` fails. Another reason for §1.1.
Claude's key: `~/.ssh/runpod_mainrun`, registered on the user's RunPod
account. Template `9rwvm97wd9` carries the sshd bootstrap start command.

### 1.6 fp32 Newton–Schulz in Muon, not bf16
Keller Jordan's Muon runs the NS orthogonalization in bf16 for speed. bf16
matmul broke run-to-run bit-exactness in the §4.12 AMP experiment, and the
whole methodology (§2.2) depends on a 0.0 noise floor. NS is a negligible
fraction of step time here (small model). **Decision: fp32 NS. Confirmed
Muon is bit-exact deterministic.**

---

## 2. Methodology

### 2.1 `MAINRUN_CONFIG` env → JSON override file, with strict validation
Every experiment is a JSON dict applied over `Hyperparameters()` via
`dataclasses.replace`. The loader **raises** on: unknown key → protected
key → wrong type → out-of-`_CHOICES` value (in that order). Empty/absent
config = exact defaults. **Why strict:** a typo'd key silently ignored
would invalidate a run without any signal.

### 2.2 **One run per config** — because the GPU noise floor is 0.0
The spec (§2, written beforehand) assumed GPU nondeterminism and mandated
"run every experiment 3×, compare means". In practice, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` +
`torch.use_deterministic_algorithms(True, warn_only=True)`, **the RTX 4090
reproduces every run bit-for-bit** (baseline `1.7519307076847304` and the
AdamW recipe `1.1920606836240772` both confirmed identical across
re-runs; no nondeterministic-op warnings fired). **Decision: one run per
config.** Every Δ in the notes is a real difference, not a mean over
noise. This is the single biggest deviation from the spec and it made the
whole sweep ~3× cheaper.

### 2.3 Gate discipline: every code change re-proves the baseline
After any change to `config.py` / `train.py` / new module, the first two
runs on the pod are: empty config → `1.7519307076847304` bit-exact, and
the current recipe → its known value bit-exact. A bug in (say) the Muon
branch then can't silently corrupt a SwiGLU number three phases later.

### 2.4 Config-gated toggles that default to baseline behaviour
Every new field defaults to the value that reproduces prior behaviour
(`optimizer="sgd"`, `ema_decay=0.0`, `mlp_kind="gelu"`, …). New code paths
are `if`-guarded so an off toggle is a true no-op. This is what makes §2.3
cheap.

### 2.5 Branch per phase, **nothing merged**
`training-experiments` (Phases 1–9, + the §4.8 vocab upsweep) →
`phase10-muon-ema` (Phase 10) → same branch for Phase 11. User's standing
instruction: "keep all branches as-is, we will merge at the end." A dud
experiment still lands as a default-off toggle (like `sdpa`, `use_amp`)
with a note saying "don't use it".

### 2.6 The `evaluate()` coverage artifact — worked around, not fixed
`evaluate()` / `iter_full_split` are protected. They stride the val set by
`span = block_size·batch_size + 1` **and drop the `len(val_ids) % span`
remainder**, while dividing the summed loss by the full char count. So
**larger `block_size`/`batch_size` ⇒ fewer val tokens scored ⇒
artificially lower reported loss** (coverage 94.3% at baseline → 75.4% at
`batch_size=256`). **Decision:** hold `block_size·batch_size` at the
baseline product (8192) for every experiment except the §4.3/§4.4 sweeps
themselves, whose raw numbers are then coverage-corrected in analysis.
This invalidated the naive §4.3/§4.4 "wins". Full detail:
`2026-09-07-evaluate-val-coverage-artifact.md`.

### 2.7 The metric is nats-per-**character**, not perplexity
`evaluate()` divides by `len(val_text)` (a character count), not a token
count. So val loss ≈ 1.75 is not a perplexity and only comparable across
runs **while `block_size·batch_size` is fixed** (see §2.6). Compression
(`val_chars_per_token`) therefore matters to the metric — which is why the
vocab_size question needed the diagnostic (§3.8).

---

## 3. The recipe — every lever, in / out, and why

Final best: **val 1.16566**, Δ −0.586 vs baseline 1.75193. (Phase 10 first
reported 1.16539; the §3.4b Session B fix — stop weight-decaying LayerNorm
gains/biases on the Muon branch — shifted it +0.00027. `1.16566` is what
current `main` reproduces; trace committed at
`results/logs/recipe-muon-ema.log`.)
```json
{"optimizer":"muon","muon_lr":0.02,"lr":0.0012,"weight_decay":0.01,
 "shuffle_batches":true,"lr_schedule":"wsd","ema_decay":0.997}
```

### 3.1 Optimizer: SGD → AdamW → **Muon** — IN
- **SGD→AdamW: −0.50**, the dominant lever. `lr` bracketed to `1.2e-3`
  (spec guessed `4e-4` → 1.2775; the bracket found an interior min at
  1.2e-3 → 1.24794). Decoupled weight decay via the nanoGPT **`ndim ≥ 2`
  param-group rule** (this was open decision **G** in the review — chosen
  over "decay everything" and "hand-list modules" because it's the
  standard and needs no maintenance). Warmup (`warmup_frac=0.06`) →
  cosine-to-floor.
- **AdamW→Muon (hybrid): −0.018.** Muon orthogonalizes the momentum of the
  2D hidden matrices (Newton–Schulz); embeddings / tied head / LayerNorm /
  biases stay on AdamW. `muon_lr` has a **knife-edge optimum at 0.02**
  (0.01 undertrains, ≥0.05 diverges to 1.4+). The AdamW group keeps
  `lr=0.0012`. Momentum 0.95.
- **Param split rule** (`split_params_for_muon`): `p.ndim == 2 and
  "token_emb" not in name` → Muon. The user was offered the design
  decision and chose "no deviation" (standard rule). Muon on the embedding
  table is known-harmful (rows are lookups, not a linear map).

### 3.1a AdamW is not a skippable stepping stone — Muon *contains* it

**`optimizer: "muon"` is a hybrid, not a replacement.** The Muon branch in
`train.py` constructs **both** a `Muon` optimizer (the 24 hidden weight
matrices) **and** a `torch.optim.AdamW` (everything else: the 8.2M-param
embedding table + tied head, `pos_emb`, every LayerNorm gain/bias, every
Linear bias) and steps them together via `CombinedOptimizer`. You cannot
run Muon without an Adam-family optimizer alongside it.

Why it must be split: Newton–Schulz asserts `G.ndim == 2` and only makes
sense for weights that act as a *linear map* on the residual stream. An
embedding table is per-token lookups; LayerNorm gains/biases are 1D. Those
need a normal optimizer, and **adaptive (AdamW) beats SGD there** — token
gradients vary wildly in scale (Session B found the AdamW-group `lr` has a
sharp optimum at 0.0009).

**What *was* a superseded intermediate:** Phases 2–9 ran the *full* model —
matrices included — on AdamW; Phase 10 moved the matrices to Muon. So the
"AdamW recipe" (1.19206) is an artifact, not a shipped config. It was not
wasted work — it set the −0.50 reference, bracketed the `lr`/`betas` that
Muon's AdamW half reuses, and was the baseline every other lever (sri,
shuffle, wsd) was measured against (some *reversed* under Muon — you only
catch that with a reference point).

**The −0.50 and −0.018 are not cleanly additive.** The −0.50 is really
"adaptive per-coordinate optimization for an embedding-heavy 27M model."
Muon's AdamW half *already delivers most of that* for the embedding, so the
−0.018 is specifically what *matrix orthogonalization* adds **on top of**
already-adaptive optimization everywhere. A one-shot `SGD → hybrid-Muon`
jump lands at ~1.17 directly; the decomposition is bookkeeping a
"which-lever-matters" exercise wants, not extra capability.

### 3.2 `scaled_residual_init` — IN under AdamW, **OUT under Muon**
Depth-scaled std (`0.02/√(2·n_layer)`) on the two per-block residual-path
projections. **−0.05 under AdamW** (contradicts the spec's §5.7 prediction
that it would be null under SGD — it helps SGD too). **Reverses to +0.002
under Muon** — Muon's orthogonalized update already bounds per-step motion,
so the init shrink is redundant and just slows those layers. **Decision:
dropped from the Muon recipe.**

### 3.3 `shuffle_batches` — IN
Per-epoch chunk shuffle on an independent RNG (`random.Random(seed+epoch)`).
Null under SGD (+0.0003), **−0.01 under AdamW**, still −0.0014 under Muon —
exactly the §5.7 prediction (matters once the optimizer exploits
per-step gradient structure). Review decision **F**: non-overlapping
chunks (drop the `% span` remainder) accepted rather than a sliding
window — simpler, and the dropped tail is < 0.2% of train.

### 3.4 `lr_schedule = "wsd"` — IN
Warmup-stable-decay beats cosine by −0.0025 (AdamW) and still −0.0013
under Muon. `stable_frac` swept in Session B — **0.5 marginally best,
0.4–0.6 within 0.0004, a wash; kept at 0.6**. `wsd_decay_shape="invsqrt"`
(1−√ tail) is a real but marginal **−0.0005** — offered as an optional add,
not baked into the headline recipe.

### 3.4b AdamW-group hyperparameters under Muon (Session B)
The AdamW group (embeddings + norms + biases) was re-tuned once Muon owned
the matrices: **`lr` 0.0012→0.0009, `weight_decay` 0.01→0.02, `β2`
0.95→0.99**, `warmup_frac` unchanged at 0.06. This is **−0.0033 on the
no-EMA base** (`R_A` = 1.16854 vs `RM` = 1.17187) but **~0 once EMA is
on** — end-of-training hyperparameter tuning and eval-time EMA are
**substitutes**, both smoothing the same end-of-training wobble.
**Decision:** the headline recipe keeps the Phase 10 values (simpler,
marginally better with EMA); `R_A`'s values are documented as the best
*no-EMA* recipe. Also settled: the Muon-branch AdamW optimizer now splits
decay/no-decay by `ndim` (was one flat group wrongly decaying LayerNorm
gains) — correct, but worth only −0.00003.

### 3.5 Eval-time weight **EMA** — IN
Shadow copy of the weights, updated after every step, swapped in around the
`evaluate()` call and swapped back. Pure eval transform — training,
step count, RNG untouched. **−0.0104 on the AdamW recipe, −0.0065 on
Muon.** `decay = 0.997`; **hard cliff after ~0.997** (`0.999^994 ≈ 0.37`
→ a third of the shadow is still random init at the last step → blows up
to 1.33). The classic `(1+step)/(k+step)` warmup ramp was removed — it
never saturates in a ~1000-step run and just clamped every decay to
~0.991.

### 3.6 `tokenizer_scope` — kept at `train_and_val` (the leaky default)
`train_only` (the non-leaky, "correct" choice) is **+0.07 to +0.12
worse** — the val-set's contribution to the BPE merge list is worth a real
slice of the char-normalized metric (vocab overlap 0.929). **Decision:
keep the leak, disclose it.** Review decisions **E** (run the overlap
diagnostic only during this comparison) and **F** (accept non-overlapping
vocab, document) both resolved here.

### 3.7 `block_size` / `batch_size` — unchanged (128 / 64) — SETTLED
Both "wins" were the §2.6 coverage artifact plus real under-training.
**Session B ran the clean test:** `block64·batch128` with the product held
at 8192 (coverage- and step-clean, 987 vs 994 steps) → **+0.00004, a dead
wash** under the Muon combo. So the "shorter context helps" hypothesis
(titles ~12 tokens ≪ 128) does not survive once the confounds are removed.
`batch_size` `256`+ OOMs a 24 GB card anyway. **Keep 128 / 64.**

**Smaller `batch_size` follow-up (2026-09-07):** `{16, 24, 32}` on the full
Muon+EMA recipe, coverage-adjusted, was **monotonically worse** (bs32
+0.025, bs24 +0.068, bs16 +0.162) — the *opposite* of the weak SGD-era
hint. `epochs` fixed at 7 means smaller batch ≈ more steps (bs16 → 3976 vs
994), but Muon+AdamW is **not step-starved** the way baseline SGD was — at
the fixed 90k-title budget it's **data-limited**, and more, noisier passes
over the same data just hurt. EMA's benefit also shrank with batch (bs64
−0.006 → bs16 −0.001). **Closes the "does the model just need more
optimizer steps?" thread: no.** Detail in the Session B note.

### 3.8 `vocab_size` — **16000, resolved in both directions**
- Down: `8000` is worse (+0.02 to +0.12) — fewer merges → val titles
  fragment → worse compression → worse char-normalized loss.
- Up: `20k/24k/28k` under the AdamW recipe — **none beat 16000** (+0.008
  to +0.073). Compression keeps rising but `frac_undertrained` (rare-token
  gradient starvation, a pre-training tokenizer property) climbs 0.05→0.16
  and predicts the degradation. The `vocab_precheck.py` diagnostic's
  refined knee (22528) is a red herring *for training*.
- Leaky vs non-leaky tokenizer bases agree within ~1% on both axes.
- Optimizer-independent (Muon never touches the embedding table), so it
  carries to the Muon base — Session B re-checks 16k/20k/24k as a tagged
  check-off for completeness.

### 3.9 `dropout` — unchanged (0.1)
Near-null (underfitting regime — final train loss well above val). `0.05`
is marginally best as a single variable but **hurts in the combo**. Would
matter if the step budget grew.

### 3.10 Model capacity — unchanged (6 / 8 / 512, 27.17M)
Larger (49.7M) and smaller (13.3M) both worse under AdamW. **The run is
budget-limited, not capacity-limited** — 1000 steps / 90k titles can't
exploit more parameters. This is why architecture changes (§3.11) are
expected to be small.

### 3.11 `attn_impl` — kept `manual` (not SDPA)
Numerically equivalent (1.9e-7 in eval). In-combo, SDPA scores +0.0016
worse purely from applying attention dropout on a different RNG stream
than the standalone `attn_drop` module. Identical at `dropout=0`. No
reason to switch.

### 3.13 SwiGLU MLP / QK-norm — OUT (Session B)
Both tested on the Muon base, param-matched where relevant. **SwiGLU
+0.006, QK-norm +0.010** — both worse. The model is budget-limited
(§3.10): a param-matched architecture change can only help if the current
architecture is the bottleneck, and it isn't. QK-norm additionally peaked
early then drifted. `muon_ns_steps` 5→6: +0.0009 (5 is enough).

### 3.12 `use_amp` / `grad_clip_norm` — exposed, off / unswept
bf16 autocast is loss-neutral (+0.0008) and breaks exact determinism —
kept off. `grad_clip_norm` promoted from a hard-coded `1.0` to a field;
not swept.

---

## 4. Instrumentation choices

- **Logging is additive only** — no existing event/field removed, so old
  and new logs parse through one schema and the committed `baseline.log`
  stays a valid reference. New: `dataset_info.max_steps` +
  `avg_tokens_per_val_title`, `validation_step.train_loss`,
  `optimizer_info`, `training_summary` (`best/final_val_loss`, +
  `raw_final_val_loss` when EMA is on), `ema_info`,
  `tokenizer_overlap_info`.
- **`training_summary.best_val_loss` is best-of-eval-snapshots** (~22
  points/run), not per-step — §4.10 is metrics-only, no weight saving.
- **`config.py` is import-safe outside the devcontainer** (no torch/utils
  imports) so its 32 unit tests run on the host. Everything needing torch
  (`test_muon`, `test_ema`, `test_arch`) runs on the pod.
- **Test style:** repo convention — `test_*` functions + a `main()` runner
  + plain `assert`, no pytest (the image has none).
- **The Streamlit dashboard was built (spec §6) then removed.** A
  `tools/dashboard.py` + `logparse.py` + `runlabel.py` (13 tests) — config
  form → `task train` launcher → live chart, plus a run-overlay tab.
  Removed 2026-09-07: the sweeps were driven by pod scripts, so it was
  never used, its history tab was always empty (pod logs are gitignored /
  ephemeral), and a UI that *looks* like a results dashboard but isn't one
  was a liability. The results record is the notes. `git log -- tools/` to
  recover it.

---

## 5. Open / deferred / dropped

- **Open — merge the branches:** `training-experiments` → `phase10-muon-ema`
  (holds Phase 10 + 11) → `main`. Fast-forward, linear, purely local
  (`origin` is Maincode's repo — nothing pushed there). The user's call.
- **Dropped — the Streamlit dashboard:** removed from the repo entirely
  (see §4). Never used; caused confusion.
- **Dropped — split the fused `qkv`** for Muon: low expected value now that
  SwiGLU and QK-norm both hurt.

**The experiment surface is exhausted.** Session B found no lever left in
the AdamW-group hyperparameters or the cheap architecture tweaks. Every
remaining idea attacks the fixed step/data budget, out of scope here.
