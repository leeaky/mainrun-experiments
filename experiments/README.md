# experiments/

The scripts that actually ran the ~190 training runs behind
[`docs/superpowers/`](../../docs/superpowers/). Kept as a record of *how*
each sweep was executed — the results live in the notes, the reproducible
end-state lives in [`../logs/`](../logs/).

## What ran what

| driver | phase | what it swept | results note |
|---|---|---|---|
| `drivers/phase4-config-sweeps.sh` | §4.3–4.9 | `block_size`, `batch_size`, `dropout`, `vocab_size=8000`, capacity (config-only, no code) | `phase4-config-sweeps-results.md` |
| `drivers/phase78-toggles.sh` | §4.11/4.12 | SDPA equivalence, amp / grad_clip / wsd, combo consolidation | `phase78-toggles-and-combo.md` |
| `drivers/vocab-upsweep.sh` | §4.8 follow-up | `vocab_size` 16k/20k/24k/28k upward | `exp-4.8-vocab-upsweep.md` |
| `drivers/phase10-session-a.py` | 10 | Muon `muon_lr × adamw_lr` grid, EMA sweep, stacking leave-one-out | `phase10-session-a-results.md` |
| `drivers/phase10-session-a-followup.py` | 10 | corrected EMA sweep (plain decay), `muon_lr` refine | ″ |
| `drivers/phase10-session-a-ema-bracket.py` | 10 | EMA decay 0.997–0.999 bracket + determinism reruns | ″ |
| `drivers/phase11-session-b.py` | 11 | AdamW-group `lr`/`wd`/`warmup`/betas → SwiGLU / QK-norm / 1−√ decay / `ns_steps` / `stable_frac` / `block64` / vocab check-off → EMA re-tune | `phase11-session-b-results.md` |
| `drivers/small-batch.py` | 11 follow-up | `batch_size` 16/24/32 on the Muon+EMA recipe, coverage-adjusted | ″ (appended) |
| `drivers/reference-logs.py` | — | regenerate `../logs/{baseline-1001,recipe-adamw,recipe-muon-ema}.log` | `../logs/README.md` |

Phases 1–3, 5–6 (config plumbing, AdamW, scaled residual init, tokenizer
scope, batch shuffling) were run as individual `task train` invocations
over SSH, not batch drivers — see their plans/notes.

## How they were run

Each driver was `scp`'d onto a fresh RunPod **Secure Cloud RTX 4090** pod
(template `9rwvm97wd9`, image `leeaky/mainrun-devcontainer:latest`)
alongside the current `config.py` / `train.py` / `muon.py` / `ema.py`, then
run with `nohup python3 -u <driver>`. Every run is a `task train` subprocess
with a `MAINRUN_CONFIG` JSON. Paths inside the drivers are pod-absolute
(`/root/mainrun/mainrun`) — **they are records, not portable scripts.**

Determinism: `CUBLAS_WORKSPACE_CONFIG=:4096:8` +
`torch.use_deterministic_algorithms(True, warn_only=True)` → 0.0 noise
floor, so one run per config.

## Also here

- `analyze.py` — dump `training_summary` / `dataset_info` / key metrics
  from one or more `mainrun` log files. `python3 analyze.py '../logs/*.log'`
- `../vocab_precheck.py` (+ `../test_vocab_precheck.py`) — the tokenizer-only
  `vocab_size` diagnostic (compression knee + rare-token `frac_undertrained`),
  built before the §4.8 sweep. Runs on CPU: `python3 vocab_precheck.py`.
