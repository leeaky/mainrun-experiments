# Reference logs

Full structured `logger.log` traces (one JSON object per line) for the
anchor configs. All except `baseline.log` were generated on 2026-09-07 from
this repo's `main`, on a RunPod Secure RTX 4090, and are **bit-exact
reproducible** (`CUBLAS_WORKSPACE_CONFIG=:4096:8` +
`torch.use_deterministic_algorithms`).

Everything matching `mainrun*.log` is gitignored (that's the live run's
output + `checkpoint.mjs`'s timestamped archives); these are named to
escape that so they can be committed.

| file | config | final val | steps | notes |
|---|---|---|---|---|
| `baseline.log` | shipped default (SGD, `lr 6e-3`) | 1.75326 | 938 | the original reference — July 2025, ~40 min CPU run, before the HF dataset grew |
| `baseline-1001.log` | same, current code + grown dataset | **1.7519307076847304** | 1001 | the real comparison point for every recipe number below |
| `recipe-adamw.log` | `{optimizer:adamw, lr:0.0012, weight_decay:0.01, scaled_residual_init:true, shuffle_batches:true, lr_schedule:wsd}` | **1.1920606836240772** | 994 | Phases 1–9 recipe (AdamW, full model) |
| `recipe-muon-ema.log` | `{optimizer:muon, muon_lr:0.02, lr:0.0012, weight_decay:0.01, shuffle_batches:true, lr_schedule:wsd, ema_decay:0.997}` | **1.1656583434249248** | 994 | the final recipe (Phase 10 + 11). `training_summary.raw_final_val_loss` = 1.17184 is the same run without the EMA weight-swap at eval. |

`recipe-muon-ema.log` final is `1.16566`, not the `1.16539` first reported
in Phase 10 — the Session B fix that stopped weight-decaying LayerNorm
gains/biases on the Muon branch shifted it +0.00027 (more correct,
marginally worse on this metric). `1.16566` is what current `main`
reproduces.

To reproduce any of these:
```bash
echo '<config json>' > mainrun/cfg.json
MAINRUN_CONFIG=mainrun/cfg.json task train   # writes mainrun/logs/mainrun.log
```
