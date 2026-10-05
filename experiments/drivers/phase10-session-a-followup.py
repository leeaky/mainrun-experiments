# Phase 10 follow-up: corrected EMA sweep (plain decay) + muon_lr refine (same note)
# Exact orchestration script as run on a RunPod Secure RTX 4090, 2026-09-07.
# Paths are pod-absolute (/root/mainrun); not runnable outside that setup —
# kept as a record of exactly what was executed.

#!/usr/bin/env python3
"""Phase 10 Session A follow-up: corrected EMA sweep (plain decay) + Muon
refinement around muon_lr=0.02 without scaled_residual_init (which slightly
hurt under Muon in Session A). Runs on the pod."""
import json, os, subprocess, time, pathlib

REPO = "/root/mainrun/mainrun"
OUT = pathlib.Path("/root/out2"); OUT.mkdir(exist_ok=True)
os.chdir(REPO)
RESULTS = []

RECIPE = {"optimizer": "adamw", "lr": 0.0012, "weight_decay": 0.01,
          "scaled_residual_init": True, "shuffle_batches": True,
          "lr_schedule": "wsd"}
# Muon base WITHOUT sri (Session A: no_sri beat full by 0.0018)
MUON_NOSRI = {"optimizer": "muon", "lr": 0.0012, "weight_decay": 0.01,
              "shuffle_batches": True, "lr_schedule": "wsd"}


def run(tag, cfg):
    env = dict(os.environ)
    p = f"/root/cfg2_{tag}.json"
    open(p, "w").write(json.dumps(cfg))
    env["MAINRUN_CONFIG"] = p
    t0 = time.time()
    r = subprocess.run(["task", "train"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    dt = round(time.time() - t0, 1)
    ev = {}
    for ln in open(f"{REPO}/logs/mainrun.log"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except Exception:
            continue
        ev.setdefault(d.get("event"), []).append(d)
    ts = (ev.get("training_summary") or [{}])[-1]
    di = (ev.get("dataset_info") or [{}])[-1]
    rec = dict(tag=tag, rc=r.returncode, sec=dt,
               final=ts.get("final_val_loss"), best=ts.get("best_val_loss"),
               raw=ts.get("raw_final_val_loss"), max_steps=di.get("max_steps"),
               cfg=cfg)
    if r.returncode != 0:
        rec["stderr_tail"] = r.stderr[-800:]
    RESULTS.append(rec)
    subprocess.run(["cp", f"{REPO}/logs/mainrun.log", str(OUT / f"{tag}.log")])
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    print(f"{tag:24s} rc={r.returncode} {dt:6.1f}s  final={rec['final']}  "
          f"best={rec['best']}  raw={rec['raw']}  steps={rec['max_steps']}",
          flush=True)
    return rec


print("=== EMA sweep on recipe (plain decay) ===", flush=True)
for d in (0.95, 0.97, 0.98, 0.99, 0.995):
    run(f"ema_{d}", {**RECIPE, "ema_decay": d})

print("=== Muon refine around 0.02, no sri ===", flush=True)
for mlr in (0.015, 0.018, 0.02, 0.022, 0.025):
    run(f"muon_nosri_m{mlr}", {**MUON_NOSRI, "muon_lr": mlr})

# pick winners
ema_runs = [r for r in RESULTS if r["tag"].startswith("ema_") and r["rc"] == 0]
best_ema = min(ema_runs, key=lambda r: r["final"])
BED = best_ema["cfg"]["ema_decay"]
muon_runs = [r for r in RESULTS if r["tag"].startswith("muon_nosri_m") and r["rc"] == 0]
best_muon = min(muon_runs, key=lambda r: r["final"])
BMLR = best_muon["cfg"]["muon_lr"]
print(f"  best ema decay = {BED} ({best_ema['final']}); "
      f"best muon_lr(no sri) = {BMLR} ({best_muon['final']})", flush=True)

print("=== combine: best Muon + EMA ===", flush=True)
for d in sorted({BED, 0.98, 0.99}):
    run(f"muon_best_ema{d}", {**MUON_NOSRI, "muon_lr": BMLR, "ema_decay": d})

print("\n=== ALL (by final) ===", flush=True)
for r in sorted(RESULTS, key=lambda r: (r["final"] is None, r["final"] or 9)):
    print(f"{r['tag']:24s} final={r['final']}  best={r['best']}  raw={r['raw']}  "
          f"steps={r['max_steps']}", flush=True)
print("\nSESSION_A2_DONE", flush=True)
