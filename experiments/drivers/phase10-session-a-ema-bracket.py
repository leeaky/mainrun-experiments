# Phase 10: EMA decay 0.997-0.999 bracket + determinism reruns (same note)
# Exact orchestration script as run on a RunPod Secure RTX 4090, 2026-09-07.
# Paths are pod-absolute (/root/mainrun); not runnable outside that setup —
# kept as a record of exactly what was executed.

#!/usr/bin/env python3
"""Phase 10 Session A final: bracket the EMA decay optimum above 0.995 for
both bases, and confirm the Muon+EMA winner is bit-exact. Runs on the pod."""
import json, os, subprocess, time, pathlib

REPO = "/root/mainrun/mainrun"
OUT = pathlib.Path("/root/out3"); OUT.mkdir(exist_ok=True)
os.chdir(REPO)
RESULTS = []

RECIPE = {"optimizer": "adamw", "lr": 0.0012, "weight_decay": 0.01,
          "scaled_residual_init": True, "shuffle_batches": True,
          "lr_schedule": "wsd"}
MUON = {"optimizer": "muon", "muon_lr": 0.02, "lr": 0.0012,
        "weight_decay": 0.01, "shuffle_batches": True, "lr_schedule": "wsd"}


def run(tag, cfg):
    env = dict(os.environ)
    p = f"/root/cfg3_{tag}.json"
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
    rec = dict(tag=tag, rc=r.returncode, sec=dt, final=ts.get("final_val_loss"),
               best=ts.get("best_val_loss"), raw=ts.get("raw_final_val_loss"),
               cfg=cfg)
    RESULTS.append(rec)
    subprocess.run(["cp", f"{REPO}/logs/mainrun.log", str(OUT / f"{tag}.log")])
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    print(f"{tag:22s} rc={r.returncode} {dt:6.1f}s  final={rec['final']}  "
          f"best={rec['best']}  raw={rec['raw']}", flush=True)
    return rec


for d in (0.997, 0.998, 0.999):
    run(f"recipe_ema{d}", {**RECIPE, "ema_decay": d})
for d in (0.995, 0.997, 0.998):
    run(f"muon_ema{d}", {**MUON, "ema_decay": d})
run("muon_ema0.995_rerun", {**MUON, "ema_decay": 0.995})

print("\n=== ALL (by final) ===", flush=True)
for r in sorted(RESULTS, key=lambda r: (r["final"] is None, r["final"] or 9)):
    print(f"{r['tag']:22s} final={r['final']}  best={r['best']}  raw={r['raw']}",
          flush=True)
print("\nSESSION_A3_DONE", flush=True)
