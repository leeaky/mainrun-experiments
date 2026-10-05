# regenerate mainrun/logs/{baseline-1001,recipe-adamw,recipe-muon-ema}.log on current main
# Exact orchestration script as run on a RunPod Secure RTX 4090, 2026-09-07.
# Paths are pod-absolute (/root/mainrun); not runnable outside that setup —
# kept as a record of exactly what was executed.

#!/usr/bin/env python3
"""Generate canonical reference logs on current `main` code. Runs on the pod.
Produces /root/out/{baseline-1001,recipe-adamw,recipe-muon-ema}.log — full
structured mainrun traces of the three anchor configs — plus a determinism
re-run of the winner."""
import json, os, subprocess, time, pathlib

REPO = "/root/mainrun/mainrun"
OUT = pathlib.Path("/root/out"); OUT.mkdir(exist_ok=True)
os.chdir(REPO)

CONFIGS = [
    ("baseline-1001",   None,                                   1.7519307076847304),
    ("recipe-adamw",    {"optimizer": "adamw", "lr": 0.0012, "weight_decay": 0.01,
                         "scaled_residual_init": True, "shuffle_batches": True,
                         "lr_schedule": "wsd"},                  1.1920606836240772),
    ("recipe-muon-ema", {"optimizer": "muon", "muon_lr": 0.02, "lr": 0.0012,
                         "weight_decay": 0.01, "shuffle_batches": True,
                         "lr_schedule": "wsd", "ema_decay": 0.997}, None),
    ("_muon-ema-rerun", {"optimizer": "muon", "muon_lr": 0.02, "lr": 0.0012,
                         "weight_decay": 0.01, "shuffle_batches": True,
                         "lr_schedule": "wsd", "ema_decay": 0.997}, None),
]

results = {}
for tag, cfg, expect in CONFIGS:
    env = dict(os.environ)
    if cfg is None:
        env.pop("MAINRUN_CONFIG", None)
    else:
        p = f"/root/cfg_{tag}.json"
        open(p, "w").write(json.dumps(cfg))
        env["MAINRUN_CONFIG"] = p
    t0 = time.time()
    r = subprocess.run(["task", "train"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    dt = round(time.time() - t0, 1)
    ts = {}
    for ln in open(f"{REPO}/logs/mainrun.log"):
        ln = ln.strip()
        if '"training_summary"' in ln:
            ts = json.loads(ln)
    final = ts.get("final_val_loss")
    results[tag] = final
    if not tag.startswith("_"):
        subprocess.run(["cp", f"{REPO}/logs/mainrun.log", str(OUT / f"{tag}.log")])
    flag = ""
    if expect is not None:
        flag = " OK" if final is not None and abs(final - expect) < 1e-12 else f" MISMATCH expected {expect}"
    print(f"{tag:18s} rc={r.returncode} {dt:6.1f}s  final={final}{flag}", flush=True)

det = "OK bit-exact" if results.get("recipe-muon-ema") == results.get("_muon-ema-rerun") else "NONDETERMINISTIC"
print(f"\nmuon-ema determinism: {det}  "
      f"({results.get('recipe-muon-ema')} vs {results.get('_muon-ema-rerun')})", flush=True)
print("\nREF_LOGS_DONE", flush=True)
