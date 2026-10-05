# batch_size 16/24/32 on the Muon+EMA recipe, coverage-adjusted (appended to the Phase 11 note)
# Exact orchestration script as run on a RunPod Secure RTX 4090, 2026-09-07.
# Paths are pod-absolute (/root/mainrun); not runnable outside that setup —
# kept as a record of exactly what was executed.

#!/usr/bin/env python3
"""Small-batch sweep on the Muon+EMA headline recipe. Runs on the pod.
Reports raw final val loss AND a coverage-adjusted figure (smaller batch =>
smaller evaluate() span => more of val scored => raw number looks worse for
a reason unrelated to the model). Also logs max_steps (smaller batch => more
optimizer steps at the fixed 7-epoch budget).
"""
import json, os, subprocess, sys, time, types, pathlib
from pathlib import Path

REPO = "/root/mainrun/mainrun"
OUT = pathlib.Path("/root/outSB"); OUT.mkdir(exist_ok=True)
os.chdir(REPO)
RESULTS = []

BASELINE = 1.7519307076847304
RECIPE = {"optimizer": "muon", "muon_lr": 0.02, "lr": 0.0012,
          "weight_decay": 0.01, "shuffle_batches": True, "lr_schedule": "wsd",
          "ema_decay": 0.997}
RECIPE_VAL = 1.16539  # approx (Phase 10; A1 grouping fix ~unchanged)

# --- reconstruct val_ids once for the coverage math ---
sys.path.insert(0, REPO)
if not Path("/root/.mainrun").exists():
    sys.modules.setdefault("utils", types.ModuleType("utils"))
from train import get_titles, train_tokenizer, BPETokenizer
_tr, _va = get_titles(100_000, 1337, 0.10)
_tok = BPETokenizer(train_tokenizer(_tr + _va, 16_000, eos_token="<eos>"))
_val_text = "<eos>".join(_va) + "<eos>"
_val_ids = _tok.encode(_val_text)
NVAL = len(_val_ids)
CHARS = len(_val_text)
print(f"val: {NVAL:,} tokens / {CHARS:,} chars", flush=True)


def coverage(batch_size, block_size=128):
    span = block_size * batch_size + 1
    n_iters = len(range(0, NVAL - span + 1, span))
    scored = n_iters * batch_size * block_size
    return scored / NVAL, scored


def run(tag, cfg, batch_size=64):
    env = dict(os.environ)
    if cfg is None:
        env.pop("MAINRUN_CONFIG", None)
    else:
        p = f"/root/cfgSB_{tag}.json"
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
    final = ts.get("final_val_loss")
    cov, scored = coverage(batch_size)
    adj = final / cov if final else None
    rec = dict(tag=tag, rc=r.returncode, sec=dt, batch_size=batch_size,
               final=final, raw_final=ts.get("raw_final_val_loss"),
               best=ts.get("best_val_loss"), max_steps=di.get("max_steps"),
               coverage=round(cov, 4), adj_final=adj, cfg=cfg)
    if r.returncode != 0:
        rec["stderr_tail"] = r.stderr[-800:]
    RESULTS.append(rec)
    subprocess.run(["cp", f"{REPO}/logs/mainrun.log", str(OUT / f"{tag}.log")])
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    if final is not None:
        print(f"{tag:16s} rc={r.returncode} {dt:6.1f}s  bs={batch_size:3d}  "
              f"steps={rec['max_steps']}  cov={cov:.3f}  final={final:.6f}  "
              f"adj={adj:.6f}", flush=True)
    else:
        print(f"{tag:16s} rc={r.returncode} FAILED\n{rec.get('stderr_tail','')}", flush=True)
    return rec


print("=== GATE ===", flush=True)
g0 = run("gate_baseline", None, 64)
assert abs(g0["final"] - BASELINE) < 1e-12, f"baseline gate FAILED: {g0['final']}"
g1 = run("recipe_bs64", RECIPE, 64)
print(f"  recipe @ bs64: {g1['final']} (expected ~{RECIPE_VAL})", flush=True)

print("=== SMALL BATCH ===", flush=True)
for bs in (32, 24, 16):
    run(f"recipe_bs{bs}", {**RECIPE, "batch_size": bs}, bs)

# also: no-EMA, to isolate (EMA may mask a per-step-noise effect)
print("=== SMALL BATCH, no EMA ===", flush=True)
NOEMA = {k: v for k, v in RECIPE.items() if k != "ema_decay"}
for bs in (64, 32, 16):
    run(f"noema_bs{bs}", {**NOEMA, "batch_size": bs}, bs)

print("\n=== ALL (by adj_final) ===", flush=True)
for r in sorted(RESULTS, key=lambda r: (r["adj_final"] is None, r["adj_final"] or 9)):
    print(f"{r['tag']:16s} bs={r['batch_size']:3d} steps={r['max_steps']:5} "
          f"cov={r['coverage']} raw_final={r['final']} adj={r['adj_final']}",
          flush=True)
print("\nSESSION_SB_DONE", flush=True)
