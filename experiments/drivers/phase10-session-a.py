# Phase 10: Muon LR grid + EMA sweep + stacking -> notes/2026-09-07-phase10-session-a-results.md
# Exact orchestration script as run on a RunPod Secure RTX 4090, 2026-09-07.
# Paths are pod-absolute (/root/mainrun); not runnable outside that setup —
# kept as a record of exactly what was executed.

#!/usr/bin/env python3
"""Phase 10 Session A driver. Runs on the pod in /root/mainrun/mainrun.
Gate -> Muon LR grid -> momentum + stacking at the winner -> EMA sweep ->
EMA on the overall best. Writes /root/out/{tag}.log + /root/out/results.json.
"""
import json, os, subprocess, time, pathlib

REPO = "/root/mainrun/mainrun"
OUT = pathlib.Path("/root/out"); OUT.mkdir(exist_ok=True)
os.chdir(REPO)
RESULTS = []

RECIPE = {"optimizer": "adamw", "lr": 0.0012, "weight_decay": 0.01,
          "scaled_residual_init": True, "shuffle_batches": True,
          "lr_schedule": "wsd"}
BASELINE_VAL = 1.7519307076847304
RECIPE_VAL = 1.1920606836240772


def run(tag, cfg):
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
    ev = {}
    try:
        for ln in open(f"{REPO}/logs/mainrun.log"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            ev.setdefault(d.get("event"), []).append(d)
    except FileNotFoundError:
        pass
    ts = (ev.get("training_summary") or [{}])[-1]
    di = (ev.get("dataset_info") or [{}])[-1]
    oi = (ev.get("optimizer_info") or [{}])[-1]
    rec = dict(tag=tag, rc=r.returncode, sec=dt,
               final=ts.get("final_val_loss"), best=ts.get("best_val_loss"),
               raw=ts.get("raw_final_val_loss"),
               best_step=ts.get("best_val_step"), max_steps=di.get("max_steps"),
               params=di.get("vocab_size"), oi=oi, cfg=cfg)
    if r.returncode != 0:
        rec["stderr_tail"] = r.stderr[-800:]
    RESULTS.append(rec)
    subprocess.run(["cp", f"{REPO}/logs/mainrun.log", str(OUT / f"{tag}.log")])
    _dump()
    print(f"{tag:26s} rc={r.returncode} {dt:6.1f}s  final={rec['final']}  "
          f"best={rec['best']}  raw={rec['raw']}  steps={rec['max_steps']}",
          flush=True)
    return rec


def _dump():
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))


# ---------------- gate ----------------
print("=== GATE ===", flush=True)
g0 = run("gate_baseline", None)
assert g0["final"] is not None and abs(g0["final"] - BASELINE_VAL) < 1e-12, \
    f"baseline gate FAILED: {g0['final']!r}"
g1 = run("gate_recipe", RECIPE)
assert g1["final"] is not None and abs(g1["final"] - RECIPE_VAL) < 1e-12, \
    f"recipe gate FAILED: {g1['final']!r}"
g2 = run("gate_recipe_rerun", RECIPE)
assert g2["final"] == g1["final"], f"recipe not deterministic: {g1['final']} vs {g2['final']}"
gm1 = run("gate_muon_det_a", {**RECIPE, "optimizer": "muon", "muon_lr": 0.02})
gm2 = run("gate_muon_det_b", {**RECIPE, "optimizer": "muon", "muon_lr": 0.02})
if gm1["final"] != gm2["final"]:
    print(f"  WARNING: muon not bit-exact: {gm1['final']} vs {gm2['final']}", flush=True)
else:
    print(f"  muon deterministic: {gm1['final']}", flush=True)

# ---------------- Muon LR grid ----------------
print("=== MUON GRID ===", flush=True)
for mlr in (0.01, 0.02, 0.03, 0.05, 0.08):
    for alr in (0.0012, 0.004):
        run(f"muon_m{mlr}_a{alr}",
            {**RECIPE, "optimizer": "muon", "muon_lr": mlr, "lr": alr})
run("muon_bare_m0.03", {"optimizer": "muon", "muon_lr": 0.03})

muon_grid = [r for r in RESULTS if r["tag"].startswith("muon_m") and r["rc"] == 0
             and r["final"] is not None]
best_m = min(muon_grid, key=lambda r: r["final"])
BM_LR = best_m["cfg"]["muon_lr"]
BA_LR = best_m["cfg"]["lr"]
BEST_MUON = {**RECIPE, "optimizer": "muon", "muon_lr": BM_LR, "lr": BA_LR}
print(f"  best muon grid: {best_m['tag']} final={best_m['final']} "
      f"(muon_lr={BM_LR}, adamw_lr={BA_LR})", flush=True)

# ---------------- momentum + stacking at the winner ----------------
print("=== MUON MOMENTUM + STACKING ===", flush=True)
for mom in (0.9, 0.98):
    run(f"muon_mom{mom}", {**BEST_MUON, "muon_momentum": mom})
no_sri = {k: v for k, v in BEST_MUON.items() if k != "scaled_residual_init"}
run("muon_no_sri", no_sri)
no_shuf = {k: v for k, v in BEST_MUON.items() if k != "shuffle_batches"}
run("muon_no_shuffle", no_shuf)
run("muon_cosine", {**BEST_MUON, "lr_schedule": "cosine"})

# ---------------- EMA sweep on the AdamW recipe ----------------
print("=== EMA SWEEP ===", flush=True)
for d in (0.99, 0.995, 0.999, 0.9995):
    run(f"ema{d}", {**RECIPE, "ema_decay": d})
ema_runs = [r for r in RESULTS if r["tag"].startswith("ema0.") and r["rc"] == 0
            and r["final"] is not None]
best_ema = min(ema_runs, key=lambda r: r["final"])
BED = best_ema["cfg"]["ema_decay"]
print(f"  best ema on recipe: decay={BED} final={best_ema['final']} "
      f"(raw {best_ema['raw']})", flush=True)

# ---------------- EMA on the overall best optimizer ----------------
print("=== EMA ON BEST ===", flush=True)
overall_best = BEST_MUON if best_m["final"] < RECIPE_VAL else RECIPE
run("ema_on_best", {**overall_best, "ema_decay": BED})
if best_m["final"] < RECIPE_VAL:
    # also the second-best ema decay on muon, cheap insurance
    others = sorted(ema_runs, key=lambda r: r["final"])
    if len(others) > 1:
        run("ema_on_best_2nd", {**BEST_MUON, "ema_decay": others[1]["cfg"]["ema_decay"]})

# ---------------- summary ----------------
print("\n=== ALL RESULTS (by final val) ===", flush=True)
for r in sorted(RESULTS, key=lambda r: (r["final"] is None, r["final"] or 9)):
    print(f"{r['tag']:26s} final={r['final']}  best={r['best']}  "
          f"raw={r['raw']}  steps={r['max_steps']}  {r['sec']}s", flush=True)
_dump()
print("\nSESSION_A_DONE", flush=True)
