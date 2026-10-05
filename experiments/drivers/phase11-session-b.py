# Phase 11 Session B: AdamW-group + SwiGLU/QK-norm/wsd-shape + stable_frac + block64 + vocab check-off + EMA re-tune -> notes/2026-09-07-phase11-session-b-results.md
# Exact orchestration script as run on a RunPod Secure RTX 4090, 2026-09-07.
# Paths are pod-absolute (/root/mainrun); not runnable outside that setup —
# kept as a record of exactly what was executed.

#!/usr/bin/env python3
"""Phase 11 (Session B) driver. Runs on the pod in /root/mainrun/mainrun.
Gate -> Part A (AdamW-group lr/wd/warmup/betas, sequential) -> Part B (arch
toggles, stable_frac, block64, vocab check-off) -> Part C (stack winners,
EMA re-tune, leave-one-out). Writes /root/outB/{tag}.log + results.json.
"""
import json, os, subprocess, time, pathlib

REPO = "/root/mainrun/mainrun"
OUT = pathlib.Path("/root/outB"); OUT.mkdir(exist_ok=True)
os.chdir(REPO)
RESULTS = []

BASELINE = 1.7519307076847304
ADAMW_RECIPE = {"optimizer": "adamw", "lr": 0.0012, "weight_decay": 0.01,
                "scaled_residual_init": True, "shuffle_batches": True,
                "lr_schedule": "wsd"}
ADAMW_RECIPE_VAL = 1.1920606836240772
# Muon base (Phase 10 winner minus EMA); reference re-measured on the pod (RM_val)
RM = {"optimizer": "muon", "muon_lr": 0.02, "lr": 0.0012, "weight_decay": 0.01,
      "shuffle_batches": True, "lr_schedule": "wsd"}


def run(tag, cfg):
    env = dict(os.environ)
    if cfg is None:
        env.pop("MAINRUN_CONFIG", None)
    else:
        p = f"/root/cfgB_{tag}.json"
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
    rec = dict(tag=tag, rc=r.returncode, sec=dt, final=ts.get("final_val_loss"),
               best=ts.get("best_val_loss"), raw=ts.get("raw_final_val_loss"),
               max_steps=di.get("max_steps"), vocab=di.get("vocab_size"),
               oi=oi, cfg=cfg)
    if r.returncode != 0:
        rec["stderr_tail"] = r.stderr[-1000:]
    RESULTS.append(rec)
    subprocess.run(["cp", f"{REPO}/logs/mainrun.log", str(OUT / f"{tag}.log")])
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    print(f"{tag:20s} rc={r.returncode} {dt:6.1f}s  final={rec['final']}  "
          f"best={rec['best']}  raw={rec['raw']}  steps={rec['max_steps']}",
          flush=True)
    return rec


def best_of(prefix, fallback_cfg, fallback_final):
    xs = [r for r in RESULTS if r["tag"].startswith(prefix) and r["rc"] == 0
          and r["final"] is not None]
    if not xs:
        return fallback_cfg, fallback_final
    b = min(xs, key=lambda r: r["final"])
    return b["cfg"], b["final"]


# ================= GATE =================
print("=== GATE ===", flush=True)
g0 = run("gate_baseline", None)
assert abs(g0["final"] - BASELINE) < 1e-12, f"baseline gate FAILED: {g0['final']}"
g1 = run("gate_adamw_recipe", ADAMW_RECIPE)
assert abs(g1["final"] - ADAMW_RECIPE_VAL) < 1e-12, f"adamw recipe gate FAILED: {g1['final']}"
m1 = run("gate_RM", RM)
m2 = run("gate_RM_rerun", RM)
assert m1["final"] == m2["final"], f"Muon base not deterministic: {m1['final']} vs {m2['final']}"
RM_VAL = m1["final"]
print(f"  Muon base reference RM_VAL = {RM_VAL}  "
      f"(Phase 10 pre-fix was 1.17187; A1 param-grouping fix moved it)", flush=True)

# ================= PART A =================
print("=== PART A: AdamW-group lr ===", flush=True)
for lr in (0.0006, 0.0009, 0.0012, 0.0018, 0.0024):
    run(f"A1_lr{lr}", {**RM, "lr": lr})
cA1, fA1 = best_of("A1_lr", RM, RM_VAL)
print(f"  A1 winner lr={cA1.get('lr')} -> {fA1}", flush=True)

print("=== PART A: weight_decay ===", flush=True)
for wd in (0.0, 0.005, 0.01, 0.02, 0.05):
    run(f"A2_wd{wd}", {**cA1, "weight_decay": wd})
run("A2_noembed", {**cA1, "weight_decay_embeddings": False})
cA2, fA2 = best_of("A2_", cA1, fA1)
print(f"  A2 winner -> {fA2}  ({ {k: v for k, v in cA2.items() if k not in RM or RM[k] != v} })", flush=True)

print("=== PART A: warmup_frac ===", flush=True)
for wf in (0.02, 0.04, 0.06, 0.10, 0.15):
    run(f"A3_wf{wf}", {**cA2, "warmup_frac": wf})
cA3, fA3 = best_of("A3_wf", cA2, fA2)
print(f"  A3 winner warmup_frac={cA3.get('warmup_frac', 0.06)} -> {fA3}", flush=True)

print("=== PART A: betas ===", flush=True)
for b1, b2 in ((0.9, 0.95), (0.9, 0.99), (0.95, 0.95)):
    run(f"A4_b{b1}x{b2}", {**cA3, "beta1": b1, "beta2": b2})
cA4, fA4 = best_of("A4_b", cA3, fA3)
R_A = cA4
run("A_final", R_A)
ra = [r for r in RESULTS if r["tag"] == "A_final"][0]
RA_VAL = ra["final"] if ra["final"] is not None else fA4
print(f"  R_A = { {k: v for k, v in R_A.items() if k not in RM or RM[k] != v} }  -> {RA_VAL}", flush=True)

# ================= PART B =================
print("=== PART B: single toggles ===", flush=True)
run("B_swiglu", {**R_A, "mlp_kind": "swiglu"})
run("B_qknorm", {**R_A, "qk_norm": True})
run("B_invsqrt", {**R_A, "wsd_decay_shape": "invsqrt"})
run("B_ns6", {**R_A, "muon_ns_steps": 6})

print("=== PART B: stable_frac ===", flush=True)
for sf in (0.4, 0.5, 0.6, 0.7, 0.8):
    run(f"B_sf{sf}", {**R_A, "stable_frac": sf})

print("=== PART B: block64/batch128 (product held) ===", flush=True)
run("B_block64", {**R_A, "block_size": 64, "batch_size": 128})

print("=== PART B: VOCAB CHECK-OFF (expect 16000 wins) ===", flush=True)
for v in (16000, 20000, 24000):
    run(f"B_vocab{v}", {**R_A, "vocab_size": v})

# ================= PART C =================
print("=== PART C: assemble winners ===", flush=True)
winners = {}


def _beat(tag):
    xs = [r for r in RESULTS if r["tag"] == tag]
    return xs and xs[0]["rc"] == 0 and xs[0]["final"] is not None \
        and xs[0]["final"] < RA_VAL - 0.001


if _beat("B_swiglu"):
    winners["mlp_kind"] = "swiglu"
if _beat("B_qknorm"):
    winners["qk_norm"] = True
if _beat("B_invsqrt"):
    winners["wsd_decay_shape"] = "invsqrt"
if _beat("B_ns6"):
    winners["muon_ns_steps"] = 6
if _beat("B_block64"):
    winners["block_size"] = 64
    winners["batch_size"] = 128
sf_runs = [r for r in RESULTS if r["tag"].startswith("B_sf") and r["rc"] == 0
           and r["final"] is not None]
if sf_runs:
    bsf = min(sf_runs, key=lambda r: r["final"])
    if bsf["final"] < RA_VAL - 0.001 and bsf["cfg"].get("stable_frac", 0.6) != 0.6:
        winners["stable_frac"] = bsf["cfg"]["stable_frac"]
print(f"  Part B winners: {winners}", flush=True)

R_FINAL = {**R_A, **winners}
run("C_final_noema", R_FINAL)

print("=== PART C: EMA re-tune ===", flush=True)
for d in (0.99, 0.995, 0.997, 0.998):
    run(f"C_ema{d}", {**R_FINAL, "ema_decay": d})
ema_runs = [r for r in RESULTS if r["tag"].startswith("C_ema") and r["rc"] == 0
            and r["final"] is not None]
best_ema = min(ema_runs, key=lambda r: r["final"]) if ema_runs else None
R_FINAL_EMA = {**R_FINAL, "ema_decay": best_ema["cfg"]["ema_decay"]} if best_ema else R_FINAL

print("=== PART C: leave-one-out on Part B winners ===", flush=True)
for k in list(winners):
    loo = {kk: vv for kk, vv in R_FINAL_EMA.items() if kk != k}
    if k == "batch_size":
        continue
    if k == "block_size":
        loo.pop("batch_size", None)
    run(f"C_loo_no_{k}", loo)

print("=== PART C: final confirm x2 ===", flush=True)
run("C_confirm_a", R_FINAL_EMA)
run("C_confirm_b", R_FINAL_EMA)

# ================= SUMMARY =================
print("\n=== ALL RESULTS (by final val) ===", flush=True)
for r in sorted(RESULTS, key=lambda r: (r["final"] is None, r["final"] or 9)):
    mark = "  <-- CHECK-OFF" if r["tag"].startswith("B_vocab") else ""
    print(f"{r['tag']:20s} final={r['final']}  raw={r['raw']}  "
          f"steps={r['max_steps']}{mark}", flush=True)
print(f"\nRM_VAL={RM_VAL}  RA_VAL={RA_VAL}", flush=True)
print(f"R_FINAL_EMA = {json.dumps(R_FINAL_EMA)}", flush=True)
print("\nSESSION_B_DONE", flush=True)
