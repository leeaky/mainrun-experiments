# §4.8 upward sweep 16k/20k/24k/28k -> notes/2026-09-07-exp-4.8-vocab-upsweep.md
# Exact orchestration script as run on a RunPod Secure RTX 4090, 2026-09-07.
# Paths are pod-absolute (/root/mainrun); not runnable outside that setup —
# kept as a record of exactly what was executed.

#!/usr/bin/env bash
# vocab_size upward sweep under the final AdamW recipe.
# Runs on the pod. Expects /root/stage/ to contain: config.py train.py test_config.py
# cfg_16000.json cfg_20000.json cfg_24000.json cfg_28000.json
set -uo pipefail
exec > /root/sweep.log 2>&1
echo "SWEEP_START $(date -u +%FT%TZ)"

# ---- clone fresh ----
rm -rf /root/mainrun
git clone --depth 1 https://github.com/MaincodeHQ/mainrun.git /root/mainrun
git -C /root/mainrun config user.email pod@local
git -C /root/mainrun config user.name pod

# ---- overlay branch code ----
cp /root/stage/config.py       /root/mainrun/mainrun/config.py
cp /root/stage/train.py        /root/mainrun/mainrun/train.py
cp /root/stage/test_config.py  /root/mainrun/mainrun/test_config.py
cp /root/stage/cfg_*.json      /root/mainrun/mainrun/
mkdir -p /root/mainrun/mainrun/sweep_out

cd /root/mainrun/mainrun

echo "=== CUDA ==="
python3 -c "import torch;print('cuda_available',torch.cuda.is_available());print('dev',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "=== config unit tests ==="
python3 test_config.py 2>&1 | tail -5

run () {
  local tag="$1" cfg="$2"
  echo "=== RUN ${tag} $(date -u +%TZ) ==="
  if [ -z "$cfg" ]; then unset MAINRUN_CONFIG; else export MAINRUN_CONFIG="$cfg"; fi
  task train > "sweep_out/${tag}.stdout" 2>&1
  local rc=$?
  cp logs/mainrun.log "sweep_out/${tag}.log" 2>/dev/null || echo "NO LOG for ${tag}"
  echo "${tag} rc=${rc}"
  python3 - "sweep_out/${tag}.log" <<'PY'
import json,sys
p=sys.argv[1]
ev={}
try:
    for ln in open(p):
        ln=ln.strip()
        if not ln: continue
        try: d=json.loads(ln)
        except Exception: continue
        e=d.get("event") or d.get("_event") or d.get("type")
        ev.setdefault(e,[]).append(d)
except FileNotFoundError:
    print("  (no log)"); sys.exit()
def last(name,*keys):
    xs=ev.get(name)
    if not xs: return
    d=xs[-1]
    print("  %s: %s"%(name,{k:d.get(k) for k in keys} if keys else d))
last("dataset_info","vocab_size","max_steps","batches_per_epoch","tokens_per_epoch","avg_tokens_per_val_title")
last("model_info","parameters_count")
last("optimizer_info")
last("training_summary","best_val_loss","best_val_step","final_val_loss")
vs=ev.get("validation_step") or []
if vs:
    d=vs[-1]
    print("  last validation_step: step=%s loss=%s train_loss=%s"%(d.get("step"),d.get("loss"),d.get("train_loss")))
print("  n validation_step events: %d"%len(vs))
PY
}

run gate ""
run vocab16000 /root/mainrun/mainrun/cfg_16000.json
run vocab20000 /root/mainrun/mainrun/cfg_20000.json
run vocab24000 /root/mainrun/mainrun/cfg_24000.json
run vocab28000 /root/mainrun/mainrun/cfg_28000.json

echo "=== tokenizer-only measure (leaky basis = train_and_val, matches training) ==="
python3 - <<'PY'
import sys, types
from pathlib import Path
sys.path.insert(0, "/root/mainrun/mainrun")
if not Path("/root/.mainrun").exists():
    sys.modules.setdefault("utils", types.ModuleType("utils"))
from train import get_titles, train_tokenizer, BPETokenizer
import numpy as np
tr, va = get_titles(100_000, 1337, 0.10)
EOS="<eos>"
train_text = EOS.join(tr)+EOS
val_text   = EOS.join(va)+EOS
corpus = tr+va                       # train_and_val — same as training default
print(f"{'vocab':>7} {'actual':>7} {'val_cpt':>9} {'undertrn<5x':>12} {'frac':>7}")
for v in (16000,20000,24000,28000):
    tok = BPETokenizer(train_tokenizer(corpus, v, eos_token=EOS))
    act = int(tok.vocab_size)
    vids = tok.encode(val_text)
    cpt = len(val_text)/len(vids)
    tids = np.asarray(tok.encode(train_text), dtype=np.int64)
    cnt = np.bincount(tids, minlength=act)[:act]
    nu = int((cnt<5).sum())
    print(f"{v:>7} {act:>7} {cpt:>9.4f} {nu:>12} {nu/act:>7.4f}")
PY

echo "SWEEP_DONE $(date -u +%FT%TZ)"
