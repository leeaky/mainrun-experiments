# §4.11 SDPA / §4.12 amp+grad_clip+wsd + combo -> notes/2026-09-07-phase78-toggles-and-combo.md
# Exact orchestration script as run on a RunPod Secure RTX 4090, 2026-09-07.
# Paths are pod-absolute (/root/mainrun); not runnable outside that setup —
# kept as a record of exactly what was executed.

#!/bin/bash
cd /root/mainrun
run() { printf '%s' "$2" > /tmp/c.json; MAINRUN_CONFIG=/tmp/c.json task train > "/root/p78_$1.out" 2>&1 \
  && cp mainrun/logs/mainrun.log "/root/p78_$1.log" && echo "OK $1" || echo "FAIL $1"; }

# gate: defaults unchanged
unset MAINRUN_CONFIG && task train > /root/p78_gate.out 2>&1 && cp mainrun/logs/mainrun.log /root/p78_gate.log \
  && echo "gate $(python3 -c "import json;print([json.loads(l) for l in open(chr(47)+'root/mainrun/mainrun/logs/mainrun.log') if 'validation_step' in l][-1]['loss'])")"

B='"optimizer":"adamw","lr":0.0012,"weight_decay":0.01,"scaled_residual_init":true,"shuffle_batches":true'

run recipe          "{$B,\"run_label\":\"adamw+sri+shuffle\"}"
run recipe_drop05   "{$B,\"dropout\":0.05,\"run_label\":\"recipe+drop05\"}"
run recipe_wsd      "{$B,\"lr_schedule\":\"wsd\",\"run_label\":\"recipe+wsd\"}"
run recipe_sdpa     "{$B,\"attn_impl\":\"sdpa\",\"run_label\":\"recipe+sdpa\"}"
run recipe_block64  "{$B,\"block_size\":64,\"run_label\":\"recipe+block64\"}"
run recipe_block32  "{$B,\"block_size\":32,\"run_label\":\"recipe+block32\"}"
run recipe_amp      "{$B,\"use_amp\":true,\"run_label\":\"recipe+amp\"}"
echo P78_DONE
