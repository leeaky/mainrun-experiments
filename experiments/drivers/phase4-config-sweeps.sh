# §4.3/4.4/4.7/4.8/4.9 config-only sweeps -> notes/2026-09-07-phase4-config-sweeps-results.md
# Exact orchestration script as run on a RunPod Secure RTX 4090, 2026-09-07.
# Paths are pod-absolute (/root/mainrun); not runnable outside that setup —
# kept as a record of exactly what was executed.

#!/bin/bash
# Phase 4 config-only sweep matrix. Run on the pod in /root/mainrun.
cd /root/mainrun
run() {  # $1 = run_label, $2 = MAINRUN_CONFIG json (or "" for baseline)
  local label="$1" cfg="$2"
  if [ -z "$cfg" ]; then
    unset MAINRUN_CONFIG
  else
    printf '%s' "$cfg" > /tmp/cfg.json
    export MAINRUN_CONFIG=/tmp/cfg.json
  fi
  if task train > "/root/r_${label}.out" 2>&1; then
    cp mainrun/logs/mainrun.log "/root/r_${label}.log"
    echo "OK   $label"
  else
    echo "FAIL $label (see /root/r_${label}.out)"
  fi
  unset MAINRUN_CONFIG
}

A='"optimizer":"adamw","lr":0.0012,"weight_decay":0.01'

run ref_sgd   ""
run ref_adamw "{$A,\"run_label\":\"adamw\"}"

# 4.3 block_size
run block32       "{\"block_size\":32,\"run_label\":\"block32\"}"
run block64       "{\"block_size\":64,\"run_label\":\"block64\"}"
run adamw_block32 "{$A,\"block_size\":32,\"run_label\":\"adamw+block32\"}"
run adamw_block64 "{$A,\"block_size\":64,\"run_label\":\"adamw+block64\"}"

# 4.4 batch_size
run batch32        "{\"batch_size\":32,\"run_label\":\"batch32\"}"
run batch128       "{\"batch_size\":128,\"run_label\":\"batch128\"}"
run adamw_batch32  "{$A,\"batch_size\":32,\"run_label\":\"adamw+batch32\"}"
run adamw_batch128 "{$A,\"batch_size\":128,\"run_label\":\"adamw+batch128\"}"

# 4.7 dropout
run drop00        "{\"dropout\":0.0,\"run_label\":\"drop0.0\"}"
run drop005       "{\"dropout\":0.05,\"run_label\":\"drop0.05\"}"
run adamw_drop00  "{$A,\"dropout\":0.0,\"run_label\":\"adamw+drop0.0\"}"
run adamw_drop005 "{$A,\"dropout\":0.05,\"run_label\":\"adamw+drop0.05\"}"

# 4.8 vocab_size
run vocab8000       "{\"vocab_size\":8000,\"run_label\":\"vocab8000\"}"
run adamw_vocab8000 "{$A,\"vocab_size\":8000,\"run_label\":\"adamw+vocab8000\"}"

# 4.9 capacity (head_dim = 64 held)
run cap_small       "{\"n_layer\":4,\"n_head\":6,\"d_model\":384,\"run_label\":\"cap-small\"}"
run cap_large       "{\"n_layer\":8,\"n_head\":10,\"d_model\":640,\"run_label\":\"cap-large\"}"
run adamw_cap_small "{$A,\"n_layer\":4,\"n_head\":6,\"d_model\":384,\"run_label\":\"adamw+cap-small\"}"
run adamw_cap_large "{$A,\"n_layer\":8,\"n_head\":10,\"d_model\":640,\"run_label\":\"adamw+cap-large\"}"

echo PHASE4_ALL_DONE
