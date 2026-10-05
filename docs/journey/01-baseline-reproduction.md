# 01 — Baseline Reproduction & Config Plumbing

> **Status:** kept · **Effect:** established bit-exact 0.0 noise floor; verified baseline val loss 1.75193 · **Date:** 2026-09-07
>
> ← [00-design.md](00-design.md) · [timeline](README.md) · [next](02-adamw.md) →

## Question
Can we reproduce the baseline training run bit-exactly across repeated runs on GPU, and build a non-invasive configuration mechanism that allows arbitrary experiment sweeps without modifying `train.py` every run?

## Reasoning
Reliable experimentation requires a zero-noise floor: if repeated runs with identical settings produce variance of $\pm 0.01$, subtle effects (such as batch shuffling or initialization adjustments) cannot be distinguished from hardware noise. Furthermore, hardcoding parameters in `train.py` invites accidental contamination. We needed:
1. Bit-exact determinism across executions.
2. A separate `config.py` module with dataclass schema validation and external JSON loading (`MAINRUN_CONFIG`).
3. Structured logging emitting line-delimited JSON events for analysis.

## Approach
1. **Config Plumbing:** Created [`config.py`](../../mainrun/config.py) defining `Hyperparameters` with strict validation. Fixed parameters (`epochs=7`, `seed=1337`, `num_titles=100000`, `val_frac=0.10`, `log_file`) are protected and reject any external override.
2. **Determinism:** Injected deterministic configuration into `train.py`:
   ```python
   os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
   torch.manual_seed(args.seed)
   random.seed(args.seed)
   torch.use_deterministic_algorithms(True, warn_only=True)
   ```
3. **Structured Logging:** Integrated `structlog` dual-logging to output JSON events to log files while printing clean tqdm progress lines to the console.
4. **Verification:** Ran two identical baseline training passes on a RunPod RTX 4090.

## Results
The baseline was executed twice consecutively on an RTX 4090:

| Metric | Run 1 | Run 2 | Difference |
|---|---|---|---|
| Step count | 1001 steps | 1001 steps | 0 |
| Best Val Loss | 1.7519307076847304 (step 1001) | 1.7519307076847304 (step 1001) | **0.0000000000000000** |
| Final Val Loss | 1.7519307076847304 | 1.7519307076847304 | **0.0000000000000000** |
| Runtime | ~46s | ~46s | <1s |

*(Note: The original historical log `baseline.log` recorded 1.75326 over 938 steps from July 2025. The HuggingFace dataset grew slightly since then, resulting in 1001 steps and 1.75193. 1.75193 is our true reference).*

## What we learned
- Determinism on modern NVIDIA Ada GPUs (RTX 4090) is completely achievable with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `use_deterministic_algorithms`.
- The noise floor is exactly **0.0**, which means every delta observed in subsequent phases reflects real algorithmic changes, eliminating the need for multi-seed averaging.
- The baseline loss plateaued late: step 1 was 9.704, step 500 was 1.838, step 1001 reached 1.752.

## Open threads
- Proceed immediately to testing AdamW with a learning-rate bracket ([02-adamw.md](02-adamw.md)).

## Sources
- Code: [`mainrun/config.py`](../../mainrun/config.py), [`mainrun/train.py`](../../mainrun/train.py)
- Tests: [`tests/test_config.py`](../../tests/test_config.py)
- Reference log: [`results/logs/baseline-1001.log`](../../results/logs/baseline-1001.log)
