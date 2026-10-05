"""Plain-assert tests for ema.py. Needs torch -> run on the pod.
    python3 test_ema.py
"""
import torch
import torch.nn as nn

from ema import EMA


def _named(v):
    return [("w", torch.tensor(v))]


def test_decay_zero_is_noop():
    e = EMA(_named([1.0, 1.0]), decay=0.0)
    assert e.shadow == {}
    e.update(_named([0.0, 0.0]), step=1)          # no-op
    model = nn.Linear(2, 2)
    snap = [p.detach().clone() for p in model.parameters()]
    with e.swapped_in(model):                     # no swap, no crash
        pass
    assert all(torch.equal(a, b) for a, b in zip(snap, model.parameters()))


def test_tracks_constant():
    e = EMA(_named([0.0]), decay=0.9)
    for s in range(1, 200):
        e.update(_named([10.0]), step=s)
    assert abs(e.shadow["w"].item() - 10.0) < 1e-3


def test_one_step_is_convex_combination():
    e = EMA(_named([0.0]), decay=0.9)
    e.update(_named([1.0]), step=1)
    assert abs(e.shadow["w"].item() - 0.1) < 1e-6   # 0.9*0 + 0.1*1 (fp32)


def test_swap_in_then_out_restores_exactly():
    torch.manual_seed(0)
    model = nn.Linear(3, 3)
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}
    e = EMA(model.named_parameters(), decay=0.99)
    for _ in range(50):                           # drive live weights away from EMA
        for p in model.parameters():
            p.data.add_(torch.randn_like(p))
        e.update(model.named_parameters(), step=1)
    with e.swapped_in(model):
        swapped = {n: p.detach().clone() for n, p in model.named_parameters()}
    after = {n: p.detach().clone() for n, p in model.named_parameters()}
    # inside the context we saw EMA values, not the live ones
    assert any(not torch.equal(swapped[n], after[n]) for n in orig)
    # after the context the live weights are back, bit-exact
    live = {n: p.detach().clone() for n, p in model.named_parameters()}
    assert all(torch.equal(live[n], after[n]) for n in orig)
    # and the EMA is close to the running mean, not the init
    assert all(not torch.equal(swapped[n], orig[n]) for n in orig)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
