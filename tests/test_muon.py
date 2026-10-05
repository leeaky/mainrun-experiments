"""Plain-assert tests for muon.py. Needs torch -> run on the pod.
    python3 test_muon.py
"""
import torch

from muon import zeropower_via_newtonschulz5, Muon, CombinedOptimizer

torch.manual_seed(0)


def test_newtonschulz_normalizes_spectrum_wide():
    # NS-5 does not fully orthogonalize (Keller Jordan's coeffs deliberately
    # trade exactness for speed); it drives the singular values of a
    # well-conditioned matrix into a tight band around 1.
    G = torch.randn(64, 256)
    s0 = torch.linalg.svdvals(G)
    s1 = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps=5))
    assert s1.min() > 0.5 and s1.max() < 1.4, (s1.min().item(), s1.max().item())
    assert (s1.max() / s1.min()) < 2.0 < (s0.max() / s0.min())  # cond number shrank


def test_newtonschulz_normalizes_spectrum_tall():
    G = torch.randn(256, 64)
    s1 = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps=5))
    assert s1.min() > 0.5 and s1.max() < 1.4, (s1.min().item(), s1.max().item())


def test_newtonschulz_shape_and_determinism():
    G = torch.randn(64, 48)
    a = zeropower_via_newtonschulz5(G, steps=5)
    b = zeropower_via_newtonschulz5(G, steps=5)
    assert a.shape == G.shape
    assert torch.equal(a, b)                      # fp32, no RNG -> bit-exact


def test_muon_step_changes_param_no_nan():
    p = torch.nn.Parameter(torch.randn(16, 24))
    opt = Muon([p], lr=0.02, momentum=0.95)
    before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt.step()
    assert torch.isfinite(p).all()
    assert not torch.equal(p.detach(), before)


def test_muon_deterministic():
    g = torch.randn(16, 24)
    outs = []
    for _ in range(2):
        torch.manual_seed(1)
        p = torch.nn.Parameter(torch.zeros(16, 24))
        opt = Muon([p], lr=0.02, momentum=0.95)
        p.grad = g.clone()
        opt.step()
        outs.append(p.detach().clone())
    assert torch.equal(outs[0], outs[1])


def test_combined_optimizer_applies_shared_factor():
    p1 = torch.nn.Parameter(torch.randn(8, 8)); p1.grad = torch.zeros(8, 8)
    p2 = torch.nn.Parameter(torch.randn(8, 8)); p2.grad = torch.zeros(8, 8)
    a = Muon([p1], lr=0.02)
    b = torch.optim.AdamW([p2], lr=0.001)
    co = CombinedOptimizer([a, b], sched_fn=lambda s: 0.5 if s >= 5 else 1.0)
    assert a.param_groups[0]["lr"] == 0.02        # __init__ applied factor for step 0
    assert b.param_groups[0]["lr"] == 0.001
    co.apply_schedule(5)
    assert abs(a.param_groups[0]["lr"] - 0.01) < 1e-12
    assert abs(b.param_groups[0]["lr"] - 0.0005) < 1e-12


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
