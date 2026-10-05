"""Plain-assert tests for the Session B architecture toggles (SwiGLU MLP,
QK-norm). Needs torch -> run on the pod.
    python3 test_arch.py
"""
import torch

from train import GPTConfig, GPT

_BASE = dict(vocab_size=16000, block_size=128, n_layer=6, n_head=8,
             d_model=512, dropout=0.1)
_TINY = dict(vocab_size=64, block_size=8, n_layer=2, n_head=2,
             d_model=16, dropout=0.0)


def _nparams(cfg):
    return sum(p.numel() for p in GPT(cfg).parameters() if p.requires_grad)


def test_swiglu_param_matched_to_gelu():
    g = _nparams(GPTConfig(**_BASE, mlp_kind="gelu"))
    s = _nparams(GPTConfig(**_BASE, mlp_kind="swiglu"))
    assert abs(s - g) / g < 0.02, (g, s)


def test_swiglu_changes_output():
    x = torch.randint(0, 64, (2, 8))
    torch.manual_seed(0)
    a, _ = GPT(GPTConfig(**_TINY, mlp_kind="gelu")).eval()(x)
    torch.manual_seed(0)
    b, _ = GPT(GPTConfig(**_TINY, mlp_kind="swiglu")).eval()(x)
    assert not torch.allclose(a, b)


def test_swiglu_matrices_are_2d_for_muon():
    m = GPT(GPTConfig(**_TINY, mlp_kind="swiglu"))
    names = [n for n, p in m.named_parameters() if "mlp" in n and p.ndim == 2]
    # 3 matrices per block, 2 blocks
    assert len([n for n in names if "w_gate" in n or "w_up" in n or "w_down" in n]) == 6


def test_qk_norm_off_is_bit_exact():
    x = torch.randint(0, 64, (2, 8))
    torch.manual_seed(0)
    a, _ = GPT(GPTConfig(**_TINY)).eval()(x)
    torch.manual_seed(0)
    b, _ = GPT(GPTConfig(**_TINY, qk_norm=False)).eval()(x)
    assert torch.equal(a, b)


def test_qk_norm_on_changes_output_and_adds_params():
    x = torch.randint(0, 64, (2, 8))
    torch.manual_seed(0)
    off = GPT(GPTConfig(**_TINY, qk_norm=False))
    torch.manual_seed(0)
    on = GPT(GPTConfig(**_TINY, qk_norm=True))
    assert _nparams(GPTConfig(**_TINY, qk_norm=True)) > _nparams(GPTConfig(**_TINY))
    a, _ = off.eval()(x)
    b, _ = on.eval()(x)
    assert not torch.allclose(a, b)


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
