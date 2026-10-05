"""§4.11 verification: sdpa vs manual attention agree in eval mode (dropout off)."""
import sys, torch
sys.path.insert(0, "mainrun")
from train import CausalSelfAttention, GPTConfig

def build(impl):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=100, block_size=64, n_layer=6, n_head=8,
                    d_model=512, dropout=0.1, attn_impl=impl)
    m = CausalSelfAttention(cfg).eval()
    return m

m_manual = build("manual")
m_sdpa = build("sdpa")
# copy weights so only the impl differs
m_sdpa.load_state_dict(m_manual.state_dict())

torch.manual_seed(1)
x = torch.randn(4, 40, 512)
with torch.no_grad():
    y_manual = m_manual(x)
    y_sdpa = m_sdpa(x)

maxdiff = (y_manual - y_sdpa).abs().max().item()
print(f"max |manual - sdpa| = {maxdiff:.2e}")
assert maxdiff < 1e-5, f"sdpa/manual disagree by {maxdiff}"
print("SDPA EQUIVALENCE OK (atol 1e-5)")
