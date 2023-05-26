import torch
from attention import build_attn_bias

def foo(seq_len):
    query = torch.randn((16, seq_len, 768), device='cuda',dtype=torch.bfloat16)
    key = torch.randn((16, seq_len, 768), device='cuda',dtype=torch.bfloat16)
    attn_bias = torch.zeros((1, 12, 1, seq_len), device='cuda',dtype=torch.bfloat16)
    attn_bias = build_attn_bias(
                'torch',
                attn_bias,
                n_heads=12,
                seq_len=seq_len,
                causal=True,
                alibi=True)
    return attn_bias[:, :, -query.size(1):, -key.size(1):]


print(foo(2048))
opt_foo1 = torch.compile(foo)
print(opt_foo1(2048))