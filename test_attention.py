import torch
import xformers.ops as xops

torch.manual_seed(42)
device = "cuda:0"
dtype = torch.float32

B = 4
H = 16
N = 2048
D = 128

q = torch.rand(B, H, N, D, dtype=dtype, device=device)
k = torch.rand(B, H, N, D, dtype=dtype, device=device)
v = torch.rand(B, H, N, D, dtype=dtype, device=device)

m = torch.ones((N, N), dtype=torch.bool, device=device).triu(N - N + 1)

shapes = [[1, 1, 1, 1],
            [1, 1, 1, N],
            [1, 1, N, N],
            [1, H, N, N],
            [B, H, N, N],
            [B, 1, N, N],
            [B, 1, N, N],]

q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)

for shape in shapes:
    m = torch.randint(0, 1, size=shape, device=device).to(torch.bool)
    m = m.float().masked_fill(m, float("-inf"))

    m = m.expand(B, H, N, N)

    try:
        out = xops.memory_efficient_attention(q, k, v, attn_bias=m)
        print(f"success: {shape}")
    except:
        print(f"failure: {shape}")
