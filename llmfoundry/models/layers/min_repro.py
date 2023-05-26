import torch
import math
def build_attn_bias(
    attn_impl,
    attn_bias,
    n_heads,
    seq_len,
    causal=False,
    alibi=False,
    alibi_bias_max=8,
):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            # in place add alibi to attn bias
            device, dtype = attn_bias.device, attn_bias.dtype
            attn_bias = attn_bias.add(
                build_alibi_bias(
                    n_heads,
                    seq_len,
                    full=not causal,
                    alibi_bias_max=alibi_bias_max,
                    device=device,
                    dtype=dtype,
                ))
        return attn_bias
    else:
        raise ValueError(f'{attn_impl=} is an invalid setting.')


def gen_slopes(n_heads, alibi_bias_max=8, device=None):
    _n_heads = 2**math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = (1. / torch.pow(2, m))

    if _n_heads != n_heads:
        # if n_heads is not a power of two,
        # Huggingface and FasterTransformer calculate slopes normally,
        # then return this strided concatenation of slopes
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]

    return slopes.view(1, n_heads, 1, 1)


def build_alibi_bias(
    n_heads,
    seq_len,
    full=False,
    alibi_bias_max=8,
    device=None,
    dtype=None,
):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32,
                              device=device).view(1, 1, 1, seq_len)
    if full:
        # generate 1 x Heads x SeqLen x SeqLen alibi bias mask
        # otherwise the mask is 1 x Heads x 1 x SeqLen (which is broadcast to the appropriate size)
        alibi_bias = alibi_bias - torch.arange(
            1 - seq_len, 1, dtype=torch.int32, device=device).view(
                1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)

    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)


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