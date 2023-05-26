import torch
from attention import build_alibi_bias
# def build_attn_bias(
#     attn_impl,
#     attn_bias,
#     n_heads,
#     seq_len,
#     causal=False,
#     alibi=False,
#     alibi_bias_max=8,
# ):
#     if attn_impl == 'flash':
#         return None
#     elif attn_impl in ['torch', 'triton']:
#         if alibi:
#             # in place add alibi to attn bias
#             device, dtype = attn_bias.device, attn_bias.dtype
#             attn_bias = attn_bias.add(
#                 build_alibi_bias(
#                     n_heads,
#                     seq_len,
#                     full=not causal,
#                     alibi_bias_max=alibi_bias_max,
#                     device=device,
#                     dtype=dtype,
#                 ))
#         return attn_bias
#     else:
#         raise ValueError(f'{attn_impl=} is an invalid setting.')

def build_attn_bias_with_xformers(
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
    elif attn_impl == 'xformers':
        if alibi:
            # in place add alibi to attn bias
            device, dtype = attn_bias.device, attn_bias.dtype
            attn_bias = attn_bias.add(
                build_alibi_bias(
                    n_heads,
                    seq_len,
                    full=True,
                    alibi_bias_max=alibi_bias_max,
                    device=device,
                    dtype=dtype,
                ))
            if causal:
                min_val = torch.finfo(attn_bias.dtype).min
                s = attn_bias.shape[-1]
                causal_mask = attn_bias.new_ones(s, s, dtype=torch.float16)
                causal_mask = causal_mask.tril()
                causal_mask = causal_mask.to(torch.bool)
                causal_mask = ~causal_mask
                causal_mask = causal_mask[-s:, -s:]
                attn_bias = attn_bias.masked_fill(causal_mask.view(1, 1, s, s),
                                                    min_val)
        return attn_bias
    else:
        raise ValueError(f'{attn_impl=} is an invalid setting.')


# def test_without_xformers(seq_len):
#     query = torch.randn((16, seq_len, 768), device='cuda',dtype=torch.bfloat16)
#     key = torch.randn((16, seq_len, 768), device='cuda',dtype=torch.bfloat16)
#     attn_bias = torch.zeros((1, 12, 1, seq_len), device='cuda',dtype=torch.bfloat16)
#     attn_bias = build_attn_bias(
#                 'torch',
#                 attn_bias,
#                 n_heads=12,
#                 seq_len=seq_len,
#                 causal=True,
#                 alibi=True)
#     return attn_bias[:, :, -query.size(1):, -key.size(1):]

def test_with_xformers(seq_len):
    query = torch.randn((16, seq_len, 768), device='cuda',dtype=torch.bfloat16)
    key = torch.randn((16, seq_len, 768), device='cuda',dtype=torch.bfloat16)
    attn_bias = torch.zeros((1, 12, 1, seq_len), device='cuda',dtype=torch.bfloat16)
    attn_bias = build_attn_bias_with_xformers(
                'torch',
                attn_bias,
                n_heads=12,
                seq_len=seq_len,
                causal=True,
                alibi=True)
    return attn_bias[:, :, -query.size(1):, -key.size(1):]


# print(test_without_xformers(2048))
print(test_with_xformers(2048))
# opt_without= torch.compile(test_without_xformers)
opt_with = torch.compile(test_with_xformers)
# print(opt_without(2048))
print(opt_with(2048))