
## Notation
b: batch_size
s: seq_len
m: d_model
h: num_heads
f: d_ff = 8m / 3
k: d_k = m / h
l: num_layers
v: vocab_size


## Breakdown

### Rmsnorm
Number of parameters: m
FLOPs: 2bsm (@ weight) + 2bsm (rms) = 4bsm


### Swiglu
Number of parameters: 3mf
FLOPs: 2bsmf (@ w1) + 2bsmf (@ w3) + 2bsfm (@ w2) = 6bsfm

### Rope
Number of parameters: 0
FLOPs: 2bsk = 2bsm / h

### scaled_dot_product_attention
Number of parameters: 0
FLOPs: 2bhsks (Q @ K) + 2bhssk (@ V) + bhss (softmax) = 4bhsks = 4bsms


### MultiheadSelfAttention
Number of parameters: 4mm
FLOPs: 2bsmm (@ q_proj) + 2bsmm (@ k_proj) + 2bsmm (@ v_proj) + 2bsm (rope Q) + 2bsm (rope K) + 2bsmm (@ output_proj) + 4bsms (scaled_dot_product_attention) = 8bsmm + 4bsms = 2bsm(4m + 2s)

### TransformerBlock
Number of parameters: 4mm + 2m + 3mf = m(4m + 3f) = 12mm
FLOPs: 2bsm(4m + 2s) + 8bsm + 6bsfm = 2bsm(2s + 4m + 3f) = 2bsm(2s + 12m)


### TransformerLm
Number of parameters:  vm (embedding) + 12mml (transformers) + m (ln_final) + mv (lm_head) = 12mml + 2mv

FLOPs: 2bsm(2s + 12m)l (transformers) + 4bsm(ln_final) + 2bsmv (lm_head) = 2bsm[(2s + 12m)l + v]


## Number of parameters
GPT2 small: 162 M (calculated) v.s. 124 M (real)
GPT2 medium: 404 M (calculated) v.s. 355 M (real)
GPT2 large: 836 M (calculated) v.s. 762 M (real)
GPT2 XL: 1.635 B (calculated) v.s. 1.5 B (real)