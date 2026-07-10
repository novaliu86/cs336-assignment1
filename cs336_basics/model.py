import math

import torch
from einops import einsum, rearrange, repeat


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.std = (2 / (in_features + out_features)) ** 0.5
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=self.std,
            a=-3 * self.std,
            b=3 * self.std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, " ... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1,
            a=-3,
            b=3,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class Rmsnorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(d_model))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = einsum(x / rms, self.weight,
                        "... d_model, d_model -> ... d_model")
        # Return the result in the original dtype
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * x


class Swiglu(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape:  [batch_size, seq_len, d_model]
        Output shape: [batch_size, seq_len, d_model]
        """
        # SwiGLU formula: (SiLU(w1(x)) * w3(x)) @ w2
        up_proj1 = self.w1(x)
        gate = silu(up_proj1)
        up_proj3 = self.w3(x)

        # Element-wise multiplication followed by final down projection
        return self.w2(gate * up_proj3)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rearranges adjacent elements in pairs to execute the 2D rotation matrix trick.
    Input shape:  [..., head_dim]
    """
    # 1. Split the final 'head_dim' into pairs of elements (x1 and x2)
    # 'd' represents the total pairs, so 2*d == head_dim
    x1, x2 = rearrange(x, '... (d pair) -> pair ... d', pair=2)

    # 2. Invert x2 and stitch them back together in interleaved order: [-x2, x1]
    # We combine them back into the final 'head_dim' dimension
    rotated = rearrange([-x2, x1], 'pair ... d -> ... (d pair)', pair=2)

    return rotated


class Rope(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # 1. Compute the inverse frequencies for the rotation angles
        # Only compute for d_k // 2 because frequencies are applied to pairs
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        # 2. Generate a sequential time axis (0, 1, 2, ..., max_seq_len - 1)
        t = torch.arange(max_seq_len, dtype=torch.float32)

        # 3. Outer product: calculate frequency matrix [max_seq_len, d_k // 2]
        freqss = torch.outer(t, inv_freq)

        # 4. Duplicate each column to map pairs cleanly [max_seq_len, d_k]
        # This yields [freq0, freq0, freq1, freq1, ...]
        emb = repeat(freqss, '... d -> ... (d 2)')

        # 5. Register buffers so they track with model.to(device) automatically
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        if token_positions == None:
            seq_len = x.size(-2)
            token_positions = torch.arange(seq_len)

        # print(f"x[0][1][1]: {x[0][1][1]}")
        # print(f"token_positions[1]: {token_positions[1]}")
        # print(f"rotate_half(x)[0][1][1]: {rotate_half(x)[0][1][1]}")
        # print(
        #     f"self.cos[token_positions][1][0]: {self.cos[token_positions][1][0]}")
        # print(
        #     f"self.sin[token_positions][1][0]: {self.sin[token_positions][1][0]}")
        # print(
        #     f"self.cos[token_positions][1][1]: {self.cos[token_positions][1][1]}")
        # print(
        #     f"self.sin[token_positions][1][1]: {self.sin[token_positions][1][1]}")
        # print(f"shape of x: {x.shape}")
        # print(f"shape of token_positions: {token_positions.shape}")
        # print(
        #     f"shape of self.cos[token_positions]: {self.cos[token_positions].shape}")
        return einsum(x, self.cos[token_positions], "... seq_len d_k, ... seq_len d_k -> ... seq_len d_k") + einsum(rotate_half(x), self.sin[token_positions], "... seq_len d_k, ... seq_len d_k -> ... seq_len d_k")


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies the stable Softmax activation function along the i-th dimension.

    Formula: exp(x_i - max(x)) / sum(exp(x_i - max(x)))
    """
    # 1. Find the maximum value along the i-th dimension for numerical stability.
    # keepdim=True ensures the dimension isn't dropped, keeping shapes aligned.
    max_val = torch.max(x, dim=dim, keepdim=True).values

    # 2. Subtract the max value (Safe Shift) and exponentiate.
    # If an item was originally very large, subtracting the max drops it down to 0,
    # making exp(0) == 1.0 (safely preventing infinity errors).
    exp_x = torch.exp(x - max_val)

    # 3. Sum the exponentiated values along the i-th dimension.
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    # 4. Perform element-wise division to get final probabilities.
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: torch.Tensor,  # Float[Tensor, " ... queries d_k"]
    K: torch.Tensor,  # Float[Tensor, " ... keys d_k"]
    V: torch.Tensor,  # Float[Tensor, " ... keys d_v"]
    mask: torch.Tensor | None = None,  # Bool[Tensor, " ... queries keys"]
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(K.size(-1))
    x = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") * scale
    if mask is not None:
        # Fills masked positions with a large negative number so exp(-inf) approaches 0.0
        x = x.masked_fill(~mask, float('-inf'))
    attention_weights = softmax(x, dim=-1)
    return einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


def create_causal_mask(seq_len: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generates a boolean causal mask of shape [1, 1, seq_len, seq_len].

    True means KEEP/ATTEND, False means IGNORE/HIDE.
    """
    # 1. Create a matrix of ones with dimensions [seq_len, seq_len]
    ones = torch.ones(seq_len, seq_len, device=device)

    # 2. Extract the lower-triangular part (including the diagonal)
    lower_triangular = torch.tril(ones)

    # 3. Cast to boolean type
    bool_mask = lower_triangular.to(torch.bool)

    # 4. Reshape to [1, 1, seq_len, seq_len] to allow automatic broadcasting
    # across arbitrary batch sizes and parallel attention heads
    return bool_mask.unsqueeze(0).unsqueeze(0)


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be cleanly divisible by num_heads"

        self.num_heads = num_heads
        hd_k = d_model
        hd_v = d_model
        self.q_proj = Linear(hd_k, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(hd_k, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(hd_v, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, hd_v, device=device, dtype=dtype)

        if theta and max_seq_len:
            self.rope = Rope(theta, d_model / num_heads, max_seq_len)
        else:
            self.rope = None

        self.reset_parameters()

    def reset_parameters(self):
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.output_proj.reset_parameters()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)

        Q = rearrange(q_proj, '... s (h d) -> ... h s d', h=self.num_heads)
        K = rearrange(k_proj, '... s (h d) -> ... h s d', h=self.num_heads)

        if self.rope != None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        V = rearrange(v_proj, '... s (h d) -> ... h s d', h=self.num_heads)

        seq_len = x.size(-2)
        mask = create_causal_mask(seq_len, device=x.device)

        out_concat = scaled_dot_product_attention(Q, K, V, mask)
        out_concat = rearrange(out_concat, '... h s d -> ... s (h d)')
        return self.output_proj(out_concat)



class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float | None = None, max_seq_len: int | None = None, device = None, dtype = None):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ln1 = Rmsnorm(d_model, device=device, dtype=dtype)
        self.ffn = Swiglu(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = Rmsnorm(d_model, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        self.attn.reset_parameters()
        self.ln1.reset_parameters()
        self.ffn.reset_parameters()
        self.ln2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TransformerLm(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        device = None,
        dtype = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device = device, dtype = dtype)
        self.layers = torch.nn.ModuleList([ TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device = device, dtype = dtype) for _ in range(num_layers)])
        self.ln_final = Rmsnorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        self.token_embeddings.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.ln_final.reset_parameters()
        self.lm_head.reset_parameters()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)