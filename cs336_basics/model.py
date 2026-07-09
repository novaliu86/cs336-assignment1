import math

import torch
from einops import einsum, rearrange, repeat


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.std = (2 / (in_features + out_features)) ** 0.5
        self.weights = torch.nn.Parameter(
            torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(
            self.weights,
            mean=0.0,
            std=self.std,
            a=-3 * self.std,
            b=3 * self.std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, " ... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.empty(vocab_size, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(
            self.weights,
            mean=0.0,
            std=1,
            a=-3,
            b=3,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class Rmsnorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weights = torch.nn.Parameter(torch.empty(d_model))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = einsum(x / rms, self.weights,
                        "... d_model, d_model -> ... d_model")
        # Return the result in the original dtype
        return result.to(in_dtype)


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
        gate = torch.sigmoid(up_proj1) * up_proj1
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
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
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
