import torch
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.std = (2 / (in_features + out_features)) ** 0.5
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
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
        result = einsum(x / rms, self.weight, "... d_model, d_model -> ... d_model")
        # Return the result in the original dtype
        return result.to(in_dtype)

