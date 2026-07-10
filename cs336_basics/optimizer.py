from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0:
            raise ValueError(f"Invalid beta 1: {b1}")
        if betas[1] < 0:
            raise ValueError(f"Invalid beta 2: {b2}")
        if weight_decay < 0:
            raise ValueError(f"Invalid decay rate: {dr}")

        self.eps = eps
        defaults = {"lr": lr, "b1": betas[0], "b2": betas[1], "dr": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["b1"]
            b2 = group["b2"]
            dr = group["dr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)

                grad = p.grad.data

                lr_t = lr * math.sqrt(1 - b2 ** t)/ (1 - b1 ** t)

                p.data *= (1 - lr * dr)

                m = state.get("m", torch.zeros(p.size()))
                m = b1 * m + (1 - b1) * grad
                state["m"] = m

                v = state.get("v", torch.zeros(p.size()))
                v = b2 * v + (1 - b2) * (grad ** 2)
                state["v"] = v

                p.data -= lr_t * m / (torch.sqrt(v) + self.eps) # Update weight tensor in-place.

                state["t"] = t + 1 # Increment iteration number.
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it <= warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate

    return min_learning_rate + (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate) / 2


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total_squared_sum = 0.0
    for p in parameters:
        if p.grad is not None:
            # Square the L2 norm of this specific layer's gradient
            # and accumulate it to the global sum
            param_norm = torch.linalg.vector_norm(p.grad, ord=2)
            total_squared_sum += param_norm.item() ** 2
    l2_norm = math.sqrt(total_squared_sum)

    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= scale

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], 5, 0, (0.9, 0.999))
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
