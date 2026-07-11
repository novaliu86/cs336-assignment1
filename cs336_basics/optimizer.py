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
        if betas[0] < 0 or betas[0] >= 1.0:
            raise ValueError(f"Invalid beta 1: {betas[0]}")
        if betas[1] < 0 or betas[1] >= 1.0:
            raise ValueError(f"Invalid beta 2: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid decay rate: {dr}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = {"lr": lr, "b1": betas[0], "b2": betas[1], "dr": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["b1"]
            b2 = group["b2"]
            dr = group["dr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # If state is completely empty, initialize tracking arrays
                if len(state) == 0:
                    state["t"] = 0
                    # Use zeros_like to guarantee matching device and datatype properties
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                if dr != 0:
                    p.mul_(1.0 - lr * dr)

                m.mul_(b1).add_(grad, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

                bias_correction1 = 1.0 - (b1 ** t)
                bias_correction2 = 1.0 - (b2 ** t)
                step_size = lr / bias_correction1
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

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

@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clips the global gradient norm entirely on-device using tensors.
    Cleaned up to remove redundant device tracking logic.
    """
    param_list = list(parameters)

    # 1. Gather on-device squared sums directly from available layers
    device_squared_sums = [
        p.grad.pow(2).sum() for p in param_list if p.grad is not None
    ]

    if not device_squared_sums:
        return

    # 2. Combine and compute the square root entirely on the active device
    total_squared_tensor = torch.stack(device_squared_sums).sum()
    l2_norm_tensor = torch.sqrt(total_squared_tensor)

    # 3. Branchless scaling factor calculation using torch.clamp
    clamp_max = torch.clamp(l2_norm_tensor, min=max_l2_norm)
    scale_tensor = max_l2_norm / clamp_max

    # 4. Apply the scaling factor in-place asynchronously
    for p in param_list:
        if p.grad is not None:
            p.grad.mul_(scale_tensor)


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], 5, 0, (0.9, 0.999))
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
