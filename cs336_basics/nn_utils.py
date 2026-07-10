
import torch
from einops import rearrange

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


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes numerically stable Cross-Entropy Loss from raw logits.

    Shapes:
        inputs:  [batch_size, num_classes] (Raw unnormalized logit scores)
        targets: [batch_size]              (Integer class label IDs)
    """
    # 1. For numerical stability, find the max logit along the class dimension
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values

    # 2. Compute the stable Log-Sum-Exp denominator: max + log(sum(exp(x - max)))
    # This prevents float overflows from exponentiating massive numbers.
    log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(inputs - max_logits), dim=-1, keepdim=True))

    # 3. Calculate log-probabilities for all classes
    log_probs = inputs - log_sum_exp

    # 4. Use your preferred einops + gather pattern to extract the target log-probabilities
    targets_2d = rearrange(targets, 'b -> b 1')
    target_log_probs = torch.gather(log_probs, dim=1, index=targets_2d)
    target_log_probs = rearrange(target_log_probs, 'b 1 -> b')

    # 5. Apply the negative log likelihood and return the mean loss across the batch
    loss = -target_log_probs
    return torch.mean(loss)
