
import torch
import numpy as np

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a batch of inputs and targets from a flat array of token IDs.

    Args:
        dataset: A 1D NumPy integer array containing the full dataset tokens.
        batch_size: The number of sequences to sample in the batch (B).
        context_length: The sequence length for the transformer window (T).
        device: The target PyTorch device string (e.g., 'cpu', 'cuda:0').

    Returns:
        inputs:  A PyTorch LongTensor of shape (batch_size, context_length) on the target device.
        targets: A PyTorch LongTensor of shape (batch_size, context_length) on the target device.
    """
    # 1. Calculate the maximum safe starting index in the array.
    # We subtract context_length because the target window is shifted forward by 1 token.
    max_start_idx = len(dataset) - context_length

    # 2. Randomly sample starting positions for each row in the batch
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    # 3. Use an index grid to extract all sequences in parallel via NumPy advanced indexing.
    # We construct a 2D matrix of shape (batch_size, context_length) containing the exact element coordinates.
    offset_grid = np.arange(context_length)
    sampling_matrix = start_indices[:, np.newaxis] + offset_grid

    # 4. Extract inputs and targets using the precomputed matrix positions
    x_np = dataset[sampling_matrix]
    y_np = dataset[sampling_matrix + 1] # Shifted forward by 1 token

    # 5. Convert to PyTorch LongTensors (int64) and push them directly to the requested device
    inputs = torch.from_numpy(x_np.astype(np.int64)).to(device)
    targets = torch.from_numpy(y_np.astype(np.int64)).to(device)

    return inputs, targets
