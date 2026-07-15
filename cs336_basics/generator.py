import torch
import torch.nn as nn
import json
from pathlib import Path

from cs336_basics.train import torch_dtype_from_string
from cs336_basics.model import TransformerLm
from cs336_basics.optimizer import AdamW
from cs336_basics.serialization import load_checkpoint
from cs336_basics.nn_utils import softmax
from cs336_basics.tokenizer import Tokenizer

def load_model(run_name: str) -> tuple[TransformerLm, torch.device]:

    # Open the file and load its contents directly into a Python dictionary
    with open(Path(f"runs/{run_name}/config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device(cfg["data"]["device"])
    model_dtype = torch_dtype_from_string(cfg["model"]["torch_dtype"])
    d_ff = cfg["model"]["d_ff"] if cfg["model"]["d_ff"] is not None else 4 * cfg["model"]["d_model"]
    context_length = cfg["model"]["context_length"]

    model = TransformerLm(
        vocab_size=cfg["model"]["vocab_size"],
        context_length=context_length,
        d_model=cfg["model"]["d_model"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        d_ff=d_ff,
        theta=cfg["model"]["rope_theta"],
        device=device,
        dtype=model_dtype
    ).to(device)
    model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr_max"],
        betas=(cfg["optim"]["beta1"], cfg["optim"]["beta2"]),
        eps=cfg["optim"]["eps"],
        weight_decay=cfg["optim"]["weight_decay"]
    )

    load_checkpoint(Path(f"runs/{run_name}/checkpoints/ckpt.best.pt"), model, optimizer)

    return (model, device, context_length)

@torch.no_grad()
def generate_text(
    model: nn.Module, 
    prompt_tokens: list[int], 
    max_new_tokens: int, 
    context_length: int, 
    temperature: float = 0.8,
    device: str = "cpu"
) -> list[int]:
    """
    Autoregressive text generator aligned with the run_transformer_lm model interface.
    """
    # Initialize the generated sequence with your starting prompt tokens array
    generated = list(prompt_tokens)
    
    model.eval() # Ensure dropout layers are turned off
    
    for _ in range(max_new_tokens):
        # 1. Enforce context sliding window limit bounds
        context_window = generated[-context_length:]
        
        # 2. Convert raw list elements directly into a standard 2D PyTorch Batch Tensor
        # Shape: [batch_size=1, current_seq_len]
        input_tensor = torch.tensor([context_window], dtype=torch.long, device=device)
        
        # 3. Model forward pass requires ONLY the token matrix argument now
        # Output layout: [1, current_seq_len, vocab_size]
        logits = model(input_tensor)
        
        # 4. Isolate final step logit coordinates [1, vocab_size]
        next_token_logits = logits[:, -1, :]
        
        # 5. Apply temperature scaling controls
        scaled_logits = next_token_logits / temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)
        
        # 6. Sample using a multinomial random distribution lookup step
        next_token_id = torch.multinomial(probabilities, num_samples=1).item()
        
        # 7. Append back into tracking stream
        generated.append(next_token_id)
        
    return generated


if __name__ == "__main__":

    tokenizer = Tokenizer.from_files("data/BPE-TinyStoriesV2-GPT4.pkl", special_tokens=["<|endoftext|>"])

    run_name = "ts_baseline_with_mps"
    (model, device, context_length) = load_model(run_name)

    mock_model = model.eval()

    # 1. Pass a starting prompt sequence (e.g., encoded tokens for "The cat sat")
    prompt = tokenizer.encode("The cat sat")
    max_tokens_to_generate = 1000

    # 2. Run generation pass
    final_output_sequence = generate_text(
        model=mock_model,
        prompt_tokens=prompt,
        max_new_tokens=max_tokens_to_generate,
        context_length=context_length,
        temperature=1.,
        device=device
    )

    print("Original Prompt Length:", len(prompt))
    # print("Final Output Array:   ", final_output_sequence)
    print("Final Sequence Length:", len(final_output_sequence)) 
    # Expected length calculation: 3 prompt tokens + 5 new tokens = 8 total tokens!

    generated_text = tokenizer.decode(final_output_sequence)
    print(f"Generated text: {generated_text}")