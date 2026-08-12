import torch

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "none"
print(f"Accelerator: {device}")