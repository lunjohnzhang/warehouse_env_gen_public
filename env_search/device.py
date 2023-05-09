"""Global PyTorch device."""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"