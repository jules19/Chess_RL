"""
Check which PyTorch device this machine offers and which one the project will use.

Usage:
    python test/check_torch_device.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.model import resolve_device

print(f"PyTorch {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Project will use: {resolve_device('auto')}")
