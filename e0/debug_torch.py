import sys
import torch
import os

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Environment variables:")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"  FORCE_CPU: {os.environ.get('FORCE_CPU', 'Not set')}")

try:
    import torch_geometric
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
except ImportError:
    print("PyTorch Geometric not installed")

# Try to create a small tensor
print("Creating a tensor...")
x = torch.randn(5, 5)
print(f"Tensor device: {x.device}")
print("Tensor created successfully")

# Try simple operations
print("Testing tensor operations...")
y = torch.matmul(x, x)
print("Matrix multiplication successful")
z = torch.nn.functional.softmax(y, dim=1)
print("Softmax operation successful")

print("Basic PyTorch checks passed!")
