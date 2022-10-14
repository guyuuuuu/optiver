import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

x_ones = torch.ones_like(x_data)
print(x_ones)

tensor = torch.rand(1, 3, 4, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

print(tensor)
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")