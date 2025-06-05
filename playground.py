import torch

batch_size = 3

# 2次元: (batch_size, 1)
x1 = torch.tensor([[1.0], [2.0], [3.0]])
print(x1)
print(x1.shape)  # torch.Size([3, 1])

# 1次元: (batch_size,)
x2 = torch.tensor([1.0, 2.0, 3.0])
print(x2)
print(x2.shape)  # torch.Size([3])

print(x1.squeeze())
print(x1.unsqueeze())