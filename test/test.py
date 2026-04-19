import torch

x = torch.randn(5, 5, 3, 3)
print(x)
print(torch.triu(x))
