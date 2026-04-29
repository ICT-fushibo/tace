import torch
import torch.nn as nn
import torch.optim as optim


batch_size = 4
in_features = 3
out_features = 5
real_input = torch.randn(batch_size, in_features, 2, dtype=torch.float32)
complex_input = torch.view_as_complex(real_input)   # shape: [4, 3]

linear = nn.Linear(in_features, out_features, dtype=torch.complex64)  # 权重为复数

print("权重 shape:", linear.weight.shape)   # [5, 3]
print("偏置 shape:", linear.bias.shape)     # [5]
print("权重 dtype:", linear.weight.dtype)    # torch.complex64

complex_output = linear(complex_input)       # shape: [4, 5]

print("输出 shape:", complex_output.shape)
print("输出 dtype:", complex_output.dtype)

real_output = torch.view_as_real(complex_output)   # shape: [4, 5, 2]
print("转为实数表示 shape:", real_output.shape)


loss_fn = lambda x: torch.sum(torch.abs(x)**2)
loss = loss_fn(complex_output)
loss.backward()

print("权重梯度 shape:", linear.weight.grad.shape)  