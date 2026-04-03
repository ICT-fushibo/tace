import torch
from e3nn import o3
from math import prod

class IdentityLinear(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()

        self.linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=True,
        )

        weight = torch.zeros(self.linear.weight_numel)

        offset = 0
        for ins in self.linear.instructions:
            size = prod(ins.path_shape)
            if ins.i_in == -1:
                weight[offset:offset + size] = 0.0
            else:
                mul_in, mul_out = ins.path_shape

                if mul_in != mul_out:
                    raise 

                eye = torch.eye(mul_in)
                block = eye / ins.path_weight
                weight[offset:offset + size] = block.reshape(-1)

            offset += size

        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x):
        return self.linear(x, self.weight)

irreps_in = o3.Irreps("64x0e + 64x1o")
irreps_out = o3.Irreps("64x0e + 64x2o")


linear = IdentityLinear(irreps_in, irreps_out)
x = irreps_in.randn(5, -1)
assert torch.allclose(x[:, :64], linear(x)[:, :64])
print(linear(x)[:, 64:])