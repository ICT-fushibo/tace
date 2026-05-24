





import torch

torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False, precision=8)
torch.manual_seed(0)


batch = 3
mmax = 2
lmax = 3
channel = 5

random_vector = torch.randn(batch, 3)


from tace.models.so2 import SO3Rotation, so2_expand_index

num_components, expand_index = so2_expand_index(mmax, lmax)


so3_rotation = SO3Rotation(
    lmax,
    mmax,
)
so3_rotation.set_wigner(random_vector)


so3_feats = torch.randn(batch, (lmax+1)**2, channel)
so2_feats = so3_rotation.rotate(so3_feats)


from tace.models.so2.so2 import SO2Linear, W1SO2Linear

so2_linear1 =  SO2Linear(mmax, lmax, channel, channel+1)
so2_linear2 =  W1SO2Linear(mmax, lmax, channel, channel+1)
so2_linear2.m0_rlinear = so2_linear1.m0_rlinear

for m in range(1, mmax+1):
    Cout = so2_linear2.ms_clinear[m-1].fc.weight.data.shape[0]
    with torch.no_grad():
        so2_linear2.ms_clinear[m-1].fc.weight.copy_(
            so2_linear1.ms_clinear[m-1].fc.weight[:Cout]
        )
    assert torch.allclose(
        so2_linear1.ms_clinear[m-1].fc.weight[:Cout], 
        so2_linear2.ms_clinear[m-1].fc.weight
    )
    

so3_feats1 = so3_rotation.rotate_inv(so2_linear1(so2_feats))
so3_feats2 = so3_rotation.rotate_inv(so2_linear2(so2_feats))

print(so3_feats1[0, :, 0])
print(so3_feats2[0, :, 0])
