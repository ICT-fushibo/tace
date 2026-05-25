import torch
from e3nn import o3


torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False, precision=8)
torch.manual_seed(0)










batch = 3
mmax = 2
lmax = 3
channel = 1

random_vector = torch.randn(batch, 3)


random_vector2 = torch.randn(batch, 3)

A = torch.randn(3, 3)
Q, R = torch.linalg.qr(A)
if torch.det(Q) < 0:
    Q[:, 0] *= -1
rotation_matrix = Q
rotated_vector2 = random_vector2 @ rotation_matrix.T


from tace.models.so2 import SO3Rotation, so2_expand_index

num_components, expand_index = so2_expand_index(mmax, lmax)


so3_rotation = SO3Rotation(
    lmax,
    mmax,
)
so3_rotation.set_wigner(random_vector)

so3_feats = o3.spherical_harmonics(list(range(lmax+1)), random_vector2, normalize=True).unsqueeze(-1)
so2_feats = so3_rotation.rotate(so3_feats)
print(so2_feats[0, :, 0])


so3_feats = o3.spherical_harmonics(list(range(lmax+1)), rotated_vector2, normalize=True).unsqueeze(-1)
so2_feats = so3_rotation.rotate(so3_feats)
print(so2_feats[0, :, 0])

