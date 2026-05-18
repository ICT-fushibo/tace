
# import torch
# from tace.models.ictd import ICTD


# def num_elements(lmax):
#     return sum(3**l for l in range(lmax+1))


# print(num_elements(3))



# B = 2
# C = 3
# lmax = 3


# icts = []
# for l in range(lmax+1):
#     ct = torch.randn(B, 3**l, C)
#     cs = ICTD(l)[2][0]
#     st = torch.einsum('ij, bic -> bjc', cs, ct)
#     print(st.shape)

# cs [3, 3, 3, 5]
# torch.einsum(
#     'ai, bj, ck, nijkC, abcs-> nsC' ,
#     R, R, R, T, cs
# )

