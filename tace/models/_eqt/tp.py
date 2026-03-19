# ################################################################################
# # Authors: Zemin Xu
# # License: MIT, see LICENSE.md
# ################################################################################

# from typing import Optional


# import torch
# import equitorch as eqt
# from equitorch.irreps import check_irreps
 
# class  SpatialTensorProduct(torch.nn.Module):
#     def __init__(
#         self,
#         irreps_in1: eqt.irreps.Irreps,
#         irreps_in2: eqt.irreps.Irreps,
#         irreps_out: eqt.irreps.Irreps,
#         truncation: Optional[int] = None,
#         num_longitude: Optional[int] = None,
#         num_latitude: Optional[int] = None,
#     ) -> None:


#         super().__init__()

#         self.irreps_in1 = check_irreps(irreps_in1)
#         self.irreps_in2 = check_irreps(irreps_in2)
#         self.irreps_out = check_irreps(irreps_out)
#         l1s = sorted(set(ir.l for ir in irreps_in1))
#         l2s = sorted(set(ir.l for ir in irreps_in2))
#         l3s = sorted(set(ir.l for ir in irreps_out))
#         l1max = max(l1s)
#         l2max = max(l2s)
#         self.truncation = truncation or (l1max + l2max)
#         assert self.truncation >= min(l1max, l2max)
#         assert self.truncation <= l1max + l2max
#         if num_latitude is None and num_longitude is None:
#             self.num_latitude = 2 * (self.truncation + 1)
#             self.num_longitude = 2 * (self.truncation+ 1) + 1
#         else:
#             self.num_latitude = num_latitude
#             self.num_longitude = num_longitude  
#         assert isinstance(self.num_latitude, int)    
#         assert isinstance(self.num_longitude, int) 
