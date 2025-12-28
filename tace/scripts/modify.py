################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

# TODO, modify target_property


# target_property = model.target_property
# compute_flags = {}
# for p in self.extra_compute_first_derivative:
#     model.compute_first_derivative = True
#     compute_flags.update(
#         {
#             p: True
#         }
#     )
# for p in self.extra_compute_second_derivative:
#     model.compute_second_derivative = True
#     compute_flags.update(
#         {
#             p: True
#         }
#     )
# for p, flag in compute_flags.items():
#     if flag:
#         setattr(model.flags, f"compute_{p}", True)
#         target_property.append(p)