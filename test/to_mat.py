import math
import torch

from tace.lightning import load_tace
from tace.models.linear import e3nnLinear, e3nnElementLinear
model = load_tace("/home/xuzemin/TACE-OMAT24-XL-39-787670-0.0642.ckpt")

for module in model.modules():

    if isinstance(module, e3nnLinear) or isinstance(module, e3nnElementLinear):
        old_weight = module.weight
        if old_weight.ndim == 1:
            plist = torch.nn.ParameterList()
            offset = 0
            for ins in module.linear.instructions:
                shape = tuple(ins.path_shape)
                numel = math.prod(shape)
                w = (
                    old_weight.data[offset:offset + numel]
                    .view(shape)
                    .clone()
                )
                plist.append(torch.nn.Parameter(w))
                offset += numel
            del module._parameters["weight"]
            module.weight = plist
            module.use_matrix_weight = True
            # if (
            #     hasattr(module, "bias")
            #     and isinstance(module.bias, torch.nn.Parameter)
            # ):
            #     old_bias = module.bias
            #     bplist = torch.nn.ParameterList()
            #     for sl in module._bias_slices:
            #         b = old_bias.data[sl].clone()
            #         bplist.append(torch.nn.Parameter(b))
            #     del module._parameters["bias"]

            #     module.bias = bplist

        elif old_weight.ndim == 2:
            num_elements = old_weight.shape[0]
            plist = torch.nn.ParameterList()
            offset = 0
            for ins in module.linear.instructions:
                shape = tuple(ins.path_shape)
                numel = math.prod(shape)
                w = (
                    old_weight.data[:, offset:offset + numel]
                    .view(num_elements, *shape)
                    .clone()
                )
                plist.append(torch.nn.Parameter(w))
                offset += numel
            del module._parameters["weight"]
            module.weight = plist
            module.use_matrix_weight = True
            if (
                hasattr(module, "bias")
                and isinstance(module.bias, torch.nn.Parameter)
            ):

                module.bias.data = module.bias.data.view(-1)

                
torch.save(
    {
        "state_dict": model.state_dict(),
        "cfg": model.readout_fn.model_config,
        "target_property": model.get_target_property(),
        "embedding_property": model.get_embedding_property(),
        "statistics": model.readout_fn.statistics,
    }, 
    "mat.pt"
)