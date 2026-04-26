import torch
from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes



class GraphSoftmax(torch.nn.Module):
    """
        1.  Reference: https://pytorch-geometric.readthedocs.io/en/2.3.1/_modules/torch_geometric/utils/softmax.html
        3.  Add `exp_rescale` to rescale the outputs of the exponentation function so that we can downscale the 
            contributions of some neighbors with an envelope function.
        4.  Add `softcap` to limit input logits to the range [- `softcap`, + `softcap`].
        5.  Add `eps` for numerical stability.
    """
    def __init__(self, eps=1e-16):
        super().__init__()
        self.eps = eps
      
    def forward(
        self, 
        src, 
        index=None, 
        ptr=None, 
        num_nodes=None, 
        dim=0,
        exp_rescale=None
    ):
        if ptr is not None:
            dim = dim + src.dim() if dim < 0 else dim
            size = ([1] * dim) + [-1]
            count = ptr[1:] - ptr[:-1]
            ptr = ptr.view(size)
            src_max = segment(src.detach(), ptr, reduce='max')
            src_max = src_max.repeat_interleave(count, dim=dim)
            out = (src - src_max).exp()
            if exp_rescale is not None:
                out = out * exp_rescale
            out_sum = segment(out, ptr, reduce='sum') + self.eps
            out_sum = out_sum.repeat_interleave(count, dim=dim)
        elif index is not None:
            N = maybe_num_nodes(index, num_nodes)
            src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
            out = src - src_max.index_select(dim, index)
            out = out.exp()
            if exp_rescale is not None:
                out = out * exp_rescale
            out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + self.eps
            out_sum = out_sum.index_select(dim, index)
        else:
            raise NotImplementedError
        
        out = out / out_sum
        
        return out
    

    def extra_repr(self):
        return 'eps={}'.format(self.eps)