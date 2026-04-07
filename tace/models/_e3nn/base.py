################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import abc 
from typing import Optional, List


import torch
from e3nn import o3


from ..lammps import e3nnGhostExchangeMixin


def _to_full_so3_irreps(lmax: int | List[int], num_channel: int) -> o3.Irreps:
    if isinstance(lmax, int):
        return o3.Irreps([(num_channel, (l, (-1)**l)) for l in range(lmax + 1)])
    assert isinstance(lmax, List)
    return o3.Irreps([(num_channel, (l, (-1)**l)) for l in lmax])


class NodeEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_elements: int,
        num_radial_basis: int,
        num_channel: int,
        Lmax: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.num_elements = num_elements
        self.num_radial_basis = num_radial_basis
        self.num_channel = num_channel
        self.bias = bias
        self.Lmax = Lmax

        self._setup()
    
    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError
    
    
class EdgeEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_elements: int,
        num_radial_basis: int,
        num_channel: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.num_elements = num_elements
        self.num_radial_basis = num_radial_basis
        self.num_channel = num_channel
        self.bias = bias

        self._setup()
    
    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError
    

class EdgeUpdate(torch.nn.Module):
    def __init__(
        self,
        layer: int,
        num_layers: int,
        num_elements: int,
        num_radial_basis: int,
        num_channel: int,
        edge_embedding_channel: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.layer = layer
        self.num_layers = num_layers
        self.first_layer = (layer == 0)
        self.last_layer = (layer == num_layers - 1)
        self.num_elements = num_elements
        self.num_radial_basis = num_radial_basis
        self.num_channel = num_channel
        self.edge_embedding_channel=edge_embedding_channel
        self.use_bias = bias

        self._setup()
    
    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError


class Residual(torch.nn.Module):
    def __init__(
        self,
        layer: int,
        num_layers: int,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        num_channel: int,
        num_elements: int,
        liner_type: str | List[int],
        bias: bool = True,
        window: int | None = None,

    ) -> None:
        super().__init__()

        self.layer = layer
        self.num_layers = num_layers
        self.first_layer = (layer == 0)
        self.last_layer = (layer == num_layers - 1)
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.num_channel = num_channel
        self.num_elements = num_elements
        self.use_bias = bias
        self.window = min(window or layer, layer)

        if isinstance(liner_type, list):
            self.linear_type = liner_type[layer]
        else:
            self.linear_type = liner_type

        self._setup()
    
    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError
    

class Interaction(torch.nn.Module, e3nnGhostExchangeMixin):
    def __init__(
        self,
        layer: int,
        num_layers: int,
        num_elements: int,
        avg_num_neighbors: float,
        Lmax: List[int],
        lmax: int,
        correlation: List[int],
        num_channel: int,
        edge_feats_channel: int,
        target_weight: List[int],
        num_radial_basis: int,
        radial_mlp: List[int],
        radial_bias: bool,
        irreps_node_embedding: o3.Irreps,
        l1l2: Optional[str] = None,
        scatter_norm: str = 'avg_num_neighbors',
        ictp_ictc_like: bool = True,
        bias: bool = True,
        nonlinear: str | None = None,
        edge_info_type: str = 'mlp',
        resnet_type: str | List[str] = 'BB',
        resnet_linear_type: str = 'aware',
        resnet_window: int | None = None,
        use_first_resnet: bool = False,
        # num_experts: int | None = None,
        # top_k: int | None = None,
        # num_shared_experts: int | None = None,
        num_hidden_channel: int | None = None,
    ) -> None:
        super().__init__()

        self.layer = layer
        self.num_layers = num_layers
        self.Lmax = Lmax[layer]
        self.lmax = lmax
        self.correlation = correlation[layer]
        self.num_radial_basis = num_radial_basis
        self.avg_num_neighbors = avg_num_neighbors
        self.edge_feats_channel = edge_feats_channel
        self.l1l2 = l1l2
        self.num_elements = num_elements
        self.num_channel = num_channel
        self.target_weight = target_weight
        self.radial_mlp = radial_mlp
        self.radial_bias = radial_bias
        self.ictp_ictc_like = ictp_ictc_like
        self.use_bias = bias
        self.scatter_norm = scatter_norm
        if self.scatter_norm == 'no_cutoff_density':
            self.apply_density_cutoff = False
        else:
            self.apply_density_cutoff = True
        self.radial_layer_norm = False
        if self.edge_feats_channel != self.num_radial_basis:    
            self.radial_layer_norm = True
        self.nonlinear_type = None
        self.nonlinear_act = None
        if nonlinear is not None:
            self.nonlinear_act, self.nonlinear_type = nonlinear.split('_')
        self.edge_info_type = edge_info_type
        if self.edge_info_type == 'mlp':
            self.radial_act = 'silu'
        else:
            self.radial_act = 'sigmoid'
        self.use_first_resnet = use_first_resnet
        self.resnet_type = resnet_type
        # self.num_experts = num_experts
        # self.top_k = top_k
        # self.num_shared_experts = num_shared_experts or 0
        self.resnet_linear_type = resnet_linear_type
        self.resnet_window = resnet_window
        if num_hidden_channel is not None:
            assert nonlinear is not None
        self.num_hidden_channel = num_hidden_channel or num_channel

        self.irreps_sh = _to_full_so3_irreps(self.lmax, 1)
        if self.correlation == 1:
            self.irreps_out = _to_full_so3_irreps(self.Lmax, self.num_channel)
            self.irreps_hidden = _to_full_so3_irreps(self.Lmax, self.num_hidden_channel)
        else:
            self.irreps_out = _to_full_so3_irreps(self.lmax, self.num_channel)
            self.irreps_hidden = _to_full_so3_irreps(self.lmax, self.num_hidden_channel)
        if layer == 0:
            self.irreps_in = irreps_node_embedding
        else:
            self.irreps_in = _to_full_so3_irreps(self.Lmax, self.num_channel)
        if self.layer == num_layers - 1:
            self.irreps_sc = _to_full_so3_irreps(target_weight, self.num_channel)
        else:
            self.irreps_sc = _to_full_so3_irreps(self.Lmax, self.num_channel)

        self._setup()
    
    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError


class Product(torch.nn.Module):

    def __init__(
        self,
        layer: int,
        num_layers: int,
        num_elements: int,
        Lmax: List[int],
        lmax: int,
        num_channel: int,
        num_hidden_channel: int,
        target_weight: List[int],
        correlation: List[int],
        l1l2: str | None,
        ictp_ictc_like: bool,
        bias: bool,
        num_latitude: int,
        num_longitude: int,
        truncation: int,
        trainable_scale: bool,
    ) -> None:
        super().__init__()

        self.layer = layer
        self.num_layers = num_layers
        self.Lmax = Lmax[layer]
        self.lmax = lmax
        self.correlation = correlation[layer]
        self.num_channel = num_channel
        self.target_weight = target_weight
        self.num_elements = num_elements
        self.l1l2 = l1l2
        self.ictp_ictc_like = ictp_ictc_like
        self.use_bias = bias
        self.truncation = truncation
        self.trainable_scale = trainable_scale
        if num_hidden_channel is None:
            self.num_hidden_channel = self.num_channel
        else:
            self.num_hidden_channel = num_hidden_channel
        if self.truncation is None:
            self.truncation = self.correlation * self.lmax
        assert self.truncation >= self.lmax
        assert self.truncation <= self.correlation * self.lmax
        if num_latitude is None and num_longitude is None:
            self.num_latitude = 2 * (self.truncation + 1)
            self.num_longitude = 2 * (self.truncation+ 1) + 1
        else:
            self.num_latitude = num_latitude
            self.num_longitude = num_longitude  
        assert isinstance(self.num_latitude, int)    
        assert isinstance(self.num_longitude, int) 

        if self.correlation == 1:
            self.irreps_in = _to_full_so3_irreps(self.Lmax, self.num_hidden_channel)
        else:
            self.irreps_in = _to_full_so3_irreps(self.lmax, self.num_hidden_channel)
        if layer == num_layers-1:
            self.irreps_hidden = _to_full_so3_irreps(target_weight, self.num_hidden_channel)
            self.irreps_out = _to_full_so3_irreps(target_weight, self.num_channel)
        else:
            self.irreps_hidden = _to_full_so3_irreps(self.Lmax, self.num_hidden_channel)
            self.irreps_out = _to_full_so3_irreps(self.Lmax, self.num_channel)

            
        self._setup()
    
    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError


class ReadOut(torch.nn.Module):
    def __init__(
        self,
        layer: int,
        num_layers: int,
        Lmax: List[int],
        lmax: int,
        num_channel: int,
        hidden_channel: List[int], 
        target_weight: List[int],
        bias: bool,
        num_fidelities: int,
        l: int,
    ) -> None:
        super().__init__()

        self.layer = layer
        self.num_layers = num_layers
        self.num_channel = num_channel
        self.Lmax = Lmax[layer]
        self.lmax = lmax
        self.use_bias = bias
        self.num_fidelities = num_fidelities
        self.l = l
        self.scalar_act = 'silu'
        self.tensor_act = 'sigmoid'
        
        if layer == num_layers-1:
            self.irreps_in = _to_full_so3_irreps(target_weight, self.num_channel)
        else:
            self.irreps_in = _to_full_so3_irreps(self.Lmax, self.num_channel)
        self.irreps_out = _to_full_so3_irreps([l], num_fidelities)

        self.irreps_gates = [
            o3.Irreps([(c * self.num_fidelities, (0, 1))])
            for c in hidden_channel
        ]
        self.irreps_hidden = [
            o3.Irreps([(c * self.num_fidelities, (l, (-1)**l))])
            for c in hidden_channel
        ]

        self._setup()
    
    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError
