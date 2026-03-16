################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import abc 
from typing import Optional, List, Union, Tuple, Any, Dict


import torch
import equitorch as eqt


from ..lammps import eqtGhostExchangeMixin


def _to_full_so3_irreps(lmax: int | List[int]) -> str:
    if isinstance(lmax, int):
        return eqt.irreps.Irreps("+".join(f"1x{l}e" for l in range(lmax + 1)))
    assert isinstance(lmax, List)
    return eqt.irreps.Irreps("+".join(f"1x{l}e" for l in lmax))


def _parity(l: int) -> str:
    return "e" if l % 2 == 0 else "o"


def _to_sparse_o3_irreps(lmax: Union[int, List[int]]):
    if isinstance(lmax, int):
        ls = range(lmax + 1)
    else:
        ls = lmax

    return eqt.irreps.Irreps(
        "+".join(f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in ls)
    )


def _to_full_o3_irreps(lmax: int | List[int]) -> str:
    if isinstance(lmax, int):
        return eqt.irreps.Irreps("+".join(f"1x{l}o+1x{l}e" for l in range(lmax + 1)))
    assert isinstance(lmax, List)
    return eqt.irreps.Irreps("+".join(f"1x{l}o+1x{l}e" for l in lmax))


def _to_o3_possible_irreps(Lmax, lmax):
    ir_list = set()
    for ir1 in _to_sparse_o3_irreps(Lmax):
        for ir2 in _to_sparse_o3_irreps(lmax):
            for ir3 in ir1 * ir2:
                if ir3.l <= lmax:
                    ir_list.add(str(ir3))
    return eqt.irreps.Irreps("+".join(list(ir_list))).simplified()


class NodeEmbedding(torch.nn.Module):
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
    

class Interaction(torch.nn.Module, eqtGhostExchangeMixin):
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
        edge_embedding: Dict,
        target_weight: List[int],
        num_radial_basis: int,
        radial_mlp: List[int],
        radial_bias: bool,
        resnet: str,
        l1l2: Optional[str] = None,
        norm: str = 'avg_num_neighbors',
        ictp_ictc_like: bool = True,
        bias: bool = True,
        nonlinear: Optional[str] = None,
        has_linear_after_nonlinear: bool = True,
        # num_experts: Optional[int] = None,
        # top_k: Optional[int] = None,
        # aux_loss_weight: float = 1e-5,
        # z_loss_weight: float = 1e-3,
        # num_heads: int = 8,
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
        self.resnet = resnet
        self.norm = norm
        self.nonlinear = nonlinear
        self.radial_act = 'silu'
        self.radial_layer_norm = False
        self.has_linear_after_nonlinear = has_linear_after_nonlinear
        if self.edge_feats_channel != self.num_radial_basis:    
            self.radial_layer_norm = True
        self.edge_embedding = edge_embedding
        # self.top_k = top_k
        # self.num_experts = num_experts
        # self.use_moe = self.num_experts is not None and self.top_k is not None
        # self.aux_loss_weight = aux_loss_weight
        # self.z_loss_weight = z_loss_weight
        # self.num_heads = num_heads
        # self.head_channel = self.num_channel // self.num_heads

        self.irreps_sh = _to_full_so3_irreps(self.lmax)
        if self.correlation == 1:
            if self.layer == self.num_layers-1:
                self.irreps_out = _to_full_so3_irreps(target_weight)
            else:
                self.irreps_out = _to_full_so3_irreps(self.Lmax)        
        else:
            self.irreps_out = _to_full_so3_irreps(self.lmax)
        if layer == 0:
            self.irreps_in = eqt.irreps.Irreps('0e')
        else:
            self.irreps_in = _to_full_so3_irreps(self.Lmax)
        if self.layer == num_layers - 1:
            self.irreps_sc = _to_full_so3_irreps(target_weight)
        else:
            self.irreps_sc = _to_full_so3_irreps(self.Lmax)

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
            if self.layer == num_layers - 1:
                self.irreps_in = _to_full_so3_irreps(target_weight)
            else:
                self.irreps_in = _to_full_so3_irreps(self.Lmax)
        else:
            self.irreps_in = _to_full_so3_irreps(self.lmax)
        if layer == num_layers-1:
            self.irreps_out = _to_full_so3_irreps(target_weight)
        else:
            self.irreps_out = _to_full_so3_irreps(self.Lmax)
        
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
        self.bias = bias
        self.num_fidelities = num_fidelities
        self.l = l
        self.hidden = hidden_channel[0]
        self.scalar_act = 'silu'
        self.tensor_act = 'sigmoid'

        if layer == num_layers-1:
            self.irreps_in = _to_full_so3_irreps(target_weight)
        else:
            self.irreps_in = _to_full_so3_irreps(self.Lmax)
        self.irreps_out = _to_full_so3_irreps(target_weight)


        self._setup()
    
    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError
