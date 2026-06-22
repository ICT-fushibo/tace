################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
# TODO

import abc 
from typing import Union


import torch


from ..mlp import FFN


def get_scatter_norm_layer(
        scatter_norm_type: str,
        avg_num_neighbors: float,
        edge_feats_channel: int,
        edge_info_type,
        radial_bias,
        radial_layer_norm,
    ):

    if scatter_norm_type == 'density':  # From MACE
        edge_density = FFN[edge_info_type](
            [edge_feats_channel, 64, 1],
            bias=radial_bias,
            layer_norm=radial_layer_norm,
            act='silu',
        )
        alpha = torch.nn.Parameter(torch.tensor(avg_num_neighbors))
        beta = torch.nn.Parameter(torch.tensor(0.0))
        return {
            'edge_density': edge_density,
            'alpha': alpha,
            'beta': beta,
        }

    if scatter_norm_type == 'no_cutoff_density': 
        edge_density = FFN[edge_info_type](
            [edge_feats_channel, 64, 1],
            bias=radial_bias,
            layer_norm=radial_layer_norm,
            act='silu',
        )
        alpha = torch.nn.Parameter(torch.tensor(avg_num_neighbors))
        beta = torch.nn.Parameter(torch.tensor(0.0))
        return {
            'edge_density': edge_density,
            'alpha': alpha,
            'beta': beta,
        }