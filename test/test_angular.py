# TODO


import torch
from e3nn.o3 import Irreps, SphericalHarmonics, FullyConnectedTensorProduct

class GeneralizedSphericalHarmonics(torch.nn.Module):
    def __init__(self, lmax: int, irreps_out: str):
        super().__init__()
        self.lmax = lmax
        self.irreps_out = Irreps(irreps_out)

        self.sh = SphericalHarmonics(
            Irreps.spherical_harmonics(lmax, p=-1),  
            normalize=False,
            normalization="component"
        )

        irreps_in = Irreps.spherical_harmonics(lmax, p=-1)
        self.tp = FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)

    def forward(
            self, 
            # node_attrs: torch.Tensor,
            edge_vector: torch.Tensor, 
            edge_length: torch.Tensor, 
            edge_index: torch.Tensor,
            num_nodes: int,
        ):
        row, col = edge_index 
        Y_edge = self.sh(edge_vector)

        sorted_idx = torch.argsort(col)
        edge_vector = edge_vector[sorted_idx]
        edge_length = edge_length[sorted_idx]
        Y_edge = Y_edge[sorted_idx]
        row = row[sorted_idx]
        col = col[sorted_idx]

        # num_nodes = node_attrs.size(0)
        edge_counts = torch.bincount(col, minlength=num_nodes)
        edge_cumsum = torch.cumsum(edge_counts, dim=0)
        start_idx = torch.cat([torch.tensor([0], device=edge_vector.device), edge_cumsum[:-1]])
        end_idx = edge_cumsum

        pair_idx_i, pair_idx_j, pair_centers = [], [], []
        for s, e, c in zip(start_idx, end_idx, torch.arange(num_nodes, device=edge_vector.device)):
            n = e - s
            if n < 2:
                continue
            idx = torch.arange(s, e, device=edge_vector.device)
            comb = torch.combinations(idx, r=2)
            pair_idx_i.append(comb[:,0])
            pair_idx_j.append(comb[:,1])
            pair_centers.append(torch.full((comb.shape[0],), c, device=edge_vector.device, dtype=torch.long))

        pair_idx_i = torch.cat(pair_idx_i)
        pair_idx_j = torch.cat(pair_idx_j)
        pair_centers = torch.cat(pair_centers)

        Y_i = Y_edge[pair_idx_i]
        Y_j = Y_edge[pair_idx_j]
        Y_pair = self.tp(Y_i, Y_j)

        return pair_centers, pair_idx_i, pair_idx_j, Y_pair
    
