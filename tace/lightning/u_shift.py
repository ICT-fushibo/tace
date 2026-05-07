import torch


from tace.utils.torch_scatter import scatter_sum


U_SHIFT = {
    23: -2.271,  # V
    24: -2.458,  # Cr
    25: -1.602,  # Mn
    26: -1.925,  # Fe
    27: -2.004,  # Co
    28: -2.586,  # Ni
    42: -4.895,  # Mo
    74: -5.875,  # W
}

def apply_u_shift(batch, energy):

    device = energy.device
    node_attrs = batch["node_attrs"]  
    atomic_numbers = torch.tensor([
        1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,
        31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,
        51,52,53,54,55,56,57,58,59,60,
        61,62,63,64,65,66,67,68,69,70,
        71,72,73,74,75,76,77,78,79,80,
        81,82,83,89,90,91,92,93,94
    ], device=device)

    delta_E = torch.zeros_like(atomic_numbers, dtype=energy.dtype)
    for i, Z in enumerate(atomic_numbers.tolist()):
        if Z in U_SHIFT:
            delta_E[i] = U_SHIFT[Z]

    per_atom_shift = node_attrs @ delta_E  # [N]

    batch_idx = batch['batch']
    graph_shift = scatter_sum(per_atom_shift, batch_idx, dim=0)  # [num_graphs]

    is_O = (atomic_numbers == 8).nonzero(as_tuple=True)[0].item()
    is_F = (atomic_numbers == 9).nonzero(as_tuple=True)[0].item()

    has_O_atom = node_attrs[:, is_O] > 0
    has_F_atom = node_attrs[:, is_F] > 0

    has_O = scatter_sum(has_O_atom.float(), batch_idx, dim=0) > 0
    has_F = scatter_sum(has_F_atom.float(), batch_idx, dim=0) > 0

    mask = has_O | has_F  # [num_graphs]

    graph_shift = graph_shift * mask

    energy = energy + graph_shift

    return energy