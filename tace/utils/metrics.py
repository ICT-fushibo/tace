################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List


import torch
from torch import Tensor
from torchmetrics import Metric


from ..dataset.quantity import (
    MAE_PROPERTY,
    RMSE_PROPERTY,
    MAE_PER_ATOM_PROPERTY,
    RMSE_PER_ATOM_PROPERTY,
)

SCALE = 1000.0  # for example, metric units from eV to meV


def expand_dims_to(T: torch.Tensor, n_dim: int, dim: int = -1) -> torch.Tensor:
    '''jit-safe'''
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T


class MAE(Metric):
    def __init__(self, scale: float = SCALE):
        super().__init__()
        self.scale = scale
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        abs_error = torch.abs(preds - targets)
        self.sum_abs_error += abs_error.sum()
        self.count += targets.numel()

    def compute(self):
        return (self.sum_abs_error / self.count) * self.scale


class RMSE(Metric):
    def __init__(self, scale: float = SCALE):
        super().__init__()
        self.scale = scale
        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        squared_error = (preds - targets) ** 2
        self.sum_squared_error += squared_error.sum()
        self.count += targets.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_error / self.count) * self.scale


class PerAtomMAE(MAE):
    def update(self, preds: Tensor, targets: Tensor, ptr: Tensor):
        num_nodes = ptr[1:] - ptr[:-1]  # [B]
        num_nodes = expand_dims_to(num_nodes, preds.ndim, dim=-1)
        preds_per_atom = preds / num_nodes
        targets_per_atom = targets / num_nodes
        super().update(preds_per_atom, targets_per_atom)


class PerAtomRMSE(RMSE):
    def update(self, preds: Tensor, targets: Tensor, ptr: Tensor):
        num_nodes = ptr[1:] - ptr[:-1]  # [B]
        num_nodes = expand_dims_to(num_nodes, preds.ndim, dim=-1)
        preds_per_atom = preds / num_nodes
        targets_per_atom = targets / num_nodes
        super().update(preds_per_atom, targets_per_atom)


class PolarizationMetric(Metric):
    def __init__(
        self, metric_type: str = "mae", per_atom: bool = False, scale: float = SCALE
    ):
        """
        Args:
            metric_type: "mae" or "rmse"
            per_atom: whether to normalize by number of atoms
            scale: scaling factor
        """
        super().__init__()
        assert metric_type in ["mae", "rmse"], "metric_type must be 'mae' or 'rmse'"
        self.metric_type = metric_type
        self.per_atom = per_atom
        self.scale = scale

        if self.metric_type == "mae":
            self.add_state(
                "sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:  # rmse
            self.add_state(
                "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor, label: Dict[str, Tensor]):
        lattice = label["lattice"]
        error = preds - targets

        # Project to lattice basis
        error = torch.einsum("bi, bij -> bj", error, torch.linalg.inv(lattice))
        error = torch.remainder(error, 1.0)
        error = torch.where(error > 0.5, error - 1.0, error)
        error = torch.where(error < -0.5, error + 1.0, error)
        error = torch.einsum("bi, bij -> bj", error, lattice)

        # Normalize per atom if needed
        if self.per_atom:
            ptr = label["ptr"]
            num_atoms = (ptr[1:] - ptr[:-1]).reshape(-1, 1)
            error = error / num_atoms

        if self.metric_type == "mae":
            abs_error = torch.abs(error)
            self.sum_abs_error += abs_error.sum()
        else:  # rmse
            squared_error = error**2
            self.sum_squared_error += squared_error.sum()

        self.count += targets.numel()

    def compute(self):
        if self.metric_type == "mae":
            return self.sum_abs_error / self.count * self.scale
        else:  # rmse
            return torch.sqrt(self.sum_squared_error / self.count) * self.scale

class FinalCollinearMagmomsMetric(Metric):
    def __init__(self, metric_type: str = "mae", scale: float = SCALE):
        """
        Args:
            metric_type: "mae" or "rmse"
            scale: scaling factor
        """
        super().__init__()
        assert metric_type in ["mae", "rmse"], "metric_type must be 'mae' or 'rmse'"
        self.metric_type = metric_type
        self.scale = scale

        if self.metric_type == "mae":
            self.add_state(
                "sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:  # rmse
            self.add_state(
                "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        error = abs(preds) - abs(targets)
        if self.metric_type == "mae":
            abs_error = torch.abs(error)
            self.sum_abs_error += abs_error.sum()
        else:  # rmse
            squared_error = error**2
            self.sum_squared_error += squared_error.sum()
        self.count += targets.numel()

    def compute(self):
        if self.metric_type == "mae":
            return self.sum_abs_error / self.count * self.scale
        else:  # rmse
            return torch.sqrt(self.sum_squared_error / self.count) * self.scale
        
class PartialHessiansMetric(Metric):
    def __init__(self, metric_type: str = "mae", scale: float = SCALE):
        """
        Args:
            metric_type: "mae" or "rmse"
            scale: scaling factor
        """
        super().__init__()
        assert metric_type in ["mae", "rmse"], "metric_type must be 'mae' or 'rmse'"
        self.metric_type = metric_type
        self.scale = scale

        if self.metric_type == "mae":
            self.add_state(
                "sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:  # rmse
            self.add_state(
                "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self, preds: Tensor, targets: Tensor):
        error = abs(preds) - abs(targets)
        if self.metric_type == "mae":
            abs_error = torch.abs(error)
            self.sum_abs_error += abs_error.sum()
        else:  # rmse
            squared_error = error**2
            self.sum_squared_error += squared_error.sum()
        self.count += targets.numel()

    def update(self, pred: Dict[str, torch.Tensor | List[torch.Tensor]], label: Dict[str, torch.Tensor]):

        true_hessian_flat = label["hessians"]
        num_atoms_per_graph = label["ptr"][1:] - label["ptr"][:-1]
        jacs_per_graph = pred["jacs_per_graph"]
        samples_per_graph = pred["samples_per_graph"]

        offset = 0
        error_list = []
        for jac_pred, samples, n_g in zip(jacs_per_graph, samples_per_graph, num_atoms_per_graph):
            hess_size = n_g * 3 * n_g * 3
            hess_flat_g = true_hessian_flat[offset : offset + hess_size]
            hess_true = hess_flat_g.reshape(n_g, 3, n_g, 3)
            offset += hess_size

            atom_idx = samples[:, 0]
            xyz_idx = samples[:, 1]
            jac_true = hess_true[atom_idx, xyz_idx]
            error = torch.square((jac_pred - jac_true)).view(-1)
            error_list.append(error)
        errors = torch.cat(error_list, dim=-1)

        if self.metric_type == "mae":
            abs_error = torch.abs(errors)
            self.sum_abs_error += abs_error.sum()
        else:  # rmse
            squared_error = errors**2
            self.sum_squared_error += squared_error.sum()
        self.count += errors.numel() 
   
    def compute(self):
        if self.metric_type == "mae":
            return self.sum_abs_error / self.count * self.scale
        else:  # rmse
            return torch.sqrt(self.sum_squared_error / self.count) * self.scale
        
def build_metrics(prefix: str, loss_property: List[str]) -> Dict[str, Metric]:
    metrics = {}

    def add_metrics(property_name):
        if property_name in MAE_PROPERTY:
            metrics[f"{prefix}/{property_name}_mae"] = MAE()
        if property_name in RMSE_PROPERTY:
            metrics[f"{prefix}/{property_name}_rmse"] = RMSE()
        if property_name in MAE_PER_ATOM_PROPERTY:
            metrics[f"{prefix}/{property_name}_per_atom_mae"] = PerAtomMAE()
        if property_name in RMSE_PER_ATOM_PROPERTY:
            metrics[f"{prefix}/{property_name}_per_atom_rmse"] = PerAtomRMSE()
        if property_name == "polarization":
            metrics[f"{prefix}/{property_name}_mae"] = PolarizationMetric("mae", False)
            metrics[f"{prefix}/{property_name}_rmse"] = PolarizationMetric("rmse", False)
            metrics[f"{prefix}/{property_name}_per_atom_mae"] = PolarizationMetric("mae", True)
            metrics[f"{prefix}/{property_name}_per_atom_rmse"] = PolarizationMetric("rmse", True)
        # if property_name == "final_collinear_magmoms":
        #     metrics[f"{prefix}/{property_name}_mae"] = FinalCollinearMagmomsMetric("mae")
        #     metrics[f"{prefix}/{property_name}_rmse"] = FinalCollinearMagmomsMetric("rmse")
        if property_name == "hessians":
            metrics[f"{prefix}/{property_name}_mae"] = PartialHessiansMetric("mae")
            metrics[f"{prefix}/{property_name}_rmse"] = PartialHessiansMetric("rmse")

    for p in loss_property:
        add_metrics(p)

    return metrics


def update_metrics(metrics, prefix, pred, label, loss_property):
    for p in loss_property:

        if p not in pred:
            continue

        ptr = label["ptr"]
        output_value = pred[p]
        batch_value = label[p]
        if p in MAE_PROPERTY:
            metrics[f"{prefix}/{p}_mae"](output_value, batch_value)
        if p in RMSE_PROPERTY:
            metrics[f"{prefix}/{p}_rmse"](output_value, batch_value)
        if p in MAE_PER_ATOM_PROPERTY:
            metrics[f"{prefix}/{p}_per_atom_mae"](output_value, batch_value, ptr)
        if p in RMSE_PER_ATOM_PROPERTY:
            metrics[f"{prefix}/{p}_per_atom_rmse"](output_value, batch_value, ptr)
        if p == "polarization":
            metrics[f"{prefix}/{p}_mae"](output_value, batch_value, label)
            metrics[f"{prefix}/{p}_rmse"](output_value, batch_value, label)
            metrics[f"{prefix}/{p}_per_atom_mae"](output_value, batch_value, label)
            metrics[f"{prefix}/{p}_per_atom_rmse"](output_value, batch_value, label)
        # if p == "final_collinear_magmoms":
        #     metrics[f"{prefix}/{p}_mae"](output_value, batch_value)
        #     metrics[f"{prefix}/{p}_rmse"](output_value, batch_value)
        if p == "hessians":
            metrics[f"{prefix}/{p}_mae"](pred, label)
            metrics[f"{prefix}/{p}_rmse"](pred, label)
