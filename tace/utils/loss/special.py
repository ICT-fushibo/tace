################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch


from .fn import *


class OMat24sAlexMPtrjLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        stress_weight=1.0,
        energy_huber_delta=0.01,
        forces_huber_delta=0.01,
        stress_huber_delta=0.01,
        normalize: bool = False,
        **kwargs,
    ) -> None:
        "Modify code From MACE, see same logic as NequIP"
        super().__init__()
        if normalize:
            normalizer = energy_weight + forces_weight + stress_weight
            self.register_buffer(
                "energy_weight",
                torch.tensor(energy_weight / normalizer , dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "forces_weight",
                torch.tensor(forces_weight / normalizer, dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "stress_weight",
                torch.tensor(stress_weight / normalizer, dtype=torch.get_default_dtype()),
            )
        else:
            self.register_buffer(
                "energy_weight",
                torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "forces_weight",
                torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "stress_weight",
                torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
            )
        self.energy_huber_delta = energy_huber_delta
        self.forces_huber_delta = forces_huber_delta
        self.stress_huber_delta = stress_huber_delta

    def forward(self, pred, label) -> Tensor:
        num_atoms = (label.ptr[1:] - label.ptr[:-1])
        batch = label["batch"]
        energy_weight = label.energy_weight
        forces_weight = label.forces_weight[batch].unsqueeze(-1)
        stress_weight = label.stress_weight.unsqueeze(-1).unsqueeze(-1)

        loss_energy = torch.nn.functional.huber_loss(
            energy_weight * label["energy"] / num_atoms,
            energy_weight * pred["energy"] / num_atoms,
            reduction="mean",
            delta=self.energy_huber_delta,
        )
        loss_forces = conditional_huber_forces(
            forces_weight * pred["forces"],
            forces_weight * label["forces"],
            huber_delta=self.forces_huber_delta,
        )
        loss_stress = torch.nn.functional.huber_loss(
            stress_weight * label["stress"],
            stress_weight * pred["stress"],
            reduction="mean",
            delta=self.stress_huber_delta,
        )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  energy_weight        = {self.energy_weight:.3f},\n"
            f"  forces_weight        = {self.forces_weight:.3f},\n"
            f"  stress_weight        = {self.stress_weight:.3f},\n"
            f"  energy_huber_delta   = {self.energy_huber_delta:.3f},\n"
            f"  forces_huber_delta   = {self.forces_huber_delta:.3f},\n"
            f"  stress_huber_delta   = {self.stress_huber_delta:.3f},\n"
            f")"
        )


class AQCat25OC20Loss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        energy_huber_delta=0.01,
        forces_huber_delta=0.01,
        normalize: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if normalize:
            normalizer = energy_weight + forces_weight
            self.register_buffer(
                "energy_weight",
                torch.tensor(energy_weight / normalizer , dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "forces_weight",
                torch.tensor(forces_weight / normalizer, dtype=torch.get_default_dtype()),
            )
        else:
            self.register_buffer(
                "energy_weight",
                torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "forces_weight",
                torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
            )
        self.energy_huber_delta = energy_huber_delta
        self.forces_huber_delta = forces_huber_delta

    def forward(self, pred, label) -> Tensor:
        num_atoms = (label.ptr[1:] - label.ptr[:-1])
        batch = label["batch"]
        energy_weight = label.energy_weight
        forces_weight = label.forces_weight[batch].unsqueeze(-1)

        loss_energy = torch.nn.functional.huber_loss(
            energy_weight * label["energy"] / num_atoms,
            energy_weight * pred["energy"] / num_atoms,
            reduction="mean",
            delta=self.energy_huber_delta,
        )
        loss_forces = conditional_huber_forces(
            forces_weight * pred["forces"],
            forces_weight * label["forces"],
            huber_delta=self.forces_huber_delta,
        )

        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  energy_weight        = {self.energy_weight:.3f},\n"
            f"  forces_weight        = {self.forces_weight:.3f},\n"
            f"  energy_huber_delta   = {self.energy_huber_delta:.3f},\n"
            f"  forces_huber_delta   = {self.forces_huber_delta:.3f},\n"
            f")"
        )
