import math
import random
from dataclasses import dataclass

import numpy as np
import torch

from tace.models._e3nn.asymmetric_contraction import ComplexProductBasis
from tace.models.so2 import uvSO2Linear
from tace.models.so2.utils import rotate_uuu_so2_features
from tace.utils.torch_scatter import scatter_sum


DTYPE = torch.float64

@dataclass
class RotSymData:
    positions: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    center_nodes: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device | str) -> "RotSymData":
        return RotSymData(
            positions=self.positions.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            ptr=self.ptr.to(device),
            center_nodes=self.center_nodes.to(device),
            labels=self.labels.to(device),
        )

    @property
    def num_graphs(self) -> int:
        return int(self.ptr.numel() - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_rotsym_data(fold: int = 5) -> RotSymData:
    if fold < 2:
        raise ValueError("fold must be at least 2")

    rotation_offsets = (0.0, 2.0 * math.pi / (fold + 1))
    positions = []
    senders = []
    receivers = []
    batch = []
    ptr = [0]
    center_nodes = []

    for graph_index, rotation_offset in enumerate(rotation_offsets):
        node_offset = len(positions)
        center_nodes.append(node_offset)
        positions.append((0.0, 0.0, 0.0))
        batch.append(graph_index)

        for spoke in range(fold):
            angle = rotation_offset + 2.0 * math.pi * spoke / fold
            positions.append((math.cos(angle), math.sin(angle), 0.0))
            batch.append(graph_index)
            senders.append(node_offset + spoke + 1)
            receivers.append(node_offset)
        ptr.append(len(positions))

    return RotSymData(
        positions=torch.tensor(positions, dtype=DTYPE),
        edge_index=torch.tensor([senders, receivers], dtype=torch.long),
        batch=torch.tensor(batch, dtype=torch.long),
        ptr=torch.tensor(ptr, dtype=torch.long),
        center_nodes=torch.tensor(center_nodes, dtype=torch.long),
        labels=torch.tensor([0, 1], dtype=torch.long),
    )


class SO2EdgeEmbedding(torch.nn.Module):
    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        input_order: int,
        channels: int,
    ) -> None:
        super().__init__()
        if input_order >= mmax:
            raise ValueError("input_order must be smaller than mmax for this test")

        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.input_order = int(input_order)
        self.channels = int(channels)
        self.num_components = (self.lmax + 1) * (1 + 2 * self.mmax)
        padded_components = [self.lmax + 1] * (self.mmax + 1)
        self.linear = uvSO2Linear(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel_in=1,
            num_channel_out=self.channels,
            num_components_in=padded_components,
            num_components_out=padded_components,
            weight_type="w1_w2",
        )

    def forward(self, edge_vectors: torch.Tensor) -> torch.Tensor:
        angle = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
        n = self.lmax + 1
        blocks = [torch.ones(angle.shape[0], n, 1, device=angle.device, dtype=angle.dtype)]

        for m in range(1, self.mmax + 1):
            if m <= self.input_order:
                real = torch.cos(m * angle).view(-1, 1, 1).expand(-1, n, -1)
                imag = torch.sin(m * angle).view(-1, 1, 1).expand(-1, n, -1)
            else:
                real = angle.new_zeros(angle.shape[0], n, 1)
                imag = angle.new_zeros(angle.shape[0], n, 1)
            blocks.extend((real, imag))

        return self.linear(torch.cat(blocks, dim=1))


class OneLayerRotSymModel(torch.nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        fold: int,
        input_order: int,
        channels: int,
    ) -> None:
        super().__init__()
        if mode not in {"node_product", "edge_linear", "edge_product"}:
            raise ValueError(f"unknown mode: {mode}")

        self.mode = mode
        self.fold = int(fold)
        self.lmax = int(fold)
        self.mmax = int(fold)
        self.channels = int(channels)

        self.edge_embedding = SO2EdgeEmbedding(
            lmax=self.lmax,
            mmax=self.mmax,
            input_order=input_order,
            channels=channels,
        )
        self.product = ComplexProductBasis(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=channels,
            num_elements=1,
            m1m2="<=",
            agnostic=True,
        )
        descriptor_dim = self.edge_embedding.num_components * channels
        self.readout = torch.nn.Linear(descriptor_dim, 2, bias=False)

    def _edge_and_environment(
        self,
        data: RotSymData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sender, receiver = data.edge_index
        edge_vectors = data.positions[sender] - data.positions[receiver]
        edge_features = self.edge_embedding(edge_vectors)

        node_environment = scatter_sum(
            src=edge_features,
            index=receiver,
            dim=0,
            dim_size=data.positions.shape[0],
        )
        node_environment = node_environment / math.sqrt(self.fold)
        return edge_features, node_environment, receiver

    def descriptor(self, data: RotSymData) -> torch.Tensor:
        edge_features, node_environment, receiver = self._edge_and_environment(data)

        if self.mode == "node_product":
            center_environment = node_environment[data.center_nodes]
            return self.product(center_environment, None, None).flatten(1)

        # Gathering alone cannot restore angular information removed by the sum.
        # The edge anchor retains the individual direction before the final sum.

        # edge_input = edge_features + node_environment[receiver]
        edge_input = edge_features
        if self.mode == "edge_product":
            edge_input = self.product(edge_input, None, data.edge_index)

        edge_batch = data.batch[receiver]
        edge_descriptor = scatter_sum(
            src=edge_input,
            index=edge_batch,
            dim=0,
            dim_size=data.num_graphs,
        ) / math.sqrt(self.fold)
        return edge_descriptor.flatten(1)

    def forward(self, data: RotSymData) -> torch.Tensor:
        if self.mode == "node_product":
            return self.readout(self.descriptor(data))

        edge_features, node_environment, receiver = self._edge_and_environment(data)
        edge_input = edge_features + node_environment[receiver]
        if self.mode == "edge_product":
            edge_input = self.product(edge_input, None, data.edge_index)

        edge_logits = self.readout(edge_input.flatten(1))
        edge_batch = data.batch[receiver]
        return scatter_sum(
            src=edge_logits,
            index=edge_batch,
            dim=0,
            dim_size=data.num_graphs,
        ) / math.sqrt(self.fold)


def train_once(
    *,
    mode: str,
    data: RotSymData,
    fold: int,
    input_order: int,
    channels: int,
    epochs: int,
    seed: int,
) -> tuple[float, float]:
    set_seed(seed)
    model = OneLayerRotSymModel(
        mode=mode,
        fold=fold,
        input_order=input_order,
        channels=channels,
    ).to(device=data.positions.device, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)

    with torch.no_grad():
        descriptor = model.descriptor(data)
        descriptor_distance = torch.linalg.vector_norm(descriptor[0] - descriptor[1]).item()

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(data), data.labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        accuracy = (model(data).argmax(dim=-1) == data.labels).float().mean().item()
    return descriptor_distance, accuracy


def run_experiment(
    *,
    fold: int = 5,
    input_order: int = 3,
    channels: int = 4,
    epochs: int = 300,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    device: str | None = None,
) -> dict[str, dict[str, float]]:
    if 2 * input_order < fold:
        raise ValueError("correlation=2 cannot reach the requested fold frequency")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data = create_rotsym_data(fold).to(selected_device)
    results = {}

    print(
        f"RotSym edge-product test: fold={fold}, input_order={input_order}, "
        f"channels={channels}, device={selected_device}"
    )
    for mode in ("node_product", "edge_linear", "edge_product"):
        distances = []
        accuracies = []
        for seed in seeds:
            distance, accuracy = train_once(
                mode=mode,
                data=data,
                fold=fold,
                input_order=input_order,
                channels=channels,
                epochs=epochs,
                seed=seed,
            )
            distances.append(distance)
            accuracies.append(accuracy)

        results[mode] = {
            "descriptor_distance_mean": float(np.mean(distances)),
            "accuracy_mean": float(np.mean(accuracies)),
        }
        print(
            f"{mode:<14} | descriptor distance={np.mean(distances):.6e} "
            f"| accuracy={100.0 * np.mean(accuracies):6.2f}%"
        )
    return results


def test_edge_complex_product_recovers_fold_frequency() -> None:
    results = run_experiment(epochs=200, seeds=(0, 1), device="cpu")
    assert results["node_product"]["descriptor_distance_mean"] < 1.0e-4
    assert results["edge_linear"]["descriptor_distance_mean"] < 1.0e-4
    assert results["edge_product"]["descriptor_distance_mean"] > 1.0e-3
    assert results["edge_product"]["accuracy_mean"] == 1.0


def test_so2_edge_embedding_is_equivariant() -> None:
    set_seed(0)
    lmax = 5
    mmax = 5
    channels = 4
    rotation_angle = 0.37
    embedding = SO2EdgeEmbedding(
        lmax=lmax,
        mmax=mmax,
        input_order=3,
        channels=channels,
    ).to(dtype=DTYPE)

    edge_vectors = torch.randn(16, 3, dtype=DTYPE)
    edge_vectors[:, 2] = 0.0
    cosine = math.cos(rotation_angle)
    sine = math.sin(rotation_angle)
    rotation = edge_vectors.new_tensor(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_vectors = edge_vectors @ rotation.T
    actual = embedding(rotated_vectors)
    expected = rotate_uuu_so2_features(
        embedding(edge_vectors),
        rotation_angle,
        lmax,
        mmax,
        channels,
    )
    torch.testing.assert_close(actual, expected, atol=1.0e-10, rtol=1.0e-10)


if __name__ == "__main__":
    test_so2_edge_embedding_is_equivariant()
    experiment_results = run_experiment()
    if experiment_results["edge_product"]["accuracy_mean"] < 1.0:
        raise RuntimeError("edge product did not consistently separate the two orientations")
