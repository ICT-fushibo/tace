import math
import random
from dataclasses import dataclass

import numpy as np
import torch
DTYPE = torch.float64
PERMUTE_M0 = True
torch.set_default_dtype(DTYPE)
from e3nn import o3

from tace.models._e3nn.asymmetric_contraction import ComplexProductBasis
from tace.models.radial import j0SphericalBesselBasis
from tace.models.so2 import WignerD
from tace.utils.torch_scatter import scatter_sum


@dataclass
class IncompletenessData:
    positions: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    graph_labels: torch.Tensor

    def to(self, device: torch.device | str) -> "IncompletenessData":
        return IncompletenessData(
            positions=self.positions.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            graph_labels=self.graph_labels.to(device),
        )

    @property
    def num_graphs(self) -> int:
        return int(self.graph_labels.numel())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_star_environments(environments: list[torch.Tensor]) -> IncompletenessData:
    positions = []
    senders = []
    receivers = []
    batch = []
    for graph_index, environment in enumerate(environments):
        offset = len(positions)
        positions.extend(environment)
        batch.extend([graph_index] * environment.shape[0])
        for neighbor in range(1, environment.shape[0]):
            senders.append(offset + neighbor)
            receivers.append(offset)
    return IncompletenessData(
        positions=torch.stack(positions).to(dtype=DTYPE),
        edge_index=torch.tensor([senders, receivers], dtype=torch.long),
        batch=torch.tensor(batch, dtype=torch.long),
        graph_labels=torch.tensor([0, 1], dtype=torch.long),
    )


def create_two_body_counterexample() -> IncompletenessData:
    return _batch_star_environments(
        [
            torch.tensor([[0, 0, 0], [5, 0, 0], [3, 0, 4]]),
            torch.tensor([[0, 0, 0], [5, 0, 0], [-5, 0, 0]]),
        ]
    )


def create_three_body_counterexample() -> IncompletenessData:
    common = [[0, 0, 0], [5, 0, 5], [5, 5, 5], [-5, -5, 5]]
    return _batch_star_environments(
        [
            torch.tensor(common + [[0, 5, 5]]),
            torch.tensor(common + [[0, -5, 5]]),
        ]
    )


def create_four_body_counterexample() -> IncompletenessData:
    rotation = o3.matrix_y(torch.tensor(math.pi / 10, dtype=DTYPE))
    a = torch.tensor([[3, 2, -4], [0, 2, 5], [0, 2, -5]], dtype=DTYPE)
    b = torch.tensor([[3, -2, -4], [0, -2, 5], [0, -2, -5]], dtype=DTYPE)
    b = b @ rotation
    common = torch.cat((torch.zeros(1, 3, dtype=DTYPE), a, b), dim=0)
    return _batch_star_environments(
        [
            torch.cat((common, torch.tensor([[0, 5, 0]], dtype=DTYPE))),
            torch.cat((common, torch.tensor([[0, -5, 0]], dtype=DTYPE))),
        ]
    )


COUNTEREXAMPLES = {
    "two_body": create_two_body_counterexample,
    "three_body": create_three_body_counterexample,
    "four_body": create_four_body_counterexample,
}


class RadialSphericalHarmonicDensity(torch.nn.Module):
    def __init__(self, lmax: int, channels: int, cutoff: float = 10.0) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.cutoff = float(cutoff)
        self.angular = o3.SphericalHarmonics(
            list(range(self.lmax + 1)),
            normalize=True,
            normalization="component",
        )
        self.radial = j0SphericalBesselBasis(
            cutoff=self.cutoff,
            num_basis=self.channels,
        )

    def forward(self, edge_vectors: torch.Tensor) -> torch.Tensor:
        distances = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True)
        radial = self.radial(distances, None, None)
        angular = self.angular(edge_vectors)
        return angular.unsqueeze(-1) * radial.unsqueeze(1)


def compact_to_padded_so2(x: torch.Tensor, lmax: int) -> torch.Tensor:
    batch, _, channels = x.shape
    n = lmax + 1
    fields = [x[:, :n]]
    offset = n
    for m in range(1, lmax + 1):
        width = n - m
        for _ in range(2):
            field = x.new_zeros(batch, n, channels)
            field[:, m:] = x[:, offset : offset + width]
            fields.append(field)
            offset += width
    return torch.cat(fields, dim=1)


def padded_to_compact_so2(x: torch.Tensor, lmax: int) -> torch.Tensor:
    n = lmax + 1
    fields = [x[:, :n]]
    offset = n
    for m in range(1, lmax + 1):
        for _ in range(2):
            fields.append(x[:, offset + m : offset + n])
            offset += n
    return torch.cat(fields, dim=1)


class ChannelWiseM0Permutation(torch.nn.Module):
    """Move different local m=0 degrees to l=0 in different channels."""

    def __init__(self, lmax: int, channels: int) -> None:
        super().__init__()
        n = lmax + 1
        channel = torch.arange(channels)
        output_l = torch.arange(n).unsqueeze(1)
        permutation = (output_l + channel.unsqueeze(0)) % n
        self.register_buffer("permutation", permutation, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, channels = self.permutation.shape
        m0 = x[:, :n]
        index = self.permutation.unsqueeze(0).expand(x.shape[0], -1, -1)
        m0 = torch.gather(m0, dim=1, index=index)
        return torch.cat((m0, x[:, n:]), dim=1)


class RecursiveComplexProductBasis(torch.nn.Module):
    """Recursive complex products with a fixed, parameter-free m=0 permutation."""

    def __init__(self, lmax: int, channels: int, correlation: int) -> None:
        super().__init__()
        if correlation < 1:
            raise ValueError("correlation must be positive")
        self.lmax = int(lmax)
        self.products = torch.nn.ModuleList(
            [
                ComplexProductBasis(
                    mmax=lmax,
                    lmax=lmax,
                    num_channel=channels,
                    num_elements=1,
                    m1m2=None,
                    agnostic=True,
                )
                for _ in range(correlation - 1)
            ]
        )
        self.permute_m0 = ChannelWiseM0Permutation(lmax, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base = compact_to_padded_so2(x, self.lmax)
        # current = base
        # output = base
        # for product in self.products:
        #     current = product.tp(current, base)
        #     output = (output + current) # / math.sqrt(2.0)
        # if PERMUTE_M0:
        #     output = self.permute_m0(output)
        # return padded_to_compact_so2(output, self.lmax)


        x = compact_to_padded_so2(x, self.lmax)
        if PERMUTE_M0:
            output = self.permute_m0(x)
        return padded_to_compact_so2(output, self.lmax)

class EdgeBodyOrderProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        correlation: int | None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.density = RadialSphericalHarmonicDensity(lmax, channels)
        self.wigner = WignerD(
            lmax=lmax,
            mmax=lmax,
        )
        if correlation is None:
            self.product = None
        else:
            self.product = RecursiveComplexProductBasis(
                lmax, channels, correlation
            )
        self.readout = torch.nn.Linear(channels, 2)

    def edge_descriptor(
        self, data: IncompletenessData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sender, receiver = data.edge_index
        edge_vectors = data.positions[sender] - data.positions[receiver]

        edge_density = self.density(edge_vectors)
        node_density = scatter_sum(
            edge_density,
            receiver,
            dim=0,
            dim_size=data.positions.shape[0],
        )

        wigner, wigner_inv = self.wigner.get_wigner(edge_vectors)
        local_density = torch.bmm(wigner, node_density[receiver])
        if self.product is not None:
            local_density = self.product(local_density)
        scalar_wigner_inv = wigner_inv.narrow(1, 0, 1)
        global_scalar = torch.bmm(scalar_wigner_inv, local_density)
        invariant = global_scalar[:, 0]
        return invariant, data.batch[receiver]

    def forward(self, data: IncompletenessData) -> torch.Tensor:
        edge_descriptor, edge_batch = self.edge_descriptor(data)
        edge_logits = self.readout(edge_descriptor)
        graph_logits = scatter_sum(
            edge_logits,
            edge_batch,
            dim=0,
            dim_size=data.num_graphs,
        )
        edge_counts = torch.bincount(
            edge_batch, minlength=data.num_graphs
        ).to(edge_logits).unsqueeze(-1)
        return graph_logits / edge_counts


def train_once(
    data: IncompletenessData,
    *,
    correlation: int | None,
    lmax: int,
    channels: int,
    epochs: int,
    seed: int,
) -> tuple[float, float]:
    set_seed(seed)
    model = EdgeBodyOrderProbe(
        lmax=lmax,
        channels=channels,
        correlation=correlation,
    ).to(device=data.positions.device, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-3)

    with torch.no_grad():
        edge_descriptor, edge_batch = model.edge_descriptor(data)
        graph_descriptor = scatter_sum(
            edge_descriptor, edge_batch, dim=0, dim_size=data.num_graphs
        )
        counts = torch.bincount(edge_batch, minlength=data.num_graphs).to(
            graph_descriptor
        ).unsqueeze(-1)
        graph_descriptor = graph_descriptor / counts
        distance = torch.linalg.vector_norm(
            graph_descriptor[0] - graph_descriptor[1]
        ).item()

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = torch.nn.functional.cross_entropy(logits, data.graph_labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        accuracy = (
            model(data).argmax(dim=-1) == data.graph_labels
        ).float().mean().item()
    return distance, accuracy


def run_experiment(
    *,
    lmax: int = 3,
    channels: int = 8,
    max_correlation: int = 3,
    epochs: int = 400,
    seeds: tuple[int, ...] = (0, 1, 2),
    device: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    results = {}
    for name, create_data in COUNTEREXAMPLES.items():
        data = create_data().to(selected_device)
        results[name] = {}
        print(f"\n{name}: device={selected_device}")
        for correlation in (None, *range(1, max_correlation + 1)):
            variant = (
                "no_product"
                if correlation is None
                else f"product_correlation_{correlation}"
            )
            distances = []
            accuracies = []
            for seed in seeds:
                distance, accuracy = train_once(
                    data,
                    correlation=correlation,
                    lmax=lmax,
                    channels=channels,
                    epochs=epochs,
                    seed=seed,
                )
                distances.append(distance)
                accuracies.append(accuracy)
            body_order = 2 if correlation is None else correlation + 2
            results[name][variant] = {
                "body_order": float(body_order),
                "descriptor_distance_mean": float(np.mean(distances)),
                "accuracy_mean": float(np.mean(accuracies)),
            }
            print(
                f"{variant:<24} (edge body order={body_order}) | "
                f"distance={np.mean(distances):.6e} | "
                f"accuracy={100.0 * np.mean(accuracies):6.2f}%"
            )
    return results


def graph_descriptor(
    model: EdgeBodyOrderProbe,
    data: IncompletenessData,
) -> torch.Tensor:
    edge_descriptor, edge_batch = model.edge_descriptor(data)
    descriptor = scatter_sum(
        edge_descriptor, edge_batch, dim=0, dim_size=data.num_graphs
    )
    counts = torch.bincount(edge_batch, minlength=data.num_graphs).to(
        descriptor
    ).unsqueeze(-1)
    return descriptor / counts


def descriptor_distance(
    create_data,
    *,
    correlation: int | None,
    seed: int = 0,
) -> float:
    set_seed(seed)
    data = create_data()
    model = EdgeBodyOrderProbe(
        lmax=3,
        channels=8,
        correlation=correlation,
    ).to(dtype=DTYPE)
    with torch.no_grad():
        descriptor = graph_descriptor(model, data)
    return torch.linalg.vector_norm(descriptor[0] - descriptor[1]).item()


def test_recursive_complex_product_recovers_body_order_hierarchy() -> None:
    assert descriptor_distance(create_two_body_counterexample, correlation=None) < 1.0e-9
    assert descriptor_distance(create_two_body_counterexample, correlation=1) > 1.0e-3

    assert descriptor_distance(create_three_body_counterexample, correlation=1) < 1.0e-9
    assert descriptor_distance(create_three_body_counterexample, correlation=2) > 1.0e-3

    assert descriptor_distance(create_four_body_counterexample, correlation=2) < 1.0e-9
    assert descriptor_distance(create_four_body_counterexample, correlation=3) > 1.0e-3


def test_local_so2_edge_descriptor_is_rotation_invariant() -> None:
    set_seed(17)
    data = create_four_body_counterexample()
    matrix = torch.randn(3, 3, dtype=DTYPE)
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotated = IncompletenessData(
        positions=data.positions @ rotation.T,
        edge_index=data.edge_index,
        batch=data.batch,
        graph_labels=data.graph_labels,
    )

    model = EdgeBodyOrderProbe(
        lmax=3,
        channels=8,
        correlation=3,
    ).to(dtype=DTYPE)
    with torch.no_grad():
        actual, _ = model.edge_descriptor(rotated)
        expected, _ = model.edge_descriptor(data)
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)


if __name__ == "__main__":
    run_experiment()
