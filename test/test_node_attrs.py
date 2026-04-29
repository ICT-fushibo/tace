import torch
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'

E = 10_000      # num edges
Z = 89          # one-hot dim
I = 128         # output dim
N = 4000        # num nodes

torch.manual_seed(0)


edge_index = torch.randint(
    0, N, (2, E),
    device=device
)

atom_type = torch.randint(
    0, Z, (N,),
    device=device
)

y = torch.nn.functional.one_hot(
    atom_type,
    num_classes=Z
).float()

source_coefs = torch.randn(
    Z, I,
    device=device
)

target_coefs = torch.randn(
    Z, I,
    device=device
)

for _ in range(100):

    # einsum
    w1 = torch.einsum(
        'bz,zi->bi',
        y[edge_index[0]],
        source_coefs
    )

    w2 = torch.einsum(
        'bz,zi->bi',
        y[edge_index[1]],
        target_coefs
    )

    w = w1 + w2

    # gather
    w_fast = (
        source_coefs[atom_type[edge_index[0]]]
        + target_coefs[atom_type[edge_index[1]]]
    )

if device == 'cuda':
    torch.cuda.synchronize()

# ============================================================
# Benchmark: einsum
# ============================================================

n_iter = 10000

start = time.time()

for _ in range(n_iter):

    w1 = torch.einsum(
        'bz,zi->bi',
        y[edge_index[0]],
        source_coefs
    )

    w2 = torch.einsum(
        'bz,zi->bi',
        y[edge_index[1]],
        target_coefs
    )

    w = w1 + w2

if device == 'cuda':
    torch.cuda.synchronize()

einsum_time = time.time() - start


start = time.time()

for _ in range(n_iter):

    w_fast = (
        source_coefs[atom_type[edge_index[0]]]
        + target_coefs[atom_type[edge_index[1]]]
    )

if device == 'cuda':
    torch.cuda.synchronize()

gather_time = time.time() - start


max_diff = (w - w_fast).abs().max().item()

print(f'Device: {device}')
print(f'Max diff: {max_diff:e}')
print()

print(f'Einsum time : {einsum_time:.4f} s')
print(f'Gather time : {gather_time:.4f} s')
print()

print(f'Speedup: {einsum_time / gather_time:.2f}x')