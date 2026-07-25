# TACE with LAMMPS ML-IAP

The complete build, export, and distributed launch tutorial is available in
the [TACE LAMMPS documentation](https://tace.readthedocs.io/en/latest/guide/lammps.html).

Export an eager model:

```bash
tace-export-lammps \
  -m ~/.cache/tace/TACE-OAM-7M.pt \
  --backend mliap \
  --device cuda
```

Use the exported loader in the LAMMPS input:

```text
pair_style mliap unified TACE-OAM-7M.pt-lammps_mliap.pt 0
pair_coeff * * H C N
```

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
lmp -k on g 1 -sf kk \
  -pk kokkos newton on neigh half \
  -in in.lmp
```

One node with two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
mpirun --bind-to none -np 2 \
  lmp -k on g 2 -sf kk \
    -pk kokkos newton on neigh half \
    -in in.lmp
```

Multi-rank GPU runs require CUDA-aware MPI. TACE ML-IAP currently requires the
CUDA Kokkos backend; use the ASE or TorchSim interface for CPU inference.
