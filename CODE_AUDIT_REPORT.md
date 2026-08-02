# TACE Code Audit Report

## 1. Audit scope

- Repository: `/home/xuzemin/tace_opt/tace`
- Audited commit: `3b476d6`
- Audit date: 2026-08-02
- AOTI remediation and integration tests completed after the initial audit
- Scope: dataset loading and caching, model construction, derivative outputs,
  acceleration backends, AOTI export/loading, ASE and TorchSim interfaces,
  command-line scripts, and metrics

This report separates remaining confirmed defects from issues resolved during
the AOTI remediation. Line numbers refer to commit `3b476d6` and may move after
subsequent edits.

## 2. Executive summary

The audit found no remaining high-priority correctness or data-integrity issues.

The AOTI graph, ASE, TorchSim, and LAMMPS deployment paths were validated on
CPU and CUDA. Dynamic single-system execution, actual custom-op dependency
metadata, PT2 archive compatibility, and TorchSim's dependency and device
handling were corrected.

## 3. Validation performed

The initial audit included:

- `python -m compileall -q tace`: passed.
- Installed dependency consistency check with `pip check`: passed.
- Import scan over 164 TACE modules: 163 passed; one failed because two scripts
  register the same Hydra resolver in one Python process.
- Isolated reproductions were run for missing derivative-output keys,
  environment-variable type handling, and optional TorchSim import.

The AOTI remediation was validated with PyTorch 2.13 on the second of two
NVIDIA RTX 4090 GPUs and with GPU visibility disabled for CPU deployment:

- Sample-free CUDA graph packages were exported with OEQ and with CUE+AOTI.
- Native `load_tace`, ASE, and TorchSim were compared against eager inference
  for H2, H2O, a mixed two-system batch, and a periodic 192-atom water system.
- The minimum supported graph of two nodes, one edge, and one system executed
  successfully with finite energy, forces, and stress.
- A CPU graph package passed ASE, TorchSim, and one-step TorchSim MD tests.
- A 288-atom LAMMPS ML-IAP system passed `run 0` and three NVE steps with both
  eager and AOTI loaders. Initial potential energies agreed within 1e-4 eV.
- Repeated PT2 loads completed without compilation. New packages loaded through
  the current PT2 loader without the legacy-package fallback warning.
- CUE+AOTI correctly recorded OEQ as the actual custom-op dependency after the
  scatter tensor products fell back to OEQ. Loading succeeded in a fresh
  process with all acceleration environment variables disabled.
- Importing the TorchSim adapter without `torch-sim-atomistic` raised a clear
  error with the installation command. CPU-created atomic numbers and system
  indices were moved to CUDA before a real TACE-OAM-7M H2O forward pass and
  one-step TorchSim MD run on the second RTX 4090.

Limitations:

- Triton combined with AOTI was not part of this remediation.
- Multi-node and multi-GPU LAMMPS runtime behavior was not retested.
- PyTorch 2.13 reports an `AOTI_CPU_ISA` metadata warning when the CUDA package
  is loaded from LAMMPS's embedded Python process. Repeated CUDA runs completed
  correctly; this is distinct from the resolved legacy PT2 loader fallback.

## 4. Severity convention

- **P1 / High**: can produce incorrect physical behavior, silently lose data,
  or break an advertised training/evaluation workflow.
- **P2 / Medium**: affects boundary cases, optional interfaces, acceleration
  selection, reproducibility, or developer validation.
- **P3 / Low**: localized CLI, state-mutation, or maintenance issue with limited
  immediate impact.

## 5. Findings

### TACE-001: Acceleration environment setup does not normalize values to strings

**Severity:** P2 / Medium  
**Status:** Confirmed by reproduction

Relevant code:

- `tace/utils/env.py:19-22`

`os.environ` accepts only strings, but configuration values are assigned
directly. A YAML integer or boolean raises `TypeError: str expected, not int`
instead of enabling/disabling the requested feature.

Recommended correction:

- Normalize accepted values to an explicit string representation, preferably
  `"0"` and `"1"` for booleans.
- Reject ambiguous values with a configuration-path-aware message.
- Test boolean, integer, string, missing, and `force=False` behavior.

### TACE-002: `allow_unused=True` does not protect disconnected derivatives

**Severity:** P2 / Medium  
**Status:** Confirmed with a disconnected-output reproduction

Relevant code:

- `tace/models/derivative/adapter.py:94-131`
- `tace/models/compile/wrapper.py:247-267`

Autograd calls allow unused inputs, but the result is negated or otherwise
operated on before checking whether it is `None`. If an energy/readout is
independent of positions, displacement, or a field, the wrapper raises a type
error rather than returning the mathematically correct zero derivative.

Ordinary OAM inference remained connected in the isolated test, so this is most
likely to appear with specialized outputs, frozen/constant heads, isolated
graphs, or future architectures.

Recommended correction:

- Replace a `None` gradient with a correctly shaped zero tensor before applying
  signs or reshaping.
- Keep eager and compile wrappers behaviorally identical.
- Test a constant-energy dummy model in training and evaluation modes.

### TACE-003: Acceleration options are silently ineffective for fully serialized modules

**Severity:** P2 / Medium  
**Status:** Confirmed behavior/design gap

Relevant code:

- acceleration selection in `tace/lightning/torch_model.py:63-70`
- full-module and already-loaded-model paths in the loading code

Backend selection is applied while constructing a model from configuration. A
fully serialized `torch.nn.Module`, or a module object already instantiated by
the caller, bypasses that construction path. Requesting OEQ, CUE, EQT, or other
construction-time acceleration can therefore have no effect without a clear
warning.

Recommended correction:

- Document which export formats permit backend reconstruction.
- Detect incompatible acceleration requests and fail or warn explicitly.
- Prefer state-dict/config artifacts when backend substitution is expected.

### TACE-004: Duplicate Hydra resolver registration prevents importing scripts together

**Severity:** P3 / Low  
**Status:** Confirmed by module import scan

Relevant code:

- resolver registration in `tace/scripts/train.py:40`
- resolver registration in `tace/scripts/graph.py:39`
- helper implementation in `tace/utils/hydra_resolver.py:30-41`

Both scripts register the same resolver at import time. Importing them in one
Python process raises a duplicate-registration `ValueError`. This affects module
discovery, documentation tooling, and applications that embed more than one
TACE command.

Recommended correction:

- Make resolver registration idempotent by checking whether it already exists,
  or centralize one registration call.
- Add an import-order test for all script modules.

### TACE-005: Dataset split index arguments are declared but unused

**Severity:** P3 / Low  
**Status:** Confirmed by inspection

Relevant code:

- argument declarations in `tace/scripts/split.py:35-37`
- command implementation after `tace/scripts/split.py:41`

The CLI declares three index-related options, but the implementation does not
consume them. Users can provide apparently valid arguments that do not affect
the generated split.

Recommended correction:

- Implement the documented behavior or remove the options until supported.
- Add an assertion that custom indices change the output split.

### TACE-006: Optimizer configuration mutates the stored configuration

**Severity:** P3 / Low  
**Status:** Confirmed by inspection

Relevant code:

- `tace/lightning/lit_model.py:482-490`
- `tace/lightning/lit_model.py:567-572`

Optimizer and scheduler setup removes fields such as `_target_` with `pop()`
from the model's stored configuration. A repeated call to
`configure_optimizers()` can therefore behave differently or fail, and the
configuration retained for logging/checkpoint metadata is no longer the
configuration originally supplied.

Recommended correction:

- Copy the relevant configuration nodes before destructive extraction.
- Add a test that calls optimizer configuration twice and verifies that the
  stored config remains unchanged.

## 6. Suggested remediation order

### Phase 1: Stabilize deployment interfaces

1. Define behavior for acceleration requests on full-model serialization.

### Phase 2: Restore validation confidence

1. Add force-gradient and memory-regression tests for every fused backend.
2. Add multigraph tests for polarization and partially labeled
   multi-fidelity data.

### Phase 3: CLI and maintenance fixes

1. Make Hydra resolver registration idempotent.
2. Resolve unused split arguments.
3. Stop optimizer setup from mutating stored configuration.

## 7. Minimum regression matrix

The following matrix would cover the highest-risk behavior without requiring a
full scientific benchmark for every commit:

| Area | Minimum cases |
| --- | --- |
| Dataset readers | one valid file; one corrupt file; scalar metadata; per-atom metadata |
| Derivatives | energy/forces/stress/virials; dipole; polarizability; BEC; disconnected output |
| Multi-fidelity | one head missing labels; finite default scale/shift/atomic energies |
| Parity | SO(3) rotations; inversion; polar vectors; axial vectors/magnetic forces |
| AOTI | 2 nodes; 1 edge; 1/multiple graphs; varying structures; CPU; CUDA |
| Acceleration | eager; OEQ; CUE; EQT; Triton; compile; supported combinations |
| Interfaces | native model; ASE; TorchSim; state dict; full module; PT2 |

Numerical tests should compare energy, forces, stress/virials, and available
field derivatives against the eager e3nn reference using dtype-appropriate
tolerances. Fused training tests should include backward gradients and peak CUDA
memory, not only forward outputs.

## 8. Conclusion

The resolved AOTI issues covered dynamic single-system constraints, CLI
activation, actual custom-op dependency recording, PT2 archive layout, and
default-device loading. TorchSim dependency errors and retained tensor devices
were also corrected. The remaining findings concern configuration handling,
derivative boundary cases, and maintenance behavior rather than the tested
deployment paths.
