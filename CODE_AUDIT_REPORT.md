# TACE Code Audit Report

## 1. Audit scope

- Repository: `/home/xuzemin/tace_opt/tace`
- Audited commit: `3b476d6`
- Audit date: 2026-08-02
- Source changes made during this audit: none
- Scope: dataset loading and caching, model construction, derivative outputs,
  acceleration backends, AOTI export/loading, ASE and TorchSim interfaces,
  command-line scripts, and metrics

This report separates confirmed defects from runtime risks that still require a
CUDA-capable integration test. Line numbers refer to commit `3b476d6` and may
move after subsequent edits.

## 2. Executive summary

The audit found no remaining high-priority correctness or data-integrity issues.

The acceleration architecture is generally coherent, but AOTI has graph-size
boundary conditions that are not fully handled, the CUE+AOTI metadata can
describe a different backend from the one actually used, and TorchSim's optional
dependency handling currently prevents importing its adapter when `torch_sim`
is absent.

## 3. Validation performed

The following checks were performed without editing source code:

- `python -m compileall -q tace`: passed.
- Installed dependency consistency check with `pip check`: passed.
- Import scan over 164 TACE modules: 163 passed; one failed because two scripts
  register the same Hydra resolver in one Python process.
- Isolated reproductions were run for missing derivative-output keys,
  environment-variable type handling, and optional TorchSim import.

Limitations:

- A complete CUDA/Triton/AOTI compilation and execution matrix was not run in
  this audit because GPU access was unavailable inside the sandbox.
- Multi-node and LAMMPS runtime behavior was not retested in this audit.

## 4. Severity convention

- **P1 / High**: can produce incorrect physical behavior, silently lose data,
  or break an advertised training/evaluation workflow.
- **P2 / Medium**: affects boundary cases, optional interfaces, acceleration
  selection, reproducibility, or developer validation.
- **P3 / Low**: localized CLI, state-mutation, or maintenance issue with limited
  immediate impact.

## 5. Findings

### TACE-001: AOTI dynamic-shape constraints exclude valid small graphs

**Severity:** P2 / Medium  
**Status:** Confirmed by export-contract inspection; full CUDA reproduction pending

Relevant code:

- `tace/models/compile/aot.py:484-509`
- `tace/models/compile/aot.py:609-625`
- `tace/models/compile/aot.py:145-153`
- `tace/lightning/lit_model.py:684`

The exported dynamic dimensions require at least two nodes and two edges. The
sample-free export path pads graph-related entries but does not guarantee at
least two actual nodes and edges at inference.

Valid inputs such as a one-atom structure or a graph with zero edges can
violate the generated PT2 guards. This is the same class of failure as symbolic
shape assertions that pass during export but fail for a later structure.

A second boundary issue occurs when `load_tace()` passes its default device
through to AOTI. The device lookup can leave it as `None`, after which the AOTI
device checker calls `torch.device(None)` and raises `TypeError`.

Recommended correction:

- Either export dimensions with minima matching the actual graph contract or
  pad nodes and edges as well as graph indices, then remove dummy contributions
  from outputs.
- Resolve `device=None` to the package/device default before entering AOTI
  validation.
- Add CPU and CUDA tests for one atom, two atoms outside the cutoff, one normal
  molecule, and differently sized structures loaded from the same sample-free
  PT2 package.

### TACE-002: TorchSim adapter is not safely optional and can retain tensors on the wrong device

**Severity:** P2 / Medium  
**Status:** Optional-import failure confirmed; device issue confirmed by inspection

Relevant code:

- `tace/interface/torchsim.py:16-31`
- `tace/interface/torchsim.py:61`
- `tace/interface/torchsim.py:158-164`
- `tace/interface/torchsim.py:187-196`
- TorchSim optional dependency declaration in `pyproject.toml`

The module catches `ImportError` when `torch_sim` is absent, but class creation
immediately references undefined TorchSim symbols in its base class, defaults,
and annotations. Importing the adapter in a base TACE installation therefore
warns and then fails with `NameError`.

When TorchSim is installed, constructor-provided `atomic_numbers` and
`system_idx` are retained on their original device. A CUDA model can therefore
combine CPU node metadata with CUDA positions and model parameters.

Recommended correction:

- Define the adapter only when TorchSim imports successfully, or expose a small
  placeholder that raises a clear installation error on construction.
- Move all retained tensors to `self.device` and normalize dtypes during setup.
- Test import without the optional extra and inference with CPU-created metadata
  on a CUDA model.

### TACE-003: CUE+AOTI export metadata can disagree with the backend actually used

**Severity:** P2 / Medium  
**Status:** Confirmed code inconsistency; package-loading impact needs CUDA test

Relevant code:

- `tace/models/_e3nn/fused.py:151-173`
- `tace/models/compile/aot.py:645-651`

When CUE and AOTI are requested together, fused-layer setup can internally fall
back to OpenEquivariance. AOTI packaging later determines required custom
imports from environment variables, which can still indicate CUE and not OEQ.

The PT2 package can therefore record/import a different custom-op provider from
the one present in the traced graph.

Recommended correction:

- Track the backend selected by each constructed operation and build package
  metadata from the actual graph dependencies, not only from requested
  environment variables.
- Add an export/load test for every supported backend combination in a fresh
  Python process.

### TACE-004: Acceleration environment setup does not normalize values to strings

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

### TACE-005: `allow_unused=True` does not protect disconnected derivatives

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

### TACE-006: Acceleration options are silently ineffective for fully serialized modules

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

### TACE-007: Duplicate Hydra resolver registration prevents importing scripts together

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

### TACE-008: Dataset split index arguments are declared but unused

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

### TACE-009: Optimizer configuration mutates the stored configuration

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

1. Fix AOTI small-graph constraints and default-device handling.
2. Make TorchSim optional import and device movement robust.
3. Record actual custom-op dependencies during AOTI packaging.
4. Define behavior for acceleration requests on full-model serialization.

### Phase 2: Restore validation confidence

1. Add real PT2 export/reload tests across structure sizes.
2. Add force-gradient and memory-regression tests for every fused backend.
3. Add multigraph tests for polarization and partially labeled
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
| AOTI | 1 atom; 0 edges; different structures from one PT2; CPU; CUDA |
| Acceleration | eager; OEQ; CUE; EQT; Triton; compile; supported combinations |
| Interfaces | native model; ASE; TorchSim; state dict; full module; PT2 |

Numerical tests should compare energy, forces, stress/virials, and available
field derivatives against the eager e3nn reference using dtype-appropriate
tolerances. Fused training tests should include backward gradients and peak CUDA
memory, not only forward outputs.

## 8. Conclusion

The remaining findings primarily concern deployment boundaries, acceleration
selection, and maintenance behavior rather than high-priority correctness or
data-integrity failures.

No source code, configuration, or test file was modified as part of producing
this report.
