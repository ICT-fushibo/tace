# There is no need to modify this part of the logic

## Dataset reader behavior

The `fair_aselmdb` reader intentionally copies unknown `row.data` entries to
both `Atoms.info` and `Atoms.arrays`. The input schema is not known in advance,
and TACE later selects the appropriate source according to the configured
property scope and key specification. This behavior should not be replaced.

Multi-file reading also intentionally allows valid files to be used when
another file cannot be read. A reader that returns an empty list is counted as
a failed file, including a genuinely empty file.

## LMDB cache creation

LMDB conversion must complete in a single uninterrupted run.
If conversion is interrupted, remove the incomplete cache and rebuild it from
the source dataset.

## Derived-property prerequisites

The `must_be_with` metadata is a partial data-loading aid rather than a complete
dependency graph, automatic input generator, or source of default values.
Relationships between physical inputs, model outputs, and derived quantities
can be configuration-dependent, so not every valid combination is encoded or
validated automatically.

Users are expected to provide the physical inputs and upstream outputs required
by the requested prediction. For example, charge-conserving charge prediction
requires `charges` as an output and `total_charge` as an input, while
`noncollinear_magnetic_forces` prediction requires `initial_noncollinear_magmoms`
as an input.

## Optional interface dependencies

Packages that support optional interfaces are not required for the core TACE
package, but they must be installed before importing their corresponding
interface. For example, `torch-sim-atomistic` must be installed before importing
`tace.interface.torchsim`.

An immediate `ImportError` with installation guidance is the intended behavior
when such a dependency is missing. This logic does not need to be replaced with
placeholder classes, delayed failures, or fallback implementations. Interfaces
whose own dependencies are installed, such as `tace.interface.ase`, remain
independently importable.

## Polarization

Polarization training, loss calculation, and metrics require fully 3D periodic
structures with an invertible `3 x 3` lattice matrix. All structures in a batch 
that uses polarization supervision or metrics must
satisfy this requirement.

