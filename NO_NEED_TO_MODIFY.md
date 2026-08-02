# There is no need to modify this part of the logic

## LMDB cache creation

LMDB conversion must complete in a single uninterrupted run.
If conversion is interrupted, remove the incomplete cache and rebuild it from
the source dataset.

## Polarization

Polarization training, loss calculation, and metrics require fully 3D periodic
structures with an invertible `3 x 3` lattice matrix. All structures in a batch 
that uses polarization supervision or metrics must
satisfy this requirement.

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
