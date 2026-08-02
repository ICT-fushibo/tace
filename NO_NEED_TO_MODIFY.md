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
