Target & Embedding Property Tutorial
====================================

.. note::

   Not all physical quantities are introduced in this tutorial.
   We only focus on several special properties that require particular attention in practice.

1. ``charges``
--------------

The charges property can be used either as a prediction target or as an embedding feature.

When charges is used as a prediction target, you must** choose one of the following options
to enforce charge conservation:

- lagrangian
- uniform_distribution

In general, uniform_distribution tends to yield slightly lower errors.
However, we recommend always using lagrangian, as it enforces stronger and more explicit
physical constraints.

2. ``initial_noncollinear_magmoms`` and ``noncollinear_magnetic_forces``
------------------------------------------------------------------------

These two options are generally expected to be used together.

Our current universal equivariant embedding does not guarantee time-reversal symmetry.
In other words, in the absence of an external magnetic field, flipping all magnetic moments
should leave the total energy invariant.

If you would like to enforce a certain degree of symmetry in practice, you may manually apply
the desired data augmentation strategy, such as flipping magnetic moments (and corresponding
properties) in the dataset.


3. ``Electric/Magnetic Field``
------------------------------

External electric field have been tested.
External magnetic fields are, in principle, also supported, but we have not yet tested them
due to the lack of suitable datasets. If you require magnetic field support, please contact us.

Note that polarization is a physical quantity defined specifically for periodic systems.
It is multi-valued; for details, please refer to the Allegro-pol paper.

By contrast:

- Dipole moments can be defined for both non-periodic and periodic systems.
- Training Born effective charges (BEC) under external fields is supported.
- BEC can also be predicted via LES, but training LES-BEC is not supported.


4. ``Atomic Virials/Stresses``
------------------------------

These two quantities are mainly used for local atomic stress analysis.

For periodic systems, as long as energy is included in training, atomic virials and atomic stresses
can, in principle, be derived. Only prediction is supported.


5. ``Hessians`` (Not Yet Publicly Available)
----------------------------------------

Hessian training is generally intended for non-periodic systems.
Although it is theoretically applicable to periodic systems, the computational cost is prohibitive.

Even for non-periodic systems, Hessian training is extremely expensive.
Therefore, we adopt the random row sampling strategy proposed in HORM:
during training, only a subset of Hessian rows is dynamically selected for each structure.


6. ``Level``
----------------------------------

The level attribute is treated as a dataset label used to distinguish
different levels of computational accuracy (e.g., different functionals).

It can be utilized in:

- multi-fidelity training
- multi-head training


7. ``spin_on``
--------------

The spin_on flag is used to distinguish magnetic and non-magnetic systems
when magnetic moments are neither embedded nor predicted. 
This helps the model better learn magnetic effects,


8. ``LES``
----------

For physical quantities derived from LES, we currently support:

- les_energy
- les_latent_charges
- les_born_effective_charges

Detailed tutorials on how to use and export LES-related quantities will be added gradually.

The computation of les_born_effective_charges can be enabled via::

    model.readout_fn.compute_bec = True

Note that this interface may be subject to change in future versions.


