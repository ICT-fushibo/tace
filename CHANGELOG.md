# Changelog

## v0.0.9 (Coming soon)

* Add fine-tuning functionality, introduce LoRA support, and add fine-tuning scripts and tutorials.
* Consider refactor model, since there was previously no library for Cartesian models similar to e3nn, the code for all model components is relatively messy and not conducive to extension.


## v0.0.8

Substantially refactor the code by moving Cartesian tensor operations from tace to cartnn, removing unnecessary tace components

### 🎉 New Features
* Add interface to torchSim 
* Add optional dispersion corrections in Ase Calculator

## v0.0.7

Fixed several bugs and added content related to the cartnn package. However, tace currently does not use cartnn.  
From v0.0.1 to v0.0.7, we supported multiple paths for a single combination in the case of reducible Cartesian tensors, as well as several features related to reducible Cartesian tensors.  
Starting from v0.0.8, we will remove support for reducible Cartesian tensors and will only support irreducible Cartesian tensor products and contractions.

## v0.0.6

Modify the multi-GPU parallel framework to support two modes for graph data — memory and LMDB (stored on disk) — to facilitate training on large-scale datasets. 

### 🎉 New Features
* Add experimental matrice tensor product, analytical tensor product, SO(2) tensor product.

## v0.0.5

### 💥 Breaking Changes
* no longer compatible with v0.0.1-v0.0.4

### 🎉 New Features
* Add support for multihead architecture for any rank.

## v0.0.4

### 🎉 New Features
* Add LAMMPS-ML-IAP interface; included edge_forces, atomic_stresses, and atomic_virials.

## v0.0.3

### 🎉 New Features
* Add long-range plugin LES
* Supported noncolinear magnetic moments embedding while predicting magnetic forces, merged various equivariants into universal equivariant embedding

## v0.0.2

### 🎉 New Features
* Add support for embedding invariants (e.g., charges, fidelities, spin)
* Add external field and charge equilibration (QEq and uniform)


## v0.0.1

The earliest version supports the prediction of energy, forces, stress, dipole moments, and polarizability, and provides an interface with ASE.


  <!-- - 
  -  -->

<!-- ## New Contributors
* @xxx made their first contribution in [#1](https://github.com/xvzemin/tace/pull/1) -->



