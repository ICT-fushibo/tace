Note
====

The OMat24 series of RRA and ECE models will be released soon, and smaller models will also be added.
TECE-OAM-RRA-1.0 is not the final version of OAM. As the author will be attending ICML 2026 in the coming days, 
TACE v0.2.0 will not be officially updated for the next few days. 
However, the current models are already available for use.
Please do not use the fine-tuning feature until the official release of v0.2.0.

.. image:: fig/logo.svg
   :width: 100%
   :align: center

Tensor Atomic Cluster Expansion (TACE)
======================================
.. = - ~ ^ "
TACE is designed with physical priors and strong inductive biases to enhance extrapolation capability. 
It performs Atomic Cluster Expansion and Edge Cluster Expansion based on spherical tensors 
or irreducible Cartesian tensors, with an optional attention architecture.

Docs
----

https://tace.readthedocs.io/en/latest/index.html


Foundation Model and Fine-tuning
--------------------------------

https://github.com/xvzemin/tace-foundations

- ✅ Full-parameter.

- ✅ Freeze-parameter.

- ✅ LoRA.


Tutorial and Train from scratch
-------------------------------
The docs contain a complete tutorial. 

We also provide complete input files and a series of example scripts, including ASE, TorchSim ..., at 

https://github.com/xvzemin/tace/tree/main/example


.. code-block:: bash

   # Minimal training example
   git clone https://github.com/xvzemin/tace.git
   cd tace
   pip install .
   cd example/train
   tace-train -cn tace.yaml

Overview
--------

Currently, the officially supported properties include:

- Energy
- Forces (conservative | direct)
- Hessian (conservative, predict only)
- Stress (conservative | direct)
- Virials (conservative | direct)
- Charges (lagrangian or uniform_distribution)
- Dipole moment (conservative | direct)
- Polarization (conservative, multi-value for PBC systems)
- Polarizability (conservative | direct)
- Born effective charges (conservative, under electric field or LES)  (LES predict only)
- Atomic stresses (conservative, predict only)
- Atomic virials (conservative, predict only)
- absolute final collinear magmoms
- Noncollinear magnetic forces (O(3))

For embedding property, we support:

- fidelity_idx (different computational levels)
- charges
- total charge
- electric field
- initial (non)collinear magmoms
- magnetic field (O(3))


Plugins
-------

TACE currently supports the following plugin:

- **LES** (Latent Ewald Summation)


Interfaces
----------

- ✅ Supports integration with **ASE Calculator**.

- ✅ Supports integration with **LAMMPS-ML-IAP**.

- ✅ Supports integration with **TorchSim**.

- ✅ Supports integration with **OpenMM-ML (OpenMM-ML -> ASE -> TACE)**.

- ✅ Supports integration with **USPEX (USPEX -> LAMMPS-ML-IAP -> TACE)** (Python=3.9).


Citing
------

If you use TACE, please cite our papers:

.. code-block:: bibtex

   # Cartesian TACE
   @misc{xu2026spectralspatialtensoratomiccluster,
         title={Spectral/Spatial Tensor Atomic Cluster Expansion with Universal Embeddings in Cartesian Space}, 
         author={Zemin Xu and Wenbo Xie and P. Hu},
         year={2026},
         eprint={2509.14961},
         archivePrefix={arXiv},
         primaryClass={stat.ML},
         url={https://arxiv.org/abs/2509.14961}, 
   }

   # Spherical/SO(2) TACE paper, will be updated soon

If you use cartnn, Cartesian-3j, cMACE, cNequIP, cAllegro, please cite our papers:

.. code-block:: bibtex

   @misc{xu2026cartesian3jframeworkmachinelearning,
         title={A Cartesian-3j Framework for Machine Learning Interatomic Potentials}, 
         author={Zemin Xu and Chenyu Wu and Wenbo Xie and P. Hu},
         year={2026},
         eprint={2512.16882},
         archivePrefix={arXiv},
         primaryClass={physics.chem-ph},
         url={https://arxiv.org/abs/2512.16882}, 
   }

Contact
-------

For bugs or feature requests, please use https://github.com/xvzemin/tace/issues.

License
-------

The TACE code is published and distributed under the MIT License.
