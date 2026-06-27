.. note::

`TACE-OAM-RRA-Preview` is an intermediate model intended for preview
purposes only. Backward compatibility with future releases is not
guaranteed.

`TACE-OAM-RRA` and `TECE-OAM-RRA` will
be released alongside the forthcoming TACE paper, which has not yet been
made public.

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

   @misc{xu2026spectralspatialtensoratomiccluster,
         title={Spectral/Spatial Tensor Atomic Cluster Expansion with Universal Embeddings in Cartesian Space}, 
         author={Zemin Xu and Wenbo Xie and P. Hu},
         year={2026},
         eprint={2509.14961},
         archivePrefix={arXiv},
         primaryClass={stat.ML},
         url={https://arxiv.org/abs/2509.14961}, 
   }

If you use Cartesian-3j, please cite our papers:

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
