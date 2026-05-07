Tensor Atomic Cluster Expansion
===============================
.. = - ~ ^ "

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   install/install
   guide/guide
   model/model
   
.. toctree::
   :maxdepth: 1
   :caption: Changelog

   changelog/changelog

Overview
--------

TACE is a Cartesian-based machine learning model designed to predict both scalar and tensorial properties.

In principle, the framework supports any tensorial properties (either direct or conservative) determined by the underlying atomic structure. 
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
- Noncollinear magnetic forces (conservative, SO(3) now, O(3) will be improved later)
.. - Magnetization (conservative) *(not tested by us)*
.. - Magnetic susceptibility (conservative) *(not tested by us)*
.. - final (non)collinear magmoms (not time reversal)
.. - total (non)collinear magmoms *(not tested by us)*

For embedding property, we support:

- fidelity_idx (different computational levels)
- charges
- total charge
- electric field
- initial (non)collinear magmoms
- spin multiplicity *(not tested by us)*
- electron_temperature *(not tested by us)*
- magnetic field *(not tested by us, SO(3) now, O(3) will be improved later)*


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

- ✅ Supports integration with **USPEX (USPEX -> LAMMPS-ML-IAP -> TACE)**.


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

If you use Cartesian-3j or Cartesian-nj, please cite our papers:

.. code-block:: bibtex

   @misc{xu2025cartesiannjextendinge3nnirreducible,
         title={Cartesian-nj: Extending e3nn to Irreducible Cartesian Tensor Product and Contracion}, 
         author={Zemin Xu and Chenyu Wu and Wenbo Xie and Daiqian Xie and P. Hu},
         year={2025},
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