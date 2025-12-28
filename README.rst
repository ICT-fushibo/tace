Tensor Atomic Cluster Expansion
===============================
.. = - ~ ^ "

This is the official repository for TACE atomistic machine learning, 
and it also temporarily hosts the ``cartnn`` package. 
In the future, if ICTP and ICTC are fully developed, ``cartnn`` will be moved to a repository forked from ```e3nn``.
You can use Cartesian-3j and ICTD like ``from cartnn.o3 import cartesian-3j, ICTD``.

.. image:: arch.png
   :width: 100%
   :align: center

Docs
----

https://tace.readthedocs.io/en/latest/index.html

Overview
--------

TACE is a Cartesian-based machine learning model designed to predict both scalar and tensorial properties.

In principle, the framework supports any tensorial properties (either direct or conservative) determined by the underlying atomic structure. 
Currently, the officially supported properties include:

- Energy
- Forces (conservative | direct)
- Hessians (conservative, predict only)
- Stress (conservative | direct)
- Virials (conservative | direct)
- Charges (Qeq or uniform)
- Dipole moment (conservative | direct)
- Polarization (conservative, multi-value for PBC systems)
- Polarizability (conservative | direct)
- Born effective charges (conservative, under electric field or LES)  *(LES not tested by us)*
- Atomic stresses (conservative, predict only)
- Atomic virials (conservative, predict only)
- Noncollinear magnetic forces (conservative)
- Collinear magnetic forces (conservative) *(not tested by us)*
- Magnetization (conservative) *(not tested by us)*
- Magnetic susceptibility (conservative) *(not tested by us)*

For embedding property, we support:

- level (different computational levels)
- charges
- total charge
- electric field
- noncollinear magmoms (not time reversal)
- collinear magmoms *(not tested by us)*
- total collinear magmoms *(not tested by us)*
- spin multiplicity *(not tested by us)*
- electron_temperature *(not tested by us)*
- magnetic field *(not tested by us)*
- total noncollinear magmoms *(not tested by us)*


Plugins
-------

TACE currently supports the following plugin:

- **LES** (Latent Ewald Summation)


Interfaces
----------

- ✅ Supports integration with **TorchSim**.

- ✅ Supports integration with **LAMMPS-ML-IAP**.

- ✅ Supports integration with **ASE Calculator**.



Citing
------

If you use TACE, please cite our papers:

.. code-block:: bibtex

   @misc{TACE,
         title={TACE: A unified Irreducible Cartesian Tensor Framework for Atomistic Machine Learning}, 
         author={Zemin Xu and Wenbo Xie and Daiqian Xie and P. Hu},
         year={2025},
         eprint={2509.14961},
         archivePrefix={arXiv},
         primaryClass={stat.ML},
         url={https://arxiv.org/abs/2509.14961}, 
   }

If you use Cartesian-3j or Cartesian-nj, please cite our papers:

.. code-block:: bibtex

   @misc{Cartesian-nj,
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

If you have any problems, suggestions or cooperations, please contact us through xvzemin@smail.nju.edu.cn

For bugs or feature requests, please use https://github.com/xvzemin/tace/issues.

License
-------

The TACE code is published and distributed under the MIT License.

