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
   qa/qa
   
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
- Hessians (conservative, predict only)
- Stress (conservative | direct)
- Virials (conservative | direct)
- Charges (lagrangian or uniform_distribution)
- Dipole moment (conservative | direct)
- Polarization (conservative, multi-value for PBC systems)
- Polarizability (conservative | direct)
- Born effective charges (conservative, under electric field or LES )  (LES predict only)
- Atomic stresses (conservative, predict only)
- Atomic virials (conservative, predict only)
- Noncollinear magnetic forces (conservative)
- Magnetization (conservative) *(not tested by us)*
- Magnetic susceptibility (conservative) *(not tested by us)*


For embedding property, we support:

- level (different computational levels)
- charges
- total charge
- electric field
- initial noncollinear magmoms
- spin multiplicity *(not tested by us)*
- electron_temperature *(not tested by us)*
- magnetic field *(not tested by us)*

Methodology
-----------

TACE leverages **Irreducible Cartesian Tensor Decomposition (ICTD)** and the **Atomic Cluster Expansion (ACE)** 
to efficiently and accurately learn semi-local chemical environments.  

Plugins
-------

TACE currently supports the following plugin:

- **LES** (Latent Ewald Summation)


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

If you have any problems, suggestions or cooperations, please contact us through xv_chana@163.com

For bugs or feature requests, please use https://github.com/xvzemin/tace/issues.

License
-------

The TACE code is published and distributed under the MIT License.

