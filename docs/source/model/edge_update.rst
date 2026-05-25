Edge Update
===========

We recommend using ``IdentityEdgeUpdate``, ``Element2EdgeUpdate``, or ``TensorDotEdgeUpdate`` in most cases.

The most conservative choice is ``IdentityEdgeUpdate``, which does not introduce any additional information during edge updates. 
For datasets with relatively rich sampling in configuration space, we recommend using ``Element2EdgeUpdate`` to incorporate element-dependent information. 
If higher accuracy is desired and increased computational cost is acceptable, ``TensorDotEdgeUpdate`` is recommended.

.. autoclass:: tace.models._e3nn.edge.IdentityEdgeUpdate
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.edge.ElementEdgeUpdate
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.edge.Element2EdgeUpdate
   :no-members:
   :show-inheritance: