Interaction
===========

We currently recommend using ``CgtpInteraction``. It supports operator fusion
via OpenEquivariance or CuEquivariance, which can significantly reduce memory
usage and improve computational efficiency.

Although ``SO2Interaction`` is theoretically more advantageous at large angular
momentum, it currently lacks support for operator fusion libraries, and is
therefore not recommended.

.. autoclass:: tace.models._e3nn.inter.CgtpInteraction
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.inter.SO2Interaction
   :no-members:
   :show-inheritance: