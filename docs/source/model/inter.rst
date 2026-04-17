Interaction
===========

We currently recommend using ``CGTP_Interaction``. It supports operator fusion
via OpenEquivariance or CuEquivariance, which can significantly reduce memory
usage and improve computational efficiency.

Although ``SO2_Interaction`` is theoretically more advantageous at large angular
momentum, it currently lacks support for operator fusion libraries, and is
therefore not recommended.

.. autoclass:: tace.models._e3nn.inte.CGTP_Interaction
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.inte.SO2_Interaction
   :no-members:
   :show-inheritance: