TorchSim Calculator Tutorial
=======================

This tutorial demonstrates how to use a TACE model as a calculator within TorchSim.

TorchSim documentation: `torchsim <https://torchsim.github.io/torch-sim/>`_

For detailed usage and scripts (e.g., ``batchOPT``, ``batchMD``, and other scripts), see  
`https://github.com/xvzemin/tace/tree/main/example/torchSim <https://github.com/xvzemin/tace/tree/main/example/torchSim>`_


.. code-block:: python

    from tace.interface.torchsim import TACETorchSimCalc

    # === Input ===
    dtype = 'float32'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = "../TACE-v1-OMat24-L.pt" # Your Model
    level = 0  # first fidelity
    model = TACETorchSimCalc(
        model,
        level=level,
        device=device,
        dtype=dtype, 
        compute_forces=True,
        compute_stress=True,
    )

.. autoclass:: tace.interface.torchsim.torchsim.TACETorchSimCalc
   :no-members:
   :show-inheritance: