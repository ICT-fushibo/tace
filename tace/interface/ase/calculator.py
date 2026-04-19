################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import warnings


import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from torch_geometric.loader import DataLoader


from tace.models.adapter import TensorModel
from tace.lightning import load_tace
from tace.dataset.quantity import PROPERTY
from tace.dataset.graph import from_atoms
from tace.dataset.quantity import (
    PROPERTY,
    KEYS,
    KeySpecification,
    update_keyspec_from_kwargs,
)
from tace.utils._global import DTYPE, DEVICE


class TACEAseCalc(Calculator):
    """
    Initialize a TACEAseCalc. We support the most fundamental potential energy surface property and multi-fidelity, 
    multi-head, etc. For some advanced features, you need to store the attributes that need to be embedded in atoms.info, 
    atoms.arrays or add a funciton by yourself. If you only need to predict, you can directly use the `tace-eval` 
    command. It will output the predicted files, and if you add the `--test` option, it will also output the errors.

    Parameters
    ----------
    model_path : str
        Path to the trained model, file ends with pt, .pth or .ckpt.
    dtype : str, optional
        Model dtype for computations, e.g., float32 or float64.
    device : str, optional
        The device to run computations on, e.g., cpu or cuda.
        If None, the device is automatically inferred.
    fidelity_idx : int, optional
        Specify which fidelity fidelity_idx to use. 
    target_property: list(str), optional
        Extra caculate hessian, atomic_virials, Conservative polarizability, etc,
        If you want to use this parameter, you must provide all the required physical quantities.
    neighborlist_backend: str
        Support backend in one of [ase, matscipy, vesin], recommend matscipy
    **kwargs
        Additional keyword arguments passed to the ASE Calculator base class.
    """

    def __init__(
        self,
        model: str,
        *,
        dtype: str | None = None,
        device: str | None = None,
        fidelity_idx: int | None = None,
        target_property: list[str] | None = None,
        neighborlist_backend: str = "matscipy",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # === init ===
        model: TensorModel = load_tace(
            model, 
            device, 
            strict=True, 
            use_ema=True, 
            target_property=target_property
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        model_dtype = model.get_model_dtype()
        self.dtype = DTYPE[dtype or model_dtype]
        self.device = DEVICE[device or torch.device("cuda" if torch.cuda.is_available() else "cpu")]
        torch.set_default_dtype(self.dtype)
        if DTYPE[dtype] != DTYPE[model_dtype]:
            print(f"[Warning] Model dtype {(model_dtype)} does not match args.dtype {(dtype)}. Forcing dtype to {self.dtype}")
        model = model.to(dtype=self.dtype)
        self.target_property = model.get_target_property() 
        self.embedding_property = model.get_embedding_property()
        self.max_neighbors = model.get_max_neighbors()
        self.cutoff = model.get_cutoff()
        self.element = model.get_torch_element()
        self.neighborlist_backend = neighborlist_backend
        self.implemented_properties = []
        for p in self.target_property:
            ase_name = PROPERTY[p]['ase_name']
            save_name = ase_name if ase_name else p
            if save_name == 'energy':
                self.implemented_properties.extend(["energy" ,"free_energy"])
            else:
                self.implemented_properties.append(save_name)
        if fidelity_idx is not None:
            self.fidelity_idx = fidelity_idx
            model.reset_fidelity_idx(fidelity_idx) 
        else:
            self.fidelity_idx = model.get_fidelity_idx()
        self.keySpecification = KeySpecification()
        update_keyspec_from_kwargs(self.keySpecification, KEYS)
        self.model = model.to(self.device)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        atoms.info["fidelity_idx"] = self.fidelity_idx # fidelity fidelity_idx
        # === dataloader ===
        data = [
            from_atoms(
                self.element,
                atoms,
                self.cutoff,
                max_neighbors=self.max_neighbors,
                target_property=self.target_property,
                embedding_property=self.embedding_property,
                keyspec=self.keySpecification,
                training=False,
                neighborlist_backend=self.neighborlist_backend,
            ) 
        ]
        dataloader = DataLoader(
            dataset=data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batch = next(iter(dataloader)).to(self.device)
   
        # === forward ===
        outs = self.model(batch)
        # === update ===
        self.results = {}
        for p in self.target_property:
            p_rank = PROPERTY[p]['rank']
            p_scope = PROPERTY[p]['scope']
            ase_name = PROPERTY[p]['ase_name']
            save_name = ase_name if ase_name else p
            if p_scope == 'per-system':
                if p_rank == 0:
                    if p == 'energy':
                        energy = outs[p].detach().cpu().item()
                        self.results['energy'] = energy
                        self.results["free_energy"] = self.results['energy']
                    else:
                        self.results[save_name] = outs[p].detach().cpu().item()
                else:
                    self.results[save_name] = outs[p].detach().cpu().numpy().squeeze(0)
            elif p_scope == 'per-atom':
                self.results[save_name] = outs[p].detach().cpu().numpy()
            elif p_scope == 'per-edge':
                self.results[save_name] = outs[p].detach().cpu().numpy()
            else:
                self.results[save_name] = outs[p].detach().cpu().numpy()

def add_dispersion(
    base_calc: Calculator,
    damping: str = "bj",  # choices: ["zero", "bj", "zerom", "bjm"]
    dispersion_xc: str = "pbe",
    dispersion_cutoff: float = 40.0 * units.Bohr,
    **kwargs,
) -> SumCalculator:
    try:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError as e:
        raise RuntimeError(
            "Please install torch-dftd to use dispersion corrections (see https://github.com/pfnet-research/torch-dftd)"
        ) from e
    
    d3_calc = TorchDFTD3Calculator(
        dtype=base_calc.dtype,
        device=base_calc.device,
        damping=damping,
        xc=dispersion_xc,
        cutoff=dispersion_cutoff,
        **kwargs,
    )
    return SumCalculator([base_calc, d3_calc])