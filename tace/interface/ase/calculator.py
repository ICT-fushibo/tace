################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import warnings
from typing import Optional


import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from torch_geometric.loader import DataLoader


from ...lightning import load_tace
from ...dataset.quantity import PROPERTY
from ...dataset.element import TorchElement
from ...dataset.graph import from_atoms
from ...dataset.quantity import (
    PROPERTY,
    KEYS,
    KeySpecification,
    update_keyspec_from_kwargs,
)
from ...utils._global import DTYPE, DEVICE


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
    device : str | torch.device, optional
        The device to run computations on, e.g., cpu or cuda.
        If None, the device is automatically inferred.
    dtype : str, optional, default=None
        Model dtype for computations, e.g., float32 or float64.
    level : int
        Specify which fidelity level to use. 
    spin_on : bool
        If your model uses spin_on uie embedding, you can control whether 
        your calculation enables spin polarization.
    target_property: list(str)
        Extra caculate hessians, atomic_virials, Conservative polarizability, etc,
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
        dtype: Optional[str] = None,
        device: Optional[str] = None,
        level: Optional[int] = None,
        spin_on: Optional[bool] = None,
        target_property: Optional[list[str]] = None,
        neighborlist_backend: str = "matscipy",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # === init ===
        model = load_tace(
            model, 
            device, 
            strict=True, 
            use_ema=True, 
            target_property=target_property
        )
        model_dtype = model.readout_fn.cutoff.dtype
        dtype = dtype or model_dtype
        self.dtype = DTYPE[dtype]
        self.device = DEVICE[device or torch.device("cuda" if torch.cuda.is_available() else "cpu")]
        torch.set_default_dtype(self.dtype)
        if DTYPE[dtype] != DTYPE[model_dtype]:
            warnings.warn(
                f"Model dtype {model_dtype} != default dtype {dtype}. "
                f"This may cause silent type conversions."
            )
        model = model.to(dtype=self.dtype)
        self.target_property = model.get_target_property() 
        self.embedding_property = model.get_embedding_property()
        self.implemented_properties = []
        for p in self.target_property:
            ase_name = PROPERTY[p]['ase_name']
            save_name = ase_name if ase_name else p
            if save_name == 'energy':
                self.implemented_properties.extend(["energy" ,"free_energy"])
            else:
                self.implemented_properties.append(save_name)
        self.max_neighbors = getattr(model.readout_fn, "max_neighbors", None)
        self.cutoff = float(model.readout_fn.cutoff.item())
        self.element = TorchElement([int(z) for z in model.readout_fn.atomic_numbers.cpu().tolist()])
        self.neighborlist_backend = neighborlist_backend
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.keySpecification = KeySpecification()
        update_keyspec_from_kwargs(self.keySpecification, KEYS)
 
        if level is not None:
            self.level = level
            model.reset_computing_level(level) 
        else:
            self.level = model.get_computing_level()

        if spin_on is not None:
            self.spin_on = 1 if spin_on else 0
            model.reset_spin_on(self.spin_on) 
        else:
            self.spin_on = model.get_spin_on() 

        self.model = model.to(self.device)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        atoms.info['level'] = self.level # fidelity level
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
                universal_embedding=self.model.universal_embedding,
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