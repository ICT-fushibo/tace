################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import warnings
from typing import Optional, List


import torch
from ase import units
from ase.stress import full_3x3_to_voigt_6_stress
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
    multi-head, etc. For some advanced features, you need to modify the code yourself and store the attributes that need 
    to be embedded in atoms.info or atmoity.arrays. If you only need to predict, you can directly use the `tace-eval` 
    command. It will output the predicted files, and if you add the `--test` option, it will also output the errors.

    Parameters
    ----------
    model_path : str
        Path to the trained model, file ends with ``pt, .pth or .ckpt``.
    device : str, default='cpu'
        The device to run computations on, e.g., ``cpu`` or ``cuda``.
    dtype : str, optional, default=None
        Data type for computations, e.g., ``float32`` or ``float64``.
    extra_compute_first_derivative : list[str], optional, default=None
        If you wand to predict property not trained in your model, 
        You need to provide the names of the first-order derivative physical quantities for additional predictions.
        For example, if model trained on energy only, you colud also predict forces, and stress.
    extra_compute_second_derivative : list[str], optional, default=None
        If you wand to predict property not trained in your model, 
        You need to provide the names of the second-order derivative physical quantities for additional predictions.
        For example, if model trained on energy forces only, you colud also predict hessians.
        One another example is that if model trained on conservative_dipole under electric_field, you colud also predict 
        conservative_polarizability.
    dispersion : bool, default=False
        You can first create TACEAseCalc with no dispersion, 
        and then use tace.interface.ase.add_dispersion_to_calc to obtain a calculator with dispersion correction.
        This argument must always be False.
    level : int
        Specify which fidelity level to use. The default is a single head,
        i.e., the index of fidelity `level` defaults to 0.
    **kwargs
        Additional keyword arguments passed to the ASE Calculator base class.
    """

    def __init__(
        self,
        model: str,
        device: str = "cpu",
        dtype: Optional[str] = None,
        extra_compute_first_derivative: Optional[List[str]] = None,
        extra_compute_second_derivative: Optional[List[str]] = None,
        dispersion: bool = False,
        level: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if dispersion:
            raise (
                'You can first create TACEAseCalc with no dispersion, '
                'and then use tace.interface.ase.add_dispersion to obtain '
                'a calculator with dispersion correction, argument dispersion in '
                'TACEAseCalc sholud always be set to False.'
            )
        self.extra_compute_first_derivative = extra_compute_first_derivative or []
        self.extra_compute_second_derivative = extra_compute_second_derivative or []        
        model = load_tace(model, device, strict=True, use_ema=True)
        
        model_dtype = model.readout_fn.cutoff.dtype
        if dtype is not None:
            if dtype == "float64":
                torch.set_default_dtype(torch.float64)
                model = model.double()
                if model_dtype != torch.float64:
                    warnings.warn(
                        f"Model dtype {model_dtype} != default dtype {dtype}. "
                        f"This may cause silent type conversions."
                    )
            elif dtype == "float32":
                torch.set_default_dtype(torch.float32)
                model = model.float()
                if model_dtype != torch.float32:
                    warnings.warn(
                        f"Model dtype {model_dtype} != default dtype {dtype}. "
                        f"This may cause silent type conversions."
                    )
            else:
                raise ValueError(f"Unknown dtype {dtype}")
        else:
            torch.set_default_dtype(model_dtype)

        target_property = model.target_property
        compute_flags = {}
        for p in self.extra_compute_first_derivative:
            model.compute_first_derivative = True
            compute_flags.update(
                {
                    p: True
                }
            )
        for p in self.extra_compute_second_derivative:
            model.compute_second_derivative = True
            compute_flags.update(
                {
                    p: True
                }
            )
        for p, flag in compute_flags.items():
            if flag:
                setattr(model.flags, f"compute_{p}", True)
                target_property.append(p)
        self.target_property = list(set(target_property))
        self.embedding_property = model.embedding_property
        self.implemented_properties = self.target_property + ["free_energy"]
        self.universal_embedding = model.universal_embedding
        self.max_neighbors = getattr(model.readout_fn, "max_neighbors", None)
        self.cutoff = float(model.readout_fn.cutoff.item())
        self.element = TorchElement([int(z) for z in model.readout_fn.atomic_numbers.cpu().tolist()])

        for param in model.parameters():
            param.requires_grad = False

        self.keySpecification = KeySpecification()
        update_keyspec_from_kwargs(self.keySpecification, KEYS)

        self.dtype = DTYPE[dtype or model_dtype]
        self.device = DEVICE[device]
        self.level = level
        model.level = level    
        self.model = model.to(device)

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
                universal_embedding=self.universal_embedding,
                training=False,
            ) 
        ]
        dataloader = DataLoader(
            dataset=data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batch = next(iter(dataloader)).to(self.device)
        for p in self.target_property:
            for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                batch[requires_grad_p].requires_grad_(True)

        # === forward ===
        outs = self.model(batch)
        # === update ===
        self.results = {}
        for p in self.target_property:
            p_type = PROPERTY[p]['type']
            p_rank = PROPERTY[p]['rank']
            prop = outs[p]
            if p_type == 'graph':
                if p_rank == 0:
                    if p == 'energy':
                        energy = prop.detach().cpu().item()
                        self.results[p] = energy
                        self.results["free_energy"] = self.results[p]
                    else:
                        self.results[p] = prop.detach().cpu().item()
                elif set([p]) & {'stress', 'virials'} :
                    prop = prop.detach().cpu().numpy().squeeze(0)
                    prop = full_3x3_to_voigt_6_stress(prop)
                    self.results[p] = prop
                else:
                    self.results[p] = outs[p].detach().cpu().numpy().squeeze(0)
            elif p_type == 'atom':
                self.results[p] = prop.detach().cpu().numpy()
            else:
                raise

            
    def get_hessians(self, atoms=None):
        self.target_property = list(set(self.target_property+'hessians'))
        self.model.compute_forces = True
        self.model.compute_hessians = True
        self.model.compute_first_derivative = True

        data = [
            from_atoms(
                self.element,
                atoms,
                self.cutoff,
                max_neighbors=self.max_neighbors,
                target_property=self.target_property,
                embedding_property=self.embedding_property,
                keyspec=self.keySpecification,
                universal_embedding=self.universal_embedding,
                training=False,
            ) 
        ]

        dataloader = DataLoader(
            dataset=data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batch = next(iter(dataloader))
        batch.to(self.device)
        for p in self.target_property:
            for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                batch[requires_grad_p].requires_grad_(True)
        outs = self.model(batch)

        return outs["hessians"].detach().cpu().numpy() 


    def get_direct_polarizability(self, atoms=None):
        self.target_property = list(set(self.target_property+'direct_polarizability'))
        data = [
            from_atoms(
                self.element,
                atoms,
                self.cutoff,
                max_neighbors=self.max_neighbors,
                target_property=self.target_property,
                embedding_property=self.embedding_property,
                keyspec=self.keySpecification,
                universal_embedding=self.universal_embedding,
                training=False,
            ) 
        ]

        dataloader = DataLoader(
            dataset=data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batch = next(iter(dataloader))
        batch.to(self.device)
        for p in self.target_property:
            for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                batch[requires_grad_p].requires_grad_(True)
        outs = self.model(batch)
        return outs['direct_polarizability'] .detach().cpu().numpy() 


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