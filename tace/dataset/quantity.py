################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


from .utils import (
    voigt_to_matrix,
    shape_fn_for_hessians,
    default_value_for_hessians,
    default_value_for_rank0_atom,
    default_value_for_rank1_atom,
    default_value_for_rank2_atom,
    default_value_for_rank3_atom,
    default_value_for_rank4_atom,
    default_value_for_rank0_graph,
    default_value_for_rank1_graph,
    default_value_for_rank2_graph,
    default_value_for_rank3_graph,
    default_value_for_rank4_graph,
)


PROPERTY = {
    "level": {
        "ase_name": None,
        'type': 'int',
        "scope": "per-system",
        "rank": 0,
        "abbreviation": "LEVEL",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "energy": {
        "ase_name": "energy",
        'type': 'float',
        "scope": "per-system",
        "rank": 0,
        "abbreviation": "E",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },

        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "forces": {
        "ase_name": "forces",
        'type': 'float',
        "scope": "per-atom",
        "rank": 1,
        "abbreviation": "F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": {
            1: ['energy'],
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False, 
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['positions'],
    },
    "edge_forces": {
        "ase_name":None,
        'type': 'float',
        "scope": "per-edge",
        "rank": 1,
        "abbreviation": "EDGE_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom, # placeholder
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": False, # can be embedded through uee to achice DeNs
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['edge_vector'],
    },
    "direct_forces": {
        "ase_name": "forces",
        'type': 'float',
        "scope": "per-atom",
        "rank": 1,
        "abbreviation": "D_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False, 
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "hessians": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-edge",
        "rank": 2,
        "abbreviation": "HESSIAN",
        "shape": {
            "in_data": (-1,),
            "shape_fn": shape_fn_for_hessians,
        },
        "default_value_fn": default_value_for_hessians,
        "must_be_with": {
            1: ['energy', 'forces'],
        },
        "conflict_with": {},
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ['positions'],
    },
    "direct_hessians": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-edge",
        "rank": 2,
        "abbreviation": "D_HESSIAN",
        "shape": {
            "in_data": (-1,),
            "shape_fn": shape_fn_for_hessians,
        },
        "default_value_fn": default_value_for_hessians,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "stress": {
        "ase_name": "stress",
        'type': 'float',
        "scope": "per-system",
        "rank": 2,
        "abbreviation": "S",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        # "requires_grad_with": ['displacement'],
        "requires_grad_with": [], # manual set
    },
    "direct_stress": {
        "ase_name": "stress",
        'type': 'float',
        "scope": "per-system",
        "rank": 2,
        "abbreviation": "D_S",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "virials": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 2,
        "abbreviation": "V",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        # "requires_grad_with": ['displacement'],
        "requires_grad_with": [], # manual set
    },
    "direct_virials": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 2,
        "abbreviation": "D_V",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "atomic_stresses": {
        "ase_name": "stresses",
        'type': 'float',
        "scope": "per-atom",
        "rank": 2,
        "abbreviation": "A_S",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['edge_vector'],
    },
    "atomic_virials": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-atom",
        "rank": 2,
        "abbreviation": "A_V",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['edge_vector'],
    },
    "direct_dipole": {
        "ase_name": "dipole",
        'type': 'float',
        "scope": "per-system",
        "rank": 1,
        "abbreviation": "D",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {
            1: ['charges'],
        },
        "conflict_with": {
            1: ["polarization"]
        },
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "conservative_dipole": {
        "ase_name": "dipole",
        'type': 'float',
        "scope": "per-system",
        "rank": 1,
        "abbreviation": "D",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {},
        "conflict_with": {
            1: ["polarization"]
        },
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['electric_field'],
    },
    "polarization": {
        "ase_name": "polarization",
        'type': 'float',
        "scope": "per-atom",
        "rank": 1,
        "abbreviation": "P",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": [], 
        "must_be_with": {},
        "conflict_with": {
            1: ["direct_dipole", "conservative_dipole"]
        },
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['electric_field'],
    },
    "direct_polarizability": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 2,
        "abbreviation": "ALPHA",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "conservative_polarizability": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 2,
        "abbreviation": "ALPHA",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {
            1: ["direct_dipole"],
            2: ["conservative_dipole"],
            3: ["polarization"]
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ['electric_field'],
    },
    "born_effective_charges": {
        "ase_name": "born_effective_charges",
        'type': 'float',
        "scope": "per-atom",
        "rank": 2,
        "abbreviation": "BEC",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_atom,
        "must_be_with": {
            1: ["direct_dipole"],
            2: ["conservative_dipole"],
            3: ["polarization"]
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ['electric_field', 'positions'],
    },
    "magnetization": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 1,
        "abbreviation": "M",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['magnetic_field'],
    },
    "magnetic_susceptibility": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 2,
        "abbreviation": "CHI_M",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {
            1: ['magnetization']
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ['magnetic_field'],
    },
    "charges": {
        "ase_name": "charges",
        'type': 'float',
        "scope": "per-atom",
        "rank": 0,
        "abbreviation": "C",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "total_charge": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 0,
        "type": "graph",
        "abbreviation": "TC",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "spin_multiplicity": {
        "ase_name": None,
        'type': 'int',
        "scope": "per-system",
        "rank": 0,
        "abbreviation": "SM",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "initial_collinear_magmoms": {
        "ase_name": "initial_magmoms",
        'type': 'float',
        "scope": "per-atom",
        "rank": 0,
        "abbreviation": "I_C_MAG",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "initial_noncollinear_magmoms": {
        "ase_name": "initial_magmoms",
        'type': 'float',
        "scope": "per-atom",
        "rank": 1,
        "abbreviation": "I_NC_MAG",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "final_collinear_magmoms": {
        "ase_name": "magmoms",
        'type': 'float',
        "scope": "per-atom",
        "rank": 0,
        "abbreviation": "F_C_MAG",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "final_noncollinear_magmoms": {
        "ase_name": "magmoms",
        'type': 'float',
        "scope": "per-atom",
        "rank": 1,
        "abbreviation": "F_NC_MAG",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "collinear_magnetic_forces": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-atom",
        "rank": 0,
        "abbreviation": "C_MAG_F",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": {
            1: ["initial_collinear_magmoms"]
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['initial_collinear_magmoms'],
    },
    "noncollinear_magnetic_forces": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-atom",
        "rank": 1,
        "abbreviation": "NC_MAG_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": {
            1: ["initial_noncollinear_magmoms"]
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['initial_noncollinear_magmoms'],
    },
    "total_collinear_magmom": {
        "ase_name": "magmoms",
        'type': 'float',
        "scope": "per-system",
        "rank": 0,
        "abbreviation": "TCM",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "total_noncollinear_magmom": {
        "ase_name": "magmoms",
        'type': 'float',
        "scope": "per-system",
        "rank": 1,
        "abbreviation": "TNCM",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "electric_field": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 1,
        "abbreviation": "EF",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "magnetic_field": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 1,
        "abbreviation": "MF",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "temperature": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 0,
        "abbreviation": "TEMP",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "electron_temperature": {
        "ase_name": None,
        'type': 'float',
        "scope": "per-system",
        "rank": 0,
        "abbreviation": "E_TEMP",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "spin_on": {
        "ase_name": None,
        'type': 'int',
        "scope": "per-system",
        "rank": 0,
        "abbreviation": "SPIN_ON",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
}

SUPPORT_PREDICT_PROPERTY = [k for k, v in PROPERTY.items() if v['enable_prediction']]
UNIVERSAL_EMBEDDING_ALLOWED_PROPERTY = [k for k, v in PROPERTY.items() if v['enable_embedding']]

KEYS = {f"{k}_key": k for k in PROPERTY}

# should be delete in future versions TODO
class DefaultKeys(Enum):
    # basic
    ENERGY = "energy"
    FORCES = "forces"
    STRESS = "stress"
    VIRIALS = "virials"

    HESSIANS = "hessians"
    EDGE_FORCES = "edge_forces"
    ATOMIC_VIRIALS = "atomic_virials"
    ATOMIC_STRESSES = "atomic_stresses"
    
    # direct property
    DIRECT_FORCES = "direct_forces"
    DIRECT_STRESS = "direct_stress"
    DIRECT_VIRIALS = "direct_virials"
    DIRECT_DIPOLE = "direct_dipole"
    DIRECT_POLARIZABILITY = "direct_polarizability"
    DIRECT_HESSIANS = "direct_hessians"

    # charges
    CHARGES = "charges"
    TOTAL_CHARGE = "total_charge"

    # external field
    ELECTRIC_FIELD = "electric_field"
    MAGNETIC_FIELD = "magnetic_field"

    CONSERVATIVE_DIPOLE = "conservative_dipole"
    CONSERVATIVE_POLARIZABILITY = "conservative_polarizability"
    BORN_EFFECTIVE_CHARGES = "born_effective_charges" # do not consider LES
    MAGNETIZATION = "magnetization"
    MAGNETIC_SUSCEPTIBILITY = "magnetic_susceptibility"
    POLARIZATION = "polarization"

    # MAG
    INITIAL_COLLINEAR_MAGMOMS = "initial_collinear_magmoms"
    INITIAL_NONCOLLINEAR_MAGMOMS = "initial_noncollinear_magmoms"
    FINAL_COLLINEAR_MAGMOMS = "final_collinear_magmoms"
    FINAL_NONCOLLINEAR_MAGMOMS = "final_noncollinear_magmoms"
    COLLINEAR_MAGNETIC_FORCES = "collinear_magnetic_forces"
    NONCOLLINEAR_MAGNETIC_FORCES = "noncollinear_magnetic_forces"
    TOTAL_COLLINEAR_MAGMOM = "total_collinear_magmom"
    TOTAL_NONCOLLINEAR_MAGMOM = "total_noncollinear_magmom"
    SPIN_ON = "spin_on"

    # only for embedding
    LEVEL = 'level'
    TEMPERATURE = "temperature"
    ELECTRON_TEMPERATURE = "electron_temperature"
    SPIN_MULTIPLICITY = "spin_multiplicity"

    @staticmethod
    def keydict() -> dict[str, str]:
        key_dict = {}
        for member in DefaultKeys:
            key_name = f"{member.name.lower()}_key"
            key_dict[key_name] = member.value
        return key_dict


@dataclass
class KeySpecification:
    '''Modify from MACE to simplify reading property'''
    info_keys: Dict[str, str] = field(default_factory=dict)
    arrays_keys: Dict[str, str] = field(default_factory=dict)

    def update(
        self,
        info_keys: Optional[Dict[str, str]] = None,
        arrays_keys: Optional[Dict[str, str]] = None,
    ):
        if info_keys is not None:
            self.info_keys.update(info_keys)
        if arrays_keys is not None:
            self.arrays_keys.update(arrays_keys)
        return self

    @classmethod
    def from_defaults(cls):
        instance = cls()
        return update_keyspec_from_kwargs(instance, DefaultKeys.keydict())


def update_keyspec_from_kwargs(
    keyspec: KeySpecification, keydict: Dict[str, str]
) -> KeySpecification:
    '''Modify from MACE to simplify reading property'''
    infos = [f"{k}_key" for k, v in PROPERTY.items() if (v['scope'] != 'per-atom')]
    arrays = [f"{k}_key" for k, v in PROPERTY.items() if (v['scope'] == 'per-atom')]
    info_keys = {}
    arrays_keys = {}
    for key in infos:
        if key in keydict:
            info_keys[key[:-4]] = keydict[key]
    for key in arrays:
        if key in keydict:
            arrays_keys[key[:-4]] = keydict[key]
    keyspec.update(info_keys=info_keys, arrays_keys=arrays_keys)
    return keyspec


def get_target_property(cfg: Dict) -> List[str]:
    """
    Automatically infer the physical quantities required for training from the loss function.
    Ensure no conflicting physical quantities appear simultaneously.
    """
    try:
        loss = str(cfg["loss"]["_target_"]).lower()
    except KeyError as e:
        raise KeyError("Missing 'cfg.loss._target_' field in configuration") from e
    loss_property = cfg["loss"].get("loss_property", None)
    if loss_property is None:
        logging.warning(
            "The argument `cfg.loss.loss_property` must be provided by the user. "
            "It is kept optional only for backward compatibility with earlier versions. "
            "Omitting it may lead to bugs when predicted physical property names share common prefixes or substrings."
        )
        loss_property = []
        for p in SUPPORT_PREDICT_PROPERTY:
            clean_property = p.replace("_", "")
            clean_property = p
            if clean_property.lower() in loss:
                loss_property.append(p)

        for p in loss_property:
            conflict_with = PROPERTY[p]['conflict_with']
            for _, conflict_ps in conflict_with.items():
                for conflict_p in conflict_ps:
                    if conflict_p in loss_property:
                        raise ValueError(
                            f"Conflict Property detected: {p} with {conflict_with} cannot be used together."
                        )
    else:
        assert set(loss_property).issubset(list(PROPERTY))

    return loss_property


def get_embedding_property(cfg: Dict, separate: bool = False) -> List[str] | Tuple[List[str]]:
    invariant = cfg["model"]["config"].get("universal_embedding", {}).get("invariant", {})
    invariant_embedding_property = []
    for k, v in invariant.items():
        if v.get('enable', False):
            invariant_embedding_property.append(k)

    equivariant = cfg["model"]["config"].get("universal_embedding", {}).get("equivariant", {})
    equivariant_embedding_property = []
    for k, v in equivariant.items():
        if v.get('enable', False):
            equivariant_embedding_property.append(k)

    if separate:
        return invariant_embedding_property, equivariant_embedding_property
    else:
        return invariant_embedding_property + equivariant_embedding_property


# For Metrics
MAE_PROPERTY = [
        p for p in SUPPORT_PREDICT_PROPERTY 
        if p != "polarization" 
        and p != "final_collinear_magmoms"
        and p != "hessians"
    ]
RMSE_PROPERTY = [   
        p for p in SUPPORT_PREDICT_PROPERTY 
        if p != "polarization" 
        and p != "final_collinear_magmoms"
        and p != "hessians"
    ]
MAE_PER_ATOM_PROPERTY = [
    p for p, v in PROPERTY.items() 
    if v["scope"] == "per-system" 
    and p != "polarization"
    and p != "final_collinear_magmoms"
    and p != "stress"   
    and p != "direct_stress"  
        and p != "hessians"
]
RMSE_PER_ATOM_PROPERTY = [
    p for p, v in PROPERTY.items() 
    if v["scope"] == "per-system" 
    and p != "polarization" 
    and p != "final_collinear_magmoms"
    and p != "stress"    
    and p != "direct_stress"   
    and p != "hessians" 
]


fields = {f"compute_{k}": False for k, v in PROPERTY.items()}
@dataclass
class ComputeFlag:
    __annotations__ = {k: bool for k in fields} 
    locals().update(fields)


from e3nn import o3
def get_target_irreps(
        target_property: List[str], 
        cart: bool = True
    ) -> List[int] | o3.Irreps:

    if cart:
        target_irreps = []
        target_irreps.extend([0]) if "energy" in target_property else None
        target_irreps.extend([0]) if "final_collinear_magmoms" in target_property else None
        target_irreps.extend([1]) if "final_noncollinear_magmoms" in target_property else None
        target_irreps.extend([0, 2]) if "direct_polarizability" in target_property else None
        target_irreps.extend([0, 2]) if "direct_stress" in target_property or \
            "direct_virials" in target_property else None
        target_irreps.extend([1]) if "direct_dipole" in target_property else None
        target_irreps.extend([1]) if "direct_forces" in target_property else None
        target_irreps.extend([0, 2]) if "direct_hessians" in target_property else None
        return sorted(list(set(target_irreps))) # TODO, do not use set(), like sph
    else:
        target_irreps_list = []
        target_irreps_list.extend(["1x0e"]) if "energy" in target_property else None
        target_irreps = o3.Irreps("+".join(target_irreps_list)).regroup()
        return target_irreps

