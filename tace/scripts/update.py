################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
# TODO not for users now
import re
import yaml
import argparse
from pathlib import Path


import torch


from ..lightning import load_tace
from ..dataset.element import atomic_numbers

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model path, *.ckpt, *.pt, *.pth",
    )
    parser.add_argument(
        "-u", "--update",
        type=str,
        nargs='+',  
        choices=["atomic_energy", 'scale', 'shift'],
        # choices=["atomic_energy", 'avg_num_neighbors', 'scale', 'shift']
        default=None,
        help="specify statistics info to update",
    )
    parser.add_argument(
        "-t", "--to",
        type=str, 
        choices=["OMat24", 'MPtrj'],
        help="to predfine statistics",
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.to is None or args.update is None

    model = load_tace(
        args.model,
        device='cpu',
        strict=True,
        use_ema=1,
    )

    # == args.to ===
    with torch.no_grad():
        if args.to is not None:
            model.readout_fn.atomic_energy_layer.atomic_energy.copy_(
                [
                    [v for _, v in args.to['atomic_energy'].items()]
                ]
            )    
    
    # === args.update ===
    pattern = re.compile(r"statistics_(\d+)\.yaml")
    statistics_dict = {}
    for path in Path(".").iterdir():
        if path.is_file():
            m = pattern.fullmatch(path.name)
            if m:
                level = int(m.group(1))
                statistics_dict[level] = yaml.safe_load(path)
    statistics_dict: dict = dict(sorted(statistics_dict.items()))
    with torch.no_grad():
        if args.to is not None:
            model.readout_fn.atomic_energy_layer.atomic_energy.copy_(
                [
                    [v for _, v in args.to['atomic_energy'].items()]
                ]
            )    

        if 'atomic_energies' in args.update:
            atomic_energies = model.readout_fn.atomic_energy_layer.atomic_energy.detach().clone()
            for level, stats in statistics_dict.items():
                if "atomic_energy" not in stats:
                    continue
                new_row = torch.tensor(
                    [v for _, v in sorted(stats["atomic_energy"].items())],
                    dtype=atomic_energies.dtype,
                    device=atomic_energies.device,
                )
                atomic_energies[level, :] = new_row
            model.readout_fn.atomic_energy_layer.atomic_energy.copy_(atomic_energies)

        if 'scale' in args.update:
            scales = model.readout_fn.scale_shift.scale.detach().clone()
            for level, stats in statistics_dict.items():
                if "scale" not in stats:
                    continue
                new_row = torch.tensor(
                    [v for _, v in sorted(stats["scale"].items())],
                    dtype=scales.dtype,
                    device=scales.device,
                )
                scales[level, :] = new_row
            model.readout_fn.scale_shift.scale.copy_(scales)
        
        if 'shift' in args.update:
            shifts = model.readout_fn.scale_shift.shift.detach().clone()
            for level, stats in statistics_dict.items():
                if "shift" not in stats:
                    continue
                new_row = torch.tensor(
                    [v for _, v in sorted(stats["shift"].items())],
                    dtype=shifts.dtype,
                    device=shifts.device,
                )
                shifts[level, :] = new_row
            model.readout_fn.scale_shift.shift.copy_(shifts)

    torch.save(model, "new_statistics.pt")


OMat24 = {
    'atomic_energy': {
        'H': -1.11700253,
        'He': 0.00079886,
        'Li': -0.29731164,
        'Be': -0.04129868,
        'B': -0.29106192,
        'C': -1.27751531,
        'N': -3.12342715,
        'O': -1.54797136,
        'F': -0.43969356,
        'Ne': -0.01250908,
        'Na': -0.22855413,
        'Mg': -0.00943179,
        'Al': -0.21707638,
        'Si': -0.82619133,
        'P': -1.88667434,
        'S': -0.89093583,
        'Cl': -0.25816211,
        'Ar': -0.02414768,
        'K': -0.17662425,
        'Ca': -0.02568319,
        'Sc': -2.13001165,
        'Ti': -2.38688845,
        'V': -3.55934233,
        'Cr': -5.44700879,
        'Mn': -5.14749562,
        'Fe': -3.30662847,
        'Co': -1.42167737,
        'Ni': -0.63181379,
        'Cu': -0.23449167,
        'Zn': -0.01146636,
        'Ga': -0.21291259,
        'Ge': -0.77939897,
        'As': -1.70148487,
        'Se': -0.78386705,
        'Br': -0.22690657,
        'Kr': -0.02245409,
        'Rb': -0.16092396,
        'Sr': -0.02798717,
        'Y': -2.25685695,
        'Zr': -2.23690495,
        'Nb': -2.15347771,
        'Mo': -4.60251809,
        'Tc': -3.36416792,
        'Ru': -2.23062607,
        'Rh': -1.15550917,
        'Pd': -1.47553527,
        'Ag': -0.19918102,
        'Cd': -0.01475888,
        'In': -0.19767692,
        'Sn': -0.68005773,
        'Sb': -1.43073368,
        'Te': -0.65790462,
        'I': -0.18915279,
        'Xe': -0.01179476,
        'Cs': -0.13507902,
        'Ba': -0.03056979,
        'La': -0.36017439,
        'Ce': -0.86279246,
        'Pr': -0.20573327,
        'Nd': -0.2734463,
        'Pm': -0.20046965,
        'Sm': -0.25444338,
        'Eu': -8.37972664,
        'Gd': -9.58424928,
        'Tb': -0.19466184,
        'Dy': -0.24860115,
        'Ho': -0.19531288,
        'Er': -0.15401392,
        'Tm': -0.14577898,
        'Yb': -0.19655747,
        'Lu': -0.15645898,
        'Hf': -3.49380556,
        'Ta': -3.5317097,
        'W': -4.57108006,
        'Re': -4.63425205,
        'Os': -2.88247063,
        'Ir': -1.45679675,
        'Pt': -0.50290184,
        'Au': -0.18521704,
        'Hg': -0.01123956,
        'Tl': -0.17483649,
        'Pb': -0.63132037,
        'Bi': -1.3248562,
        'Ac': -0.24135757,
        'Th': -1.04601971,
        'Pa': -2.04574044,
        'U': -3.84544799,
        'Np': -7.28626119,
        'Pu': -7.3136314,
    },
}

MPtrj = {
    'atomic_energy': {
        'H': -1.1176,
        'He': -0.0005,
        'Li': -0.2974,
        'Be': -0.0181,
        'B': -0.4447,
        'C': -1.3865,
        'N': -3.1256,
        'O': -1.9067,
        'F': -0.7674,
        'Ne': -0.0121,
        'Na': -0.2285,
        'Mg': -0.0958,
        'Al': -0.3122,
        'Si': -0.8689,
        'P': -1.8879,
        'S': -1.0746,
        'Cl': -0.3714,
        'Ar': -0.0502,
        'K': -0.2277,
        'Ca': -0.0927,
        'Sc': -2.2127,
        'Ti': -2.6397,
        'V': -3.7438,
        'Cr': -5.6018,
        'Mn': -5.3235,
        'Fe': -3.5955,
        'Co': -2.1496,
        'Ni': -1.0536,
        'Cu': -0.6027,
        'Zn': -0.1645,
        'Ga': -0.4043,
        'Ge': -0.8916,
        'As': -1.6834,
        'Se': -0.8716,
        'Br': -0.2651,
        'Kr': -0.0331,
        'Rb': -0.1879,
        'Sr': -0.068,
        'Y': -2.2868,
        'Zr': -2.3603,
        'Nb': -3.1513,
        'Mo': -4.6011,
        'Tc': -3.5438,
        'Ru': -1.6595,
        'Rh': -1.6479,
        'Pd': -1.4776,
        'Ag': -0.3388,
        'Cd': -0.1672,
        'In': -0.4087,
        'Sn': -0.8167,
        'Sb': -1.4107,
        'Te': -0.7239,
        'I': -0.1703,
        'Xe': -0.0097,
        'Cs': -0.1369,
        'Ba': -0.0344,
        'La': -0.8455,
        'Ce': -1.3876,
        'Pr': -0.5491,
        'Nd': -0.5186,
        'Pm': -0.4895,
        'Sm': -0.4683,
        'Eu': -8.3662,
        'Gd': -10.4088,
        'Tb': -0.3982,
        'Dy': -0.3886,
        'Ho': -0.3834,
        'Er': -0.3857,
        'Tm': -0.3168,
        'Yb': -0.064,
        'Lu': -0.3808,
        'Hf': -3.527,
        'Ta': -3.7421,
        'W': -4.6555,
        'Re': -3.4276,
        'Os': -2.8979,
        'Ir': -1.1789,
        'Pt': -0.5638,
        'Au': -0.2872,
        'Hg': -0.1235,
        'Tl': -0.3606,
        'Pb': -0.7674,
        'Bi': -1.326,
        'Ac': -0.3866,
        'Th': -1.1045,
        'Pa': -2.553,
        'U': -4.9889,
        'Np': -7.7017,
        'Pu': -10.8084,
    },
}