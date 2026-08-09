################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from scipy.constants import (
    Avogadro,
    Boltzmann,
    Planck,
    R,
    angstrom,
    bar,
    c,
    calorie,
    e,
    epsilon_0,
    eV,
    femto,
    hbar,
    nano,
    pi,
    pico,
)

kcalPerMol = 1000 * calorie / Avogadro


# === unit convert ===
kcalpermol2ev = kcalPerMol / eV
ev2kcalpermol = eV / kcalPerMol

pa2eva3 = (1 / eV) / (1e30)
kbar2eva3 = 1e8 * pa2eva3
