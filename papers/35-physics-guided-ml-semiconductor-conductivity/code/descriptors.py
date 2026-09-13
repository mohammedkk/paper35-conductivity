"""Electronic-structure descriptor table for the synthetic conductivity benchmark.

Band gaps, effective masses, Debye temperatures, and lattice thermal
conductivities below are representative literature-typical values for each
material (compiled from standard semiconductor physics references such as
Ashcroft & Mermin and common materials-property tabulations), used here as a
convenient, physically reasonable input set for the demonstration benchmark
in ``physics.py``. They are not claimed to be exact for any single sample or
source, and effective masses are simplified conductivity-relevant averages
rather than precise multi-valley density-of-states masses. Electronegativity
differences and average atomic masses ARE computed exactly, from standard
Pauling electronegativities and atomic weights.
"""

from dataclasses import dataclass, field

# Pauling electronegativity and atomic mass (amu) for the elements used below.
# Standard periodic-table values.
ELEMENTS = {
    "Si": (1.90, 28.09),
    "Ge": (2.01, 72.63),
    "C":  (2.55, 12.01),
    "Ga": (1.81, 69.72),
    "As": (2.18, 74.92),
    "N":  (3.04, 14.01),
    "P":  (2.19, 30.97),
    "Sb": (2.05, 121.76),
    "In": (1.78, 114.82),
    "Al": (1.61, 26.98),
    "Zn": (1.65, 65.38),
    "O":  (3.44, 16.00),
    "S":  (2.58, 32.07),
    "Se": (2.55, 78.97),
    "Cd": (1.69, 112.41),
    "Te": (2.10, 127.60),
    "Pb": (2.33, 207.20),
    "Bi": (2.02, 208.98),
}


@dataclass
class Material:
    name: str
    formula: list          # constituent element symbols
    Eg: float               # band gap, eV
    me_star: float          # electron conductivity effective mass, m0
    mh_star: float          # hole conductivity effective mass, m0
    theta_D: float          # Debye temperature, K
    kappa_L: float          # lattice thermal conductivity at 300 K, W/m/K
    narrow_gap: bool = field(init=False)

    def __post_init__(self):
        self.narrow_gap = self.Eg < 0.5

    @property
    def electronegativity_diff(self):
        chis = [ELEMENTS[el][0] for el in self.formula]
        return max(chis) - min(chis) if len(chis) > 1 else 0.0

    @property
    def atomic_mass_avg(self):
        masses = [ELEMENTS[el][1] for el in self.formula]
        return sum(masses) / len(masses)


MATERIALS = [
    Material("Si",      ["Si"],       1.12, 0.260, 0.390, 645.0, 150.0),
    Material("Ge",      ["Ge"],       0.66, 0.120, 0.290, 374.0,  60.0),
    Material("Diamond", ["C"],        5.47, 0.570, 0.800, 2200.0, 2200.0),
    Material("GaAs",    ["Ga", "As"], 1.42, 0.063, 0.510, 360.0,  55.0),
    Material("GaN",     ["Ga", "N"],  3.40, 0.200, 1.400, 600.0, 130.0),
    Material("GaP",     ["Ga", "P"],  2.26, 0.820, 0.600, 445.0,  77.0),
    Material("GaSb",    ["Ga", "Sb"], 0.73, 0.039, 0.400, 265.0,  33.0),
    Material("InP",     ["In", "P"],  1.34, 0.080, 0.600, 321.0,  68.0),
    Material("InAs",    ["In", "As"], 0.36, 0.023, 0.410, 280.0,  27.0),
    Material("InSb",    ["In", "Sb"], 0.17, 0.0135, 0.430, 200.0, 18.0),
    Material("AlAs",    ["Al", "As"], 2.16, 0.500, 0.500, 417.0,  90.0),
    Material("AlSb",    ["Al", "Sb"], 1.60, 0.140, 0.400, 292.0,  57.0),
    Material("AlN",     ["Al", "N"],  6.20, 0.400, 3.500, 1150.0, 285.0),
    Material("ZnO",     ["Zn", "O"],  3.37, 0.240, 0.590, 416.0,  40.0),
    Material("ZnS",     ["Zn", "S"],  3.60, 0.280, 0.490, 340.0,  27.0),
    Material("ZnSe",    ["Zn", "Se"], 2.70, 0.170, 0.600, 340.0,  19.0),
    Material("CdS",     ["Cd", "S"],  2.42, 0.210, 0.800, 219.0,  20.0),
    Material("CdSe",    ["Cd", "Se"], 1.74, 0.130, 0.450, 181.0,   9.0),
    Material("CdTe",    ["Cd", "Te"], 1.50, 0.110, 0.350, 200.0,   7.5),
    Material("PbTe",    ["Pb", "Te"], 0.32, 0.240, 0.310, 136.0,   2.3),
    Material("PbSe",    ["Pb", "Se"], 0.28, 0.070, 0.340, 120.0,   2.0),
    Material("PbS",     ["Pb", "S"],  0.41, 0.080, 0.300, 210.0,   2.5),
    Material("4H-SiC",  ["Si", "C"],  3.26, 0.290, 1.000, 1200.0, 370.0),
    Material("Bi2Te3",  ["Bi", "Te"], 0.15, 0.320, 0.320, 155.0,   1.5),
]

# Nominal doping conditions shared across materials (cm^-3). 0.0 stands in
# for the intrinsic / undoped condition.
DOPING_LEVELS = [0.0, 1.0e17, 1.0e19]

DESCRIPTOR_NAMES = [
    "Eg", "me_star", "mh_star", "DOS_Ef_proxy", "log_Nd",
    "theta_D", "atomic_mass_avg", "electronegativity_diff", "kappa_L",
]
