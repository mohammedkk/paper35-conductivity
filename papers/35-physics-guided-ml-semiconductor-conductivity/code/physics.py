"""Reduced Boltzmann-transport (relaxation-time approximation) forward model.

This module is the *ground-truth generator* used to synthesize the benchmark:
given a material's electronic-structure descriptors and a doping level, it
computes sigma(T) from a physically motivated but deliberately reduced
transport model (single acoustic-phonon / ionized-impurity scattering
channels combined via Matthiessen's rule, non-degenerate charge-neutrality
carrier statistics). It is a simplified, pedagogical transport model
calibrated to give the right order of magnitude for common semiconductors
(e.g. intrinsic Si mobility ~1000-1500 cm^2/V/s, doped Si conductivity of a
few to a few tens of S/cm at Nd ~ 1e17 cm^-3) -- it is NOT a substitute for
first-principles (DFT + full Boltzmann transport) calculations or digitized
experimental measurements, and no such digitized experimental data is used
anywhere in this repository. It exists solely to produce a self-consistent,
physically grounded synthetic benchmark for evaluating the learning
architecture in ``models.py``.

All energies in eV, temperatures in K, Boltzmann constant in eV/K.
"""

import numpy as np

K_B_EV = 8.617333262e-5   # Boltzmann constant, eV/K
T0 = 300.0                # reference temperature, K
Q_CHARGE = 1.602176634e-19  # elementary charge, C

# Calibration constants for the reduced mobility model (see README for the
# Si-based calibration this was fit to).
A0_ACOUSTIC = 22.7
B0_IMPURITY = 2.0e20

# Narrow-gap materials get an extra smooth high-temperature correction
# representing physics the single-power-law transport model does not
# capture explicitly (e.g. a second conduction channel or non-parabolicity
# becoming relevant at high T). This is what motivates a learned residual
# correction on top of the closed-form transport layer.
BUMP_GAMMA = 0.20
BUMP_CENTER = 500.0
BUMP_WIDTH = 100.0


def intrinsic_carrier_density(T, Eg, me_star, mh_star):
    """n_i(T) via the standard effective-density-of-states approximation
    (e.g. Sze, *Physics of Semiconductor Devices*), N_c(300K, m0) = N_v(300K, m0)
    = 2.5e19 cm^-3."""
    prefactor = 2.5e19 * (me_star * mh_star) ** 0.75
    return prefactor * (T / T0) ** 1.5 * np.exp(-Eg / (2.0 * K_B_EV * T))


def carrier_density(T, Eg, me_star, mh_star, Nd):
    """Non-degenerate charge-neutrality approximation combining an extrinsic
    plateau near Nd with the intrinsic exponential rise at high T."""
    ni = intrinsic_carrier_density(T, Eg, me_star, mh_star)
    return Nd / 2.0 + np.sqrt((Nd / 2.0) ** 2 + ni ** 2)


def mobility(T, me_star, theta_D, Nd):
    """Combine acoustic-phonon-limited and ionized-impurity-limited mobility
    via Matthiessen's rule, using m* = me_star as the relevant conduction
    effective mass (n-type doping is assumed throughout for simplicity)."""
    inv_mu_ac = (me_star ** 2.5) / (A0_ACOUSTIC * (theta_D / 400.0) ** 1.5) * (T / T0) ** 1.5
    inv_mu_ii = Nd * np.sqrt(me_star) / B0_IMPURITY * (T / T0) ** -1.5
    return 1.0 / (inv_mu_ac + inv_mu_ii)


def high_temperature_bump(T, narrow_gap):
    """Extra smooth multiplicative correction for narrow-gap materials,
    switched on above ~500 K, representing transport physics beyond the
    single-power-law reduced model (see module docstring)."""
    if not narrow_gap:
        return np.ones_like(np.asarray(T, dtype=float))
    x = (np.asarray(T, dtype=float) - BUMP_CENTER) / BUMP_WIDTH
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    return 1.0 + BUMP_GAMMA * sigmoid


def conductivity(T, material, Nd):
    """sigma(T) in S/cm for a given Material (see descriptors.py) and doping
    level Nd (cm^-3), including the narrow-gap high-temperature correction."""
    T = np.asarray(T, dtype=float)
    n = carrier_density(T, material.Eg, material.me_star, material.mh_star, Nd)
    mu = mobility(T, material.me_star, material.theta_D, Nd)
    sigma = n * Q_CHARGE * mu
    sigma = sigma * high_temperature_bump(T, material.narrow_gap)
    # Numerical floor only, many orders of magnitude below anything treated
    # as "measurable" (see MEASURABLE_FLOOR in dataset.py). Its only purpose
    # is to prevent a handful of extreme-gap/low-T points from underflowing
    # IEEE-754 double precision to exact 0.0, which would make log(sigma)
    # undefined; such points are excluded from training and evaluation
    # entirely rather than fit at this value (fitting a hard floor would
    # introduce a non-physical kink no smooth transport law can represent).
    return np.maximum(sigma, 1e-250)


def dos_at_fermi_proxy(material, Nd):
    """Simple descriptor-level proxy for density of states near the Fermi
    level: grows with effective mass and with the free-carrier density at
    300 K. This is an approximate feature for the ML input vector, not a
    DFT-computed density of states."""
    n300 = max(carrier_density(300.0, material.Eg, material.me_star, material.mh_star, Nd), 1e10)
    return np.sqrt(material.me_star * material.mh_star) * np.log10(n300)
