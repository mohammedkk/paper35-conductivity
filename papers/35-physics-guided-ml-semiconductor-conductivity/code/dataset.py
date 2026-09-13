"""Builds the synthetic benchmark: (material, doping, temperature) -> conductivity.

Every sigma(T) curve is generated from the reduced transport model in
``physics.py``, then perturbed with (a) a per-curve log-normal scale factor
representing sample-to-sample variability and (b) per-point log-normal noise
representing measurement-level scatter. Curves, not individual points, are
the unit of train/validation/test splitting, so evaluation always requires
generalizing across materials and doping conditions rather than interpolating
within an observed curve.
"""

import numpy as np

from descriptors import MATERIALS, DOPING_LEVELS, DESCRIPTOR_NAMES
import physics

T_MIN, T_MAX, N_TEMPS = 80.0, 800.0, 25
EXTRAPOLATION_SPLIT_T = 400.0

CURVE_NOISE_STD = 0.08   # log-space, per curve
POINT_NOISE_STD = 0.05   # log-space, per point

# Below this, a sample reads as "immeasurably insulating" on typical
# characterization equipment; such points are excluded from training and
# evaluation rather than fit at an arbitrary floor value (see physics.py).
MEASURABLE_FLOOR = 1.0e-9


def descriptor_vector(material, Nd):
    return np.array([
        material.Eg,
        material.me_star,
        material.mh_star,
        physics.dos_at_fermi_proxy(material, Nd),
        np.log10(Nd + 1.0e5),
        material.theta_D,
        material.atomic_mass_avg,
        material.electronegativity_diff,
        material.kappa_L,
    ], dtype=np.float64)


class Curve:
    """One (material, doping) sigma(T) curve."""

    def __init__(self, material, Nd, T, sigma_true, sigma_obs, x):
        self.material = material
        self.Nd = Nd
        self.T = T
        self.sigma_true = sigma_true
        self.sigma_obs = sigma_obs
        self.x = x
        self.m_star = material.me_star + material.mh_star
        self.Eg = material.Eg
        ni = physics.intrinsic_carrier_density(T, material.Eg, material.me_star, material.mh_star)
        self.regime = np.where(ni > 3.0 * Nd, 1, np.where(ni < Nd / 3.0, -1, 0))
        self.measurable = sigma_true > MEASURABLE_FLOOR

    @property
    def curve_id(self):
        return f"{self.material.name}_Nd{self.Nd:.0e}"


def build_benchmark(seed=0):
    rng = np.random.default_rng(seed)
    T = np.linspace(T_MIN, T_MAX, N_TEMPS)
    curves = []
    for material in MATERIALS:
        for Nd in DOPING_LEVELS:
            sigma_true = physics.conductivity(T, material, Nd)
            curve_scale = np.exp(rng.normal(0.0, CURVE_NOISE_STD))
            point_noise = np.exp(rng.normal(0.0, POINT_NOISE_STD, size=T.shape))
            sigma_obs = sigma_true * curve_scale * point_noise
            x = descriptor_vector(material, Nd)
            curves.append(Curve(material, Nd, T, sigma_true, sigma_obs, x))
    return curves


def split_curves(curves, seed=0, train_frac=0.70, val_frac=0.15):
    """Split at the *material* level (all doping conditions of a given
    material stay together), so test materials are genuinely unseen rather
    than differing from a training curve only in doping level."""
    rng = np.random.default_rng(seed)
    material_names = sorted({c.material.name for c in curves})
    idx = rng.permutation(len(material_names))
    n_train = int(round(train_frac * len(material_names)))
    n_val = int(round(val_frac * len(material_names)))
    train_names = {material_names[i] for i in idx[:n_train]}
    val_names = {material_names[i] for i in idx[n_train:n_train + n_val]}
    test_names = {material_names[i] for i in idx[n_train + n_val:]}
    train = [c for c in curves if c.material.name in train_names]
    val = [c for c in curves if c.material.name in val_names]
    test = [c for c in curves if c.material.name in test_names]
    return train, val, test


def flatten(curves, T_max=None, T_min=None):
    """Flatten a list of curves into point-level arrays, optionally
    restricting to a temperature window (used for the extrapolation split)."""
    xs, Ts, ys_true, ys_obs, curve_ids, m_stars = [], [], [], [], [], []
    for c in curves:
        mask = c.measurable.copy()
        if T_max is not None:
            mask &= c.T <= T_max
        if T_min is not None:
            mask &= c.T >= T_min
        for T, s_true, s_obs in zip(c.T[mask], c.sigma_true[mask], c.sigma_obs[mask]):
            xs.append(c.x)
            Ts.append(T)
            ys_true.append(s_true)
            ys_obs.append(s_obs)
            curve_ids.append(c.curve_id)
            m_stars.append(c.m_star)
    return {
        "x": np.array(xs),
        "T": np.array(Ts),
        "sigma_true": np.array(ys_true),
        "sigma_obs": np.array(ys_obs),
        "curve_id": np.array(curve_ids),
        "m_star": np.array(m_stars),
    }


def normalize_descriptors(x_train, x_other_list):
    """Standardize descriptors using train-set statistics only."""
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True) + 1e-8
    out = [(x_train - mu) / sigma]
    for x in x_other_list:
        out.append((x - mu) / sigma)
    return out, (mu, sigma)


if __name__ == "__main__":
    curves = build_benchmark()
    print(f"{len(curves)} curves x {N_TEMPS} temperatures = {len(curves) * N_TEMPS} samples")
    print(f"Descriptors: {DESCRIPTOR_NAMES}")
    tr, va, te = split_curves(curves)
    print(f"Split: {len(tr)} train / {len(va)} val / {len(te)} test curves")
