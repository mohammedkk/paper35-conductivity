"""Physics-consistency ("shape") loss, Section 3.4.

Operates on curve-batched tensors of shape (n_curves, n_temps), all curves
sharing the same temperature grid, which makes the finite-difference slope
along the curve well defined and cheap to compute.
"""

import torch

K_B_EV = 8.617333262e-5


def _finite_diff(y, coord, dim):
    return torch.gradient(y, spacing=(coord,), dim=dim)[0]


def shape_consistency_loss(log_sigma_pred, T, regime, Eg_raw, measurable=None,
                            bound_weight=0.1, slope_bound=3.0):
    """
    log_sigma_pred : (n_curves, n_temps) predicted log_e sigma
    T              : (n_temps,) shared temperature grid, K
    regime         : (n_curves, n_temps) int tensor, +1 intrinsic-dominated,
                      -1 extrinsic-dominated, 0 ambiguous (not used for the
                      Arrhenius-slope term)
    Eg_raw         : (n_curves,) raw (unnormalized) band gap, eV
    measurable     : (n_curves, n_temps) bool tensor; points below the
                      benchmark's measurability floor are excluded, since
                      the physical target there is not reliable (see
                      dataset.py / MEASURABLE_FLOOR)
    """
    # Work in activation-energy (eV) units rather than raw d(ln sigma)/d(1/T)
    # slope units (~1/(2*k_B) ~ 5800 K), which would otherwise make the
    # squared-error term dominate the total loss by many orders of
    # magnitude. E_a,pred = -slope * 2*k_B is O(1) eV, directly comparable
    # to the descriptor's own band gap.
    inv_T = 1.0 / T
    slope_vs_inv_T = _finite_diff(log_sigma_pred, inv_T, dim=1)
    Ea_pred = -slope_vs_inv_T * 2.0 * K_B_EV
    intrinsic_mask = (regime == 1).float()
    if measurable is not None:
        intrinsic_mask = intrinsic_mask * measurable.float()
    n_intrinsic = intrinsic_mask.sum().clamp(min=1.0)
    loss_intrinsic = (((Ea_pred - Eg_raw.unsqueeze(1)) ** 2) * intrinsic_mask).sum() / n_intrinsic

    log_T = torch.log(T)
    slope_vs_logT = _finite_diff(log_sigma_pred, log_T, dim=1)
    bound_violation = torch.relu(slope_vs_logT.abs() - slope_bound) ** 2
    loss_bound = bound_violation.mean()

    return loss_intrinsic + bound_weight * loss_bound


def intrinsic_arrhenius_slopes(log_sigma, T, regime, measurable=None, min_points=3):
    """Per-curve least-squares slope of log_sigma vs 1/T restricted to the
    intrinsic-dominated window, used for the physical-consistency-violation
    metric at evaluation time. Returns (slopes, checkable_mask) as numpy
    arrays; curves with fewer than `min_points` intrinsic-dominated
    temperature points are marked not checkable.
    """
    import numpy as np

    log_sigma = np.asarray(log_sigma)
    T = np.asarray(T)
    regime = np.asarray(regime)
    inv_T = 1.0 / T
    n_curves = log_sigma.shape[0]
    slopes = np.full(n_curves, np.nan)
    checkable = np.zeros(n_curves, dtype=bool)
    meas = np.ones_like(regime, dtype=bool) if measurable is None else np.asarray(measurable)
    for i in range(n_curves):
        mask = (regime[i] == 1) & meas[i]
        if mask.sum() >= min_points:
            A = np.stack([inv_T[mask], np.ones(mask.sum())], axis=1)
            coef, *_ = np.linalg.lstsq(A, log_sigma[i][mask], rcond=None)
            slopes[i] = coef[0]
            checkable[i] = True
    return slopes, checkable
