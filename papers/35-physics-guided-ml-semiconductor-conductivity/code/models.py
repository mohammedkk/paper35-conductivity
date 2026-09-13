"""Model architectures: the physics-guided model (PGML) and the black-box
MLP baseline (also used, with an added loss term, as the "residual-only
PINN" baseline in train.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

K_B_EV = 8.617333262e-5
T0 = 300.0


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers)


N_BASE_PARAMS = 5
N_CHANNEL2_PARAMS = 3


class PhysicsParameterHead(nn.Module):
    """Maps temperature-independent descriptors to interpretable transport
    parameters theta = (E_g_eff, C_n, N_0, tau_ac0, tau_ii0[, gamma_2,
    T_center2, T_width2]), each passed through an output activation that
    guarantees physical admissibility (Section 3.3 -- see
    transport_layer_log_sigma for the closed form these parameterize).

    The base five parameters give the carrier-concentration term and the
    mobility term their own free parameters, in the same *functional
    family* as the benchmark's own generator (Section 3.5) -- a smooth
    extrinsic/intrinsic blend via a soft-plus-like combination, and
    Matthiessen-combined acoustic-phonon ($T^{-3/2}$) and ionized-impurity
    ($T^{3/2}$) mobility channels -- rather than an a priori simplification
    of it. When `use_second_channel=True`, three more parameters describe
    an explicit second conduction channel (Section 3.3) -- a logistic
    high-temperature turn-on with learned amplitude, center, and width --
    again matching the generator's own narrow-gap correction structurally,
    in place of a generic bounded residual correction network.
    """

    def __init__(self, n_descriptors, hidden=(128, 128, 64), use_second_channel=True):
        super().__init__()
        self.use_second_channel = use_second_channel
        n_out = N_BASE_PARAMS + (N_CHANNEL2_PARAMS if use_second_channel else 0)
        self.net = _mlp([n_descriptors, *hidden, n_out])

    def forward(self, x):
        z = self.net(x)
        E_g_eff = F.softplus(z[..., 0]) + 1e-3       # eV, effective activation energy
        C_n = F.softplus(z[..., 1]) + 1e-6            # intrinsic carrier-density prefactor
        N_0 = F.softplus(z[..., 2]) + 1e-6            # extrinsic carrier-density floor
        tau_ac0 = F.softplus(z[..., 3]) + 1e-6        # acoustic-phonon mobility prefactor
        tau_ii0 = F.softplus(z[..., 4]) + 1e-6        # ionized-impurity mobility prefactor
        params = [E_g_eff, C_n, N_0, tau_ac0, tau_ii0]
        if self.use_second_channel:
            gamma_2 = torch.sigmoid(z[..., 5])                        # turn-on amplitude, [0, 1]
            T_center2 = 50.0 + 900.0 * torch.sigmoid(z[..., 6])        # turn-on center, [50, 950] K
            T_width2 = F.softplus(z[..., 7]) + 10.0                    # turn-on width, >= 10 K
            params += [gamma_2, T_center2, T_width2]
        return torch.stack(params, dim=-1)


def transport_layer_log_sigma(theta, T, m_star, eps=1e-30, use_second_channel=True):
    """Richer closed-form reduced transport expression: returns
    log_e(sigma_phys). Parameter-free given theta, T, and m_star.

    Carrier concentration blends an intrinsic exponential rise with an
    extrinsic floor via the standard non-degenerate charge-neutrality
    approximation n(T) = N_0/2 + sqrt((N_0/2)^2 + n_i(T)^2); mobility
    combines two independent power-law scattering channels via
    Matthiessen's rule, 1/mu(T) = (T/T0)^1.5/tau_ac0 + (T/T0)^-1.5/tau_ii0.
    If `use_second_channel`, a logistic high-temperature correction
    log(1 + gamma_2 * sigmoid((T - T_center2)/T_width2)) is added,
    structurally identical to the generator's narrow-gap turn-on
    (Section 3.5) but with learned rather than fixed amplitude, center,
    and width.
    """
    if use_second_channel:
        E_g_eff, C_n, N_0, tau_ac0, tau_ii0, gamma_2, T_center2, T_width2 = theta.unbind(-1)
    else:
        E_g_eff, C_n, N_0, tau_ac0, tau_ii0 = theta.unbind(-1)
    log_ni = torch.log(C_n) + 1.5 * torch.log(T / T0) - E_g_eff / (2.0 * K_B_EV * T)
    ni = torch.exp(log_ni)
    n = N_0 / 2.0 + torch.sqrt((N_0 / 2.0) ** 2 + ni ** 2)
    inv_mu = (T / T0) ** 1.5 / tau_ac0 + (T / T0) ** -1.5 / tau_ii0
    log_sigma = torch.log(n + eps) - torch.log(inv_mu + eps) - torch.log(m_star)
    if use_second_channel:
        log_sigma = log_sigma + torch.log1p(gamma_2 * torch.sigmoid((T - T_center2) / T_width2))
    return log_sigma


class ResidualCorrectionNet(nn.Module):
    """Small, magnitude-bounded correction network (Section 3.3)."""

    def __init__(self, n_descriptors, hidden=(64, 32), eta=0.15):
        super().__init__()
        self.net = _mlp([n_descriptors + 2, *hidden, 1])
        self.eta = eta

    def forward(self, x, T_norm, log_sigma_phys):
        inp = torch.cat([x, T_norm.unsqueeze(-1), log_sigma_phys.unsqueeze(-1)], dim=-1)
        delta = torch.tanh(self.net(inp).squeeze(-1))
        return delta


class PGML(nn.Module):
    """Physics-guided model: physics-parameter head -> closed-form transport
    layer (optionally including the explicit second conduction channel) ->
    optional bounded residual correction on top (Eq. 4). Both mechanisms
    for handling the narrow-gap high-temperature effects the base five-
    parameter closed form does not capture -- the explicit second channel
    (`use_second_channel`) and the generic residual correction network
    (`use_rcn`) -- can be toggled independently for ablation."""

    def __init__(self, n_descriptors, eta=0.15, use_rcn=False, use_second_channel=True):
        super().__init__()
        self.pph = PhysicsParameterHead(n_descriptors, use_second_channel=use_second_channel)
        self.use_rcn = use_rcn
        self.use_second_channel = use_second_channel
        if use_rcn:
            self.rcn = ResidualCorrectionNet(n_descriptors, eta=eta)

    def forward(self, x, T, T_norm, m_star):
        theta = self.pph(x)
        log_sigma_phys = transport_layer_log_sigma(theta, T, m_star, use_second_channel=self.use_second_channel)
        if self.use_rcn:
            delta = self.rcn(x, T_norm, log_sigma_phys)
            log_sigma_pred = log_sigma_phys + torch.log1p(self.rcn.eta * delta)
        else:
            delta = torch.zeros_like(log_sigma_phys)
            log_sigma_pred = log_sigma_phys
        return log_sigma_pred, log_sigma_phys, theta, delta


class BlackBoxMLP(nn.Module):
    """Black-box baseline: descriptors + T -> log_e sigma directly, with no
    physics-derived structure. Also used as the base network for the
    "residual-only PINN" baseline (physics enters only via the loss)."""

    def __init__(self, n_descriptors, hidden=(128, 128, 64, 32)):
        super().__init__()
        self.net = _mlp([n_descriptors + 1, *hidden, 1])

    def forward(self, x, T_norm):
        inp = torch.cat([x, T_norm.unsqueeze(-1)], dim=-1)
        return self.net(inp).squeeze(-1)
