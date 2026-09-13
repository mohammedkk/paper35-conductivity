"""Trains and evaluates PGML (with ablations), a black-box MLP, a
residual-only PINN baseline, and tree-ensemble baselines on the synthetic
conductivity benchmark, reproducing the comparison in Section 5 of the
paper. Run with: `python train.py [--seed 0] [--epochs 400]`.

All neural models are trained full-batch (the whole training set fits
comfortably in memory), which keeps curve-aligned tensors simple for the
physics-consistency loss and the extrapolation split.
"""

import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn

import dataset
from descriptors import DESCRIPTOR_NAMES
from models import PGML, BlackBoxMLP
from losses import shape_consistency_loss, intrinsic_arrhenius_slopes

LN10 = math.log(10.0)
K_B_EV = 8.617333262e-5


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------

def fit_normalizer(curves):
    X = np.stack([c.x for c in curves])
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    return mu, sigma


def curve_tensors(curves, mu, sigma, T_max=None):
    """Stack curves sharing the common temperature grid into
    (n_curves, n_temps) tensors, optionally restricted to T <= T_max."""
    T_full = curves[0].T
    mask = np.ones_like(T_full, dtype=bool) if T_max is None else (T_full <= T_max)
    T = T_full[mask]

    x_raw = np.stack([c.x for c in curves])
    x_norm = (x_raw - mu) / sigma
    log_sigma_obs = np.stack([np.log(c.sigma_obs[mask]) for c in curves])
    regime = np.stack([c.regime[mask] for c in curves])
    measurable = np.stack([c.measurable[mask] for c in curves])
    Eg_raw = np.array([c.Eg for c in curves])
    m_star = np.array([c.m_star for c in curves])

    T_norm = (T - dataset.T_MIN) / (dataset.T_MAX - dataset.T_MIN)
    return dict(
        x=torch.tensor(x_norm, dtype=torch.float32),
        T=torch.tensor(T, dtype=torch.float32),
        T_norm=torch.tensor(T_norm, dtype=torch.float32),
        log_sigma_obs=torch.tensor(log_sigma_obs, dtype=torch.float32),
        regime=torch.tensor(regime, dtype=torch.long),
        measurable=torch.tensor(measurable, dtype=torch.bool),
        Eg_raw=torch.tensor(Eg_raw, dtype=torch.float32),
        m_star=torch.tensor(m_star, dtype=torch.float32),
    )


def masked_mse(pred, target, measurable):
    """Mean squared error restricted to measurable points (see
    dataset.MEASURABLE_FLOOR): points the synthetic benchmark's own
    generator places below a realistic detection limit are excluded from
    both the loss and all reported metrics, rather than fit at an
    arbitrary floor value."""
    mask = measurable.float()
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def broadcast(data):
    """Flatten curve-batched (n_curves, n_temps) tensors to point level for
    the model's forward pass, returning the flat tensors plus the original
    shape for reshaping predictions back."""
    n_c, n_t = data["log_sigma_obs"].shape
    x_flat = data["x"].unsqueeze(1).expand(-1, n_t, -1).reshape(n_c * n_t, -1)
    T_flat = data["T"].unsqueeze(0).expand(n_c, -1).reshape(-1)
    T_norm_flat = data["T_norm"].unsqueeze(0).expand(n_c, -1).reshape(-1)
    m_star_flat = data["m_star"].unsqueeze(1).expand(-1, n_t).reshape(-1)
    return x_flat, T_flat, T_norm_flat, m_star_flat, (n_c, n_t)


def flat_arrays(curves, mu, sigma, T_min=None, T_max=None):
    flat = dataset.flatten(curves, T_max=T_max, T_min=T_min)
    x_norm = (flat["x"] - mu) / sigma
    T_norm = (flat["T"] - dataset.T_MIN) / (dataset.T_MAX - dataset.T_MIN)
    log_sigma_obs = np.log(flat["sigma_obs"])
    return dict(x=x_norm, T=flat["T"], T_norm=T_norm, log_sigma_obs=log_sigma_obs)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_neural(model_factory, train_data, val_data, use_shape=False, epochs=2000, patience=150, lr=2e-3,
                  lambda_shape=0.02, lambda_bound=0.05, weight_decay=1e-4, seed=0):
    """model_factory(n_descriptors) -> nn.Module, either a PGML variant or
    BlackBoxMLP. The residual-correction boundedness penalty is applied
    whenever the model produces a (non-None) delta -- i.e. whenever PGML is
    built with use_rcn=True -- and is a harmless no-op (delta == 0)
    otherwise, so no separate flag is needed for it."""
    torch.manual_seed(seed)
    n_descriptors = train_data["x"].shape[1]
    model = model_factory(n_descriptors)
    physics_guided = isinstance(model, PGML)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def forward_curvewise(data):
        x_flat, T_flat, T_norm_flat, m_star_flat, shape = broadcast(data)
        if physics_guided:
            log_sigma_pred_flat, _, _, delta_flat = model(x_flat, T_flat, T_norm_flat, m_star_flat)
        else:
            log_sigma_pred_flat = model(x_flat, T_norm_flat)
            delta_flat = None
        log_sigma_pred = log_sigma_pred_flat.reshape(shape)
        delta = delta_flat.reshape(shape) if delta_flat is not None else None
        return log_sigma_pred, delta

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        log_sigma_pred, delta = forward_curvewise(train_data)
        data_loss = masked_mse(log_sigma_pred / LN10, train_data["log_sigma_obs"] / LN10, train_data["measurable"])
        loss = data_loss
        if use_shape:
            loss = loss + lambda_shape * shape_consistency_loss(
                log_sigma_pred, train_data["T"], train_data["regime"], train_data["Eg_raw"],
                measurable=train_data["measurable"],
            )
        if delta is not None:
            loss = loss + lambda_bound * (delta ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = forward_curvewise(val_data)
            val_loss = masked_mse(val_pred / LN10, val_data["log_sigma_obs"] / LN10, val_data["measurable"]).item()

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_flat(model, flat, m_star_array=None):
    physics_guided = isinstance(model, PGML)
    x = torch.tensor(flat["x"], dtype=torch.float32)
    T = torch.tensor(flat["T"], dtype=torch.float32)
    T_norm = torch.tensor(flat["T_norm"], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        if physics_guided:
            m_star = torch.tensor(m_star_array, dtype=torch.float32)
            log_sigma_pred, _, _, _ = model(x, T, T_norm, m_star)
        else:
            log_sigma_pred = model(x, T_norm)
    return log_sigma_pred.numpy()


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def rmse_r2(pred_log10, true_log10):
    err = pred_log10 - true_log10
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((true_log10 - true_log10.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    return rmse, r2


def per_curve_rmse_records(pred_log10, true_log10, curve_ids):
    """Group flat (pred, true) log10-sigma points by curve_id and return the
    summed squared error and point count per (material, doping) test curve,
    tagged with its material name (curve_id is f"{material}_Nd{Nd:.0e}").
    Recording sum-of-squares and n rather than a per-curve RMSE lets
    bootstrap_tests.py recombine resampled curves into a pooled RMSE
    (sqrt(total sse / total n)) using the same point-level pooling as
    rmse_r2() above, rather than an average of per-curve RMSEs -- a
    different, non-comparable statistic when curves have very different
    error magnitudes, which they do here."""
    records = []
    for cid in np.unique(curve_ids):
        mask = curve_ids == cid
        err = pred_log10[mask] - true_log10[mask]
        material = str(cid).rsplit("_Nd", 1)[0]
        records.append({
            "curve_id": str(cid), "material": material,
            "sse": float(np.sum(err ** 2)), "n": int(mask.sum()),
        })
    return records


def consistency_violation_rate(log_sigma_pred_curves, T, regime, measurable=None):
    slopes, checkable = intrinsic_arrhenius_slopes(log_sigma_pred_curves, T, regime, measurable=measurable)
    if checkable.sum() == 0:
        return float("nan"), 0
    violations = (slopes[checkable] > 0.0).sum()
    return float(violations) / float(checkable.sum()), int(checkable.sum())


GBT_PARAM_GRID = [
    dict(n_estimators=n, max_depth=d, learning_rate=lr, min_samples_leaf=leaf)
    for n in (100, 300, 500)
    for d in (2, 3, 4)
    for lr in (0.05, 0.1, 0.2)
    for leaf in (1, 5)
]


def inverse_rmse_weights(preds_by_label, labels, y_true):
    """Inverse-variance weighting: weight_i proportional to 1/RMSE_i^2,
    normalized to sum to one, where each RMSE_i is computed once on
    held-out (validation) data. Unlike the stacked Ridge meta-learner,
    this fits exactly one scalar per candidate model -- its own
    validation RMSE -- rather than a full regression over the same small
    validation split, so it should be far less prone to overfitting that
    split's noise."""
    rmses = np.array([rmse_r2(preds_by_label[l], y_true)[0] for l in labels])
    inv_var = 1.0 / (rmses ** 2 + 1e-8)
    return inv_var / inv_var.sum()


def tune_gbt(X_tr, y_tr, X_val, y_val, seed):
    """Small grid search over GBT hyperparameters, selected by RMSE on a
    held-out material-level validation split (rather than the fixed
    n_estimators=300/max_depth=3 used in earlier experiments). Returns the
    best hyperparameter dict; the caller refits on train+val with it."""
    from sklearn.ensemble import GradientBoostingRegressor

    best_params, best_rmse = None, float("inf")
    for params in GBT_PARAM_GRID:
        est = GradientBoostingRegressor(random_state=seed, **params)
        est.fit(X_tr, y_tr)
        rmse, _ = rmse_r2(est.predict(X_val), y_val)
        if rmse < best_rmse:
            best_rmse, best_params = rmse, params
    return best_params, best_rmse


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def m_star_per_curve_material(curves):
    return {c.curve_id: c.m_star for c in curves}


def m_star_flat_from_ids(flat_curve_ids, lookup):
    return np.array([lookup[cid] for cid in flat_curve_ids])


def run(seed=0, epochs=400, patience=30, out_path="results.json"):
    curves = dataset.build_benchmark(seed=seed)
    train_curves, val_curves, test_curves = dataset.split_curves(curves, seed=seed)
    mu, sigma = fit_normalizer(train_curves)

    print(f"Curves: {len(train_curves)} train / {len(val_curves)} val / {len(test_curves)} test "
          f"({len(curves[0].T)} temperatures each)")

    # ---------------- Standard (in-distribution + physical consistency) ----------------
    train_data = curve_tensors(train_curves, mu, sigma)
    val_data = curve_tensors(val_curves, mu, sigma)
    test_data = curve_tensors(test_curves, mu, sigma)

    # label -> (trainer_fn(train_data, val_data) -> model)
    def pgml_factory(use_rcn, use_second_channel):
        return lambda n_desc: PGML(n_desc, use_rcn=use_rcn, use_second_channel=use_second_channel)

    def blackbox_factory(n_desc):
        return BlackBoxMLP(n_desc)

    def make_trainer(factory, use_shape):
        return lambda tr, va: train_neural(factory, tr, va, use_shape=use_shape,
                                            epochs=epochs, patience=patience, seed=seed)

    neural_specs = {
        "PGML (2-channel)": make_trainer(pgml_factory(use_rcn=False, use_second_channel=True), use_shape=True),
        "PGML (2-channel + RCN)": make_trainer(pgml_factory(use_rcn=True, use_second_channel=True), use_shape=True),
        "PGML (RCN only, no 2-channel)": make_trainer(pgml_factory(use_rcn=True, use_second_channel=False), use_shape=True),
        "PGML (no correction)": make_trainer(pgml_factory(use_rcn=False, use_second_channel=False), use_shape=False),
        "Black-box MLP": make_trainer(blackbox_factory, use_shape=False),
        "Residual-only PINN": make_trainer(blackbox_factory, use_shape=True),
    }

    results = {}
    trained_models = {}

    for label, trainer in neural_specs.items():
        trained_models[label] = trainer(train_data, val_data)

    test_flat = flat_arrays(test_curves, mu, sigma)
    m_star_lookup = m_star_per_curve_material(test_curves)
    test_flat_ids = dataset.flatten(test_curves)["curve_id"]
    m_star_flat = m_star_flat_from_ids(test_flat_ids, m_star_lookup)
    true_log10_test = test_flat["log_sigma_obs"] / LN10

    for label, model in trained_models.items():
        pred = predict_flat(model, test_flat, m_star_array=m_star_flat)
        rmse, r2 = rmse_r2(pred / LN10, true_log10_test)

        x_flat_c, T_flat_c, T_norm_flat_c, m_star_flat_c, shape_c = broadcast(test_data)
        physics_guided = isinstance(model, PGML)
        with torch.no_grad():
            if physics_guided:
                log_sigma_pred_flat, _, _, _ = model(x_flat_c, T_flat_c, T_norm_flat_c, m_star_flat_c)
            else:
                log_sigma_pred_flat = model(x_flat_c, T_norm_flat_c)
        log_sigma_pred_curves = log_sigma_pred_flat.reshape(shape_c).numpy()
        viol_rate, n_checkable = consistency_violation_rate(
            log_sigma_pred_curves, test_data["T"].numpy(), test_data["regime"].numpy(),
            measurable=test_data["measurable"].numpy(),
        )

        results[label] = dict(rmse_log10=rmse, r2=r2, consistency_violation=viol_rate,
                               n_checkable_curves=n_checkable)
        print(f"{label:32s} RMSE(log10 sigma)={rmse:.3f}  R2={r2:.3f}  "
              f"consistency_violation={viol_rate*100:.1f}% (n={n_checkable})")

    # ---------------- Extrapolation split ----------------
    print("\nExtrapolation split (train on T<=400K, test on unseen materials at T>400K):")
    train_data_lowT = curve_tensors(train_curves, mu, sigma, T_max=dataset.EXTRAPOLATION_SPLIT_T)
    val_data_lowT = curve_tensors(val_curves, mu, sigma, T_max=dataset.EXTRAPOLATION_SPLIT_T)
    test_flat_highT = flat_arrays(test_curves, mu, sigma, T_min=dataset.EXTRAPOLATION_SPLIT_T + 1e-6)
    m_star_lookup_test = m_star_per_curve_material(test_curves)
    test_flat_highT_ids = dataset.flatten(test_curves, T_min=dataset.EXTRAPOLATION_SPLIT_T + 1e-6)["curve_id"]
    m_star_highT = m_star_flat_from_ids(test_flat_highT_ids, m_star_lookup_test)
    true_log10_highT = test_flat_highT["log_sigma_obs"] / LN10

    trained_models_lowT = {}
    for label, trainer in neural_specs.items():
        model = trainer(train_data_lowT, val_data_lowT)
        trained_models_lowT[label] = model
        pred = predict_flat(model, test_flat_highT, m_star_array=m_star_highT)
        rmse, _ = rmse_r2(pred / LN10, true_log10_highT)
        results[label]["extrapolation_rmse"] = rmse
        print(f"{label:32s} extrapolation RMSE(log10 sigma)={rmse:.3f}")

    # ---------------- Tree-ensemble baselines (sklearn) ----------------
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        train_flat = flat_arrays(train_curves, mu, sigma)
        X_train = np.concatenate([train_flat["x"], train_flat["T_norm"][:, None]], axis=1)
        y_train = train_flat["log_sigma_obs"] / LN10
        X_test = np.concatenate([test_flat["x"], test_flat["T_norm"][:, None]], axis=1)

        train_flat_lowT = flat_arrays(train_curves, mu, sigma, T_max=dataset.EXTRAPOLATION_SPLIT_T)
        X_train_lowT = np.concatenate([train_flat_lowT["x"], train_flat_lowT["T_norm"][:, None]], axis=1)
        y_train_lowT = train_flat_lowT["log_sigma_obs"] / LN10
        X_test_highT = np.concatenate([test_flat_highT["x"], test_flat_highT["T_norm"][:, None]], axis=1)

        # Full (n_curves, n_temps) test grid, including points below the
        # measurability floor that flat_arrays()/dataset.flatten() exclude --
        # needed so the physical-consistency check below sees every point a
        # curve-wise model like PGML or the black-box MLP is evaluated on,
        # exactly as for those models (see the neural-model loop above).
        x_flat_c, _, T_norm_flat_c, _, shape_c = broadcast(test_data)
        X_test_curve_grid = np.concatenate([x_flat_c.numpy(), T_norm_flat_c.numpy()[:, None]], axis=1)

        # GBT hyperparameters are tuned on the material-level validation
        # split (val_curves), not the fixed n_estimators=300/max_depth=3
        # used in earlier experiments -- separately for the standard and
        # extrapolation regimes, since the best-fitting model on the full
        # temperature range need not be best for a model trained only on
        # T<=400K. Random Forest is left untuned for comparison.
        val_flat = flat_arrays(val_curves, mu, sigma)
        X_val = np.concatenate([val_flat["x"], val_flat["T_norm"][:, None]], axis=1)
        y_val = val_flat["log_sigma_obs"] / LN10
        val_flat_lowT = flat_arrays(val_curves, mu, sigma, T_max=dataset.EXTRAPOLATION_SPLIT_T)
        X_val_lowT = np.concatenate([val_flat_lowT["x"], val_flat_lowT["T_norm"][:, None]], axis=1)
        y_val_lowT = val_flat_lowT["log_sigma_obs"] / LN10

        # Validation-set m_star lookups, needed both by the neural models'
        # validation predictions below and by the two hybrids' validation
        # predictions used as additional ensemble members further down.
        m_star_lookup_val = m_star_per_curve_material(val_curves)
        m_star_val = m_star_flat_from_ids(dataset.flatten(val_curves)["curve_id"], m_star_lookup_val)
        m_star_val_lowT = m_star_flat_from_ids(
            dataset.flatten(val_curves, T_max=dataset.EXTRAPOLATION_SPLIT_T)["curve_id"], m_star_lookup_val)

        gbt_params, gbt_val_rmse = tune_gbt(X_train, y_train, X_val, y_val, seed)
        gbt_params_lowT, gbt_val_rmse_lowT = tune_gbt(X_train_lowT, y_train_lowT, X_val_lowT, y_val_lowT, seed)
        print(f"Tuned GBT params (standard): {gbt_params} (val RMSE {gbt_val_rmse:.3f})")
        print(f"Tuned GBT params (T<=400K):  {gbt_params_lowT} (val RMSE {gbt_val_rmse_lowT:.3f})")

        # Final fit uses train only (not train+val), identical to every
        # other model's training set in this paper, so that tuning is the
        # only thing distinguishing this Gradient-Boosted Trees run from
        # earlier ones, and so Random Forest's numbers are unaffected.
        # Fitted estimators are kept (tree_models / tree_models_lowT) for
        # reuse by the ensemble experiments below.
        tree_models, tree_models_lowT = {}, {}
        for label, est, est_extra in [
            ("Random Forest",
             RandomForestRegressor(n_estimators=300, max_depth=12, random_state=seed),
             RandomForestRegressor(n_estimators=300, max_depth=12, random_state=seed)),
            ("Gradient-Boosted Trees",
             GradientBoostingRegressor(random_state=seed, **gbt_params),
             GradientBoostingRegressor(random_state=seed, **gbt_params_lowT)),
        ]:
            est.fit(X_train, y_train)
            pred = est.predict(X_test)
            rmse, r2 = rmse_r2(pred, true_log10_test)

            est_extra.fit(X_train_lowT, y_train_lowT)
            pred_highT = est_extra.predict(X_test_highT)
            extra_rmse, _ = rmse_r2(pred_highT, true_log10_highT)

            tree_models[label] = est
            tree_models_lowT[label] = est_extra

            # Physical-consistency violation is evaluated the same way as for
            # the neural models: predict log_e(sigma) on every point of the
            # shared (n_curves, n_temps) test grid -- not just the
            # measurability-filtered flat points used for RMSE/R2 above --
            # and reuse the same finite-difference Arrhenius-slope check.
            # Tree ensembles are perfectly capable of being queried at any
            # (x, T) row; only the *training* objective differs from the
            # neural models, not what can be evaluated at test time.
            log_sigma_pred_curves_tree = (est.predict(X_test_curve_grid) * LN10).reshape(shape_c)
            viol_rate, n_checkable = consistency_violation_rate(
                log_sigma_pred_curves_tree, test_data["T"].numpy(), test_data["regime"].numpy(),
                measurable=test_data["measurable"].numpy(),
            )

            results[label] = dict(rmse_log10=rmse, r2=r2, consistency_violation=viol_rate,
                                   n_checkable_curves=n_checkable, extrapolation_rmse=extra_rmse)
            print(f"{label:32s} RMSE(log10 sigma)={rmse:.3f}  R2={r2:.3f}  "
                  f"extrapolation RMSE={extra_rmse:.3f}")

        # ---------------- Hybrid: physics prior + GBT residual ----------------
        # Fits gradient-boosted trees not on log10(sigma) directly, but on the
        # residual between the observed log10(sigma) and the PGML (2-channel)
        # closed-form prediction, then adds the tree correction back on top.
        # This combines the closed form's physical prior (and, in principle,
        # its extrapolation behavior) with the tree ensemble's evident
        # advantage at fitting whatever nonlinear signal is left in a small,
        # tabular, non-smooth regression problem like this one.
        hybrid_label = "PGML (2-channel) + GBT residual"
        physics_model = trained_models["PGML (2-channel)"]
        physics_pred_train = predict_flat(physics_model, train_flat, m_star_array=m_star_flat_from_ids(
            dataset.flatten(train_curves)["curve_id"], m_star_per_curve_material(train_curves)))
        residual_train = (train_flat["log_sigma_obs"] - physics_pred_train) / LN10

        gbt_residual = GradientBoostingRegressor(random_state=seed, **gbt_params)
        gbt_residual.fit(X_train, residual_train)

        physics_pred_test = predict_flat(physics_model, test_flat, m_star_array=m_star_flat)
        hybrid_pred_test = physics_pred_test / LN10 + gbt_residual.predict(X_test)
        hybrid_rmse, hybrid_r2 = rmse_r2(hybrid_pred_test, true_log10_test)

        physics_model_lowT = trained_models_lowT["PGML (2-channel)"]
        m_star_train_lowT = m_star_flat_from_ids(
            dataset.flatten(train_curves, T_max=dataset.EXTRAPOLATION_SPLIT_T)["curve_id"],
            m_star_per_curve_material(train_curves),
        )
        physics_pred_train_lowT = predict_flat(physics_model_lowT, train_flat_lowT, m_star_array=m_star_train_lowT)
        residual_train_lowT = (train_flat_lowT["log_sigma_obs"] - physics_pred_train_lowT) / LN10
        gbt_residual_extra = GradientBoostingRegressor(random_state=seed, **gbt_params_lowT)
        gbt_residual_extra.fit(X_train_lowT, residual_train_lowT)

        physics_pred_test_highT = predict_flat(physics_model_lowT, test_flat_highT, m_star_array=m_star_highT)
        hybrid_pred_highT = physics_pred_test_highT / LN10 + gbt_residual_extra.predict(X_test_highT)
        hybrid_extra_rmse, _ = rmse_r2(hybrid_pred_highT, true_log10_highT)

        # Validation-set predictions for this hybrid, needed only as an
        # additional ensemble member in the extended ensemble below (not
        # used to fit anything about the hybrid itself).
        physics_pred_val = predict_flat(physics_model, val_flat, m_star_array=m_star_val)
        hybrid_pred_val = physics_pred_val / LN10 + gbt_residual.predict(X_val)
        physics_pred_val_lowT = predict_flat(physics_model_lowT, val_flat_lowT, m_star_array=m_star_val_lowT)
        hybrid_pred_val_lowT = physics_pred_val_lowT / LN10 + gbt_residual_extra.predict(X_val_lowT)

        results[hybrid_label] = dict(rmse_log10=hybrid_rmse, r2=hybrid_r2, consistency_violation=float("nan"),
                                      n_checkable_curves=0, extrapolation_rmse=hybrid_extra_rmse)
        print(f"{hybrid_label:32s} RMSE(log10 sigma)={hybrid_rmse:.3f}  R2={hybrid_r2:.3f}  "
              f"extrapolation RMSE={hybrid_extra_rmse:.3f}")

        # ---------------- Hybrid 2: GBT with physics-parameter features ----------------
        # A different way of combining physics and boosting: instead of
        # correcting the closed form's residual, hand the tree ensemble the
        # closed form's own fitted parameters (E_g,eff, C_n, N_0, tau_ac0,
        # tau_ii0, and the second-channel parameters) as extra engineered
        # features alongside the raw descriptors and T, and let it predict
        # log10(sigma) directly. This tests whether GBT benefits from the
        # physics-parameter head's nonlinear re-expression of the descriptors
        # even when it is not constrained to route through the closed form.
        feat_label = "GBT + physics-parameter features"

        def theta_features(model, x_normalized):
            with torch.no_grad():
                theta = model.pph(torch.tensor(x_normalized, dtype=torch.float32))
            return theta.numpy()

        X_train_feat = np.concatenate([X_train, theta_features(physics_model, train_flat["x"])], axis=1)
        X_test_feat = np.concatenate([X_test, theta_features(physics_model, test_flat["x"])], axis=1)
        gbt_feat = GradientBoostingRegressor(random_state=seed, **gbt_params)
        gbt_feat.fit(X_train_feat, y_train)
        feat_pred_test = gbt_feat.predict(X_test_feat)
        feat_rmse, feat_r2 = rmse_r2(feat_pred_test, true_log10_test)

        X_train_lowT_feat = np.concatenate(
            [X_train_lowT, theta_features(physics_model_lowT, train_flat_lowT["x"])], axis=1)
        X_test_highT_feat = np.concatenate(
            [X_test_highT, theta_features(physics_model_lowT, test_flat_highT["x"])], axis=1)
        gbt_feat_extra = GradientBoostingRegressor(random_state=seed, **gbt_params_lowT)
        gbt_feat_extra.fit(X_train_lowT_feat, y_train_lowT)
        feat_pred_highT = gbt_feat_extra.predict(X_test_highT_feat)
        feat_extra_rmse, _ = rmse_r2(feat_pred_highT, true_log10_highT)

        # Validation-set predictions for this hybrid, needed only as an
        # additional ensemble member in the extended ensemble below.
        X_val_feat = np.concatenate([X_val, theta_features(physics_model, val_flat["x"])], axis=1)
        feat_pred_val = gbt_feat.predict(X_val_feat)
        X_val_lowT_feat = np.concatenate(
            [X_val_lowT, theta_features(physics_model_lowT, val_flat_lowT["x"])], axis=1)
        feat_pred_val_lowT = gbt_feat_extra.predict(X_val_lowT_feat)

        results[feat_label] = dict(rmse_log10=feat_rmse, r2=feat_r2, consistency_violation=float("nan"),
                                    n_checkable_curves=0, extrapolation_rmse=feat_extra_rmse)
        print(f"{feat_label:32s} RMSE(log10 sigma)={feat_rmse:.3f}  R2={feat_r2:.3f}  "
              f"extrapolation RMSE={feat_extra_rmse:.3f}")

        # ---------------- Ensemble: combining every model tried so far ----------------
        # Two ways of integrating every base model in this experiment (the
        # four PGML variants, the black-box MLP, the residual-only PINN,
        # and the two tree ensembles) into one final predictor, rather than
        # relying on any single architecture or closed form to close the
        # remaining gap on its own.
        def collect_predictions(neural_models, tree_models_dict, flat, m_star_arr, X_flat):
            preds = {}
            for label, model in neural_models.items():
                preds[label] = predict_flat(model, flat, m_star_array=m_star_arr) / LN10
            for label, model in tree_models_dict.items():
                preds[label] = model.predict(X_flat)
            return preds

        val_preds = collect_predictions(trained_models, tree_models, val_flat, m_star_val, X_val)
        test_preds = collect_predictions(trained_models, tree_models, test_flat, m_star_flat, X_test)
        val_preds_lowT = collect_predictions(trained_models_lowT, tree_models_lowT, val_flat_lowT,
                                              m_star_val_lowT, X_val_lowT)
        test_preds_highT = collect_predictions(trained_models_lowT, tree_models_lowT, test_flat_highT,
                                                m_star_highT, X_test_highT)
        base_labels = list(val_preds.keys())

        # 1) Simple average: unweighted mean of all eight base predictions.
        avg_label = "Ensemble (simple average)"
        avg_pred_test = np.mean([test_preds[l] for l in base_labels], axis=0)
        avg_rmse, avg_r2 = rmse_r2(avg_pred_test, true_log10_test)
        avg_pred_highT = np.mean([test_preds_highT[l] for l in base_labels], axis=0)
        avg_extra_rmse, _ = rmse_r2(avg_pred_highT, true_log10_highT)
        results[avg_label] = dict(rmse_log10=avg_rmse, r2=avg_r2, consistency_violation=float("nan"),
                                   n_checkable_curves=0, extrapolation_rmse=avg_extra_rmse)
        print(f"{avg_label:32s} RMSE(log10 sigma)={avg_rmse:.3f}  R2={avg_r2:.3f}  "
              f"extrapolation RMSE={avg_extra_rmse:.3f}")

        # 2) Stacked ensemble: a non-negative-weight Ridge meta-learner fits
        # the combination weights on the validation split's base-model
        # predictions (never seen by the base models' own training), then
        # applies those weights to the test-set base predictions. Fit
        # separately for the standard and extrapolation regimes, since the
        # best combination of models need not be the same in both.
        from sklearn.linear_model import Ridge

        stack_label = "Ensemble (stacked, Ridge)"
        P_val = np.stack([val_preds[l] for l in base_labels], axis=1)
        P_test = np.stack([test_preds[l] for l in base_labels], axis=1)
        stacker = Ridge(alpha=1.0, positive=True)
        stacker.fit(P_val, y_val)
        stack_pred_test = stacker.predict(P_test)
        stack_rmse, stack_r2 = rmse_r2(stack_pred_test, true_log10_test)

        P_val_lowT = np.stack([val_preds_lowT[l] for l in base_labels], axis=1)
        P_test_highT = np.stack([test_preds_highT[l] for l in base_labels], axis=1)
        stacker_extra = Ridge(alpha=1.0, positive=True)
        stacker_extra.fit(P_val_lowT, y_val_lowT)
        stack_pred_highT = stacker_extra.predict(P_test_highT)
        stack_extra_rmse, _ = rmse_r2(stack_pred_highT, true_log10_highT)

        results[stack_label] = dict(rmse_log10=stack_rmse, r2=stack_r2, consistency_violation=float("nan"),
                                     n_checkable_curves=0, extrapolation_rmse=stack_extra_rmse,
                                     stacked_weights=dict(zip(base_labels, stacker.coef_.tolist())),
                                     stacked_weights_extrapolation=dict(zip(base_labels, stacker_extra.coef_.tolist())))
        print(f"{stack_label:32s} RMSE(log10 sigma)={stack_rmse:.3f}  R2={stack_r2:.3f}  "
              f"extrapolation RMSE={stack_extra_rmse:.3f}")
        print("  Stacked weights (standard):     "
              + ", ".join(f"{l}={w:.2f}" for l, w in zip(base_labels, stacker.coef_)))
        print("  Stacked weights (extrapolation): "
              + ", ".join(f"{l}={w:.2f}" for l, w in zip(base_labels, stacker_extra.coef_)))

        # 3) Validation-RMSE-weighted average: a middle ground between the
        # simple average (no data-driven weighting) and the stacked Ridge
        # (many free weights fit on the same 12 validation curves). Each
        # model gets exactly one number estimated from validation data --
        # its own RMSE there -- combined by inverse-variance weighting,
        # which uses far less of the validation split's limited
        # information than a full regression fit.
        wavg_label = "Ensemble (weighted by validation RMSE)"
        weights = inverse_rmse_weights(val_preds, base_labels, y_val)
        wavg_pred_test = np.sum([w * test_preds[l] for w, l in zip(weights, base_labels)], axis=0)
        wavg_rmse, wavg_r2 = rmse_r2(wavg_pred_test, true_log10_test)

        weights_lowT = inverse_rmse_weights(val_preds_lowT, base_labels, y_val_lowT)
        wavg_pred_highT = np.sum([w * test_preds_highT[l] for w, l in zip(weights_lowT, base_labels)], axis=0)
        wavg_extra_rmse, _ = rmse_r2(wavg_pred_highT, true_log10_highT)

        results[wavg_label] = dict(
            rmse_log10=wavg_rmse, r2=wavg_r2, consistency_violation=float("nan"),
            n_checkable_curves=0, extrapolation_rmse=wavg_extra_rmse,
            validation_weights=dict(zip(base_labels, weights.tolist())),
            validation_weights_extrapolation=dict(zip(base_labels, weights_lowT.tolist())),
        )
        print(f"{wavg_label:32s} RMSE(log10 sigma)={wavg_rmse:.3f}  R2={wavg_r2:.3f}  "
              f"extrapolation RMSE={wavg_extra_rmse:.3f}")
        print("  Validation-RMSE weights (standard):     "
              + ", ".join(f"{l}={w:.2f}" for l, w in zip(base_labels, weights)))
        print("  Validation-RMSE weights (extrapolation): "
              + ", ".join(f"{l}={w:.2f}" for l, w in zip(base_labels, weights_lowT)))

        # ---------------- Extended ensemble: hybrids as members too ----------------
        # A sixth experiment: rather than treating the two physics-plus-
        # boosting hybrids only as standalone candidates (Section 5.4) or
        # leaving them out of the ensemble entirely (Section 5.5), add both
        # as two more candidate members alongside the original eight, for
        # ten base predictors combined by the same two strategies.
        ext_val_preds = dict(val_preds, **{hybrid_label: hybrid_pred_val, feat_label: feat_pred_val})
        ext_test_preds = dict(test_preds, **{hybrid_label: hybrid_pred_test, feat_label: feat_pred_test})
        ext_val_preds_lowT = dict(val_preds_lowT,
                                   **{hybrid_label: hybrid_pred_val_lowT, feat_label: feat_pred_val_lowT})
        ext_test_preds_highT = dict(test_preds_highT,
                                     **{hybrid_label: hybrid_pred_highT, feat_label: feat_pred_highT})
        ext_labels = base_labels + [hybrid_label, feat_label]

        avg_ext_label = "Ensemble (10 models, simple average)"
        avg_ext_pred_test = np.mean([ext_test_preds[l] for l in ext_labels], axis=0)
        avg_ext_rmse, avg_ext_r2 = rmse_r2(avg_ext_pred_test, true_log10_test)
        avg_ext_pred_highT = np.mean([ext_test_preds_highT[l] for l in ext_labels], axis=0)
        avg_ext_extra_rmse, _ = rmse_r2(avg_ext_pred_highT, true_log10_highT)
        results[avg_ext_label] = dict(rmse_log10=avg_ext_rmse, r2=avg_ext_r2, consistency_violation=float("nan"),
                                       n_checkable_curves=0, extrapolation_rmse=avg_ext_extra_rmse)
        print(f"{avg_ext_label:32s} RMSE(log10 sigma)={avg_ext_rmse:.3f}  R2={avg_ext_r2:.3f}  "
              f"extrapolation RMSE={avg_ext_extra_rmse:.3f}")

        stack_ext_label = "Ensemble (10 models, stacked Ridge)"
        P_val_ext = np.stack([ext_val_preds[l] for l in ext_labels], axis=1)
        P_test_ext = np.stack([ext_test_preds[l] for l in ext_labels], axis=1)
        stacker_ext = Ridge(alpha=1.0, positive=True)
        stacker_ext.fit(P_val_ext, y_val)
        stack_ext_pred_test = stacker_ext.predict(P_test_ext)
        stack_ext_rmse, stack_ext_r2 = rmse_r2(stack_ext_pred_test, true_log10_test)

        P_val_ext_lowT = np.stack([ext_val_preds_lowT[l] for l in ext_labels], axis=1)
        P_test_ext_highT = np.stack([ext_test_preds_highT[l] for l in ext_labels], axis=1)
        stacker_ext_extra = Ridge(alpha=1.0, positive=True)
        stacker_ext_extra.fit(P_val_ext_lowT, y_val_lowT)
        stack_ext_pred_highT = stacker_ext_extra.predict(P_test_ext_highT)
        stack_ext_extra_rmse, _ = rmse_r2(stack_ext_pred_highT, true_log10_highT)

        results[stack_ext_label] = dict(
            rmse_log10=stack_ext_rmse, r2=stack_ext_r2, consistency_violation=float("nan"),
            n_checkable_curves=0, extrapolation_rmse=stack_ext_extra_rmse,
            stacked_weights=dict(zip(ext_labels, stacker_ext.coef_.tolist())),
            stacked_weights_extrapolation=dict(zip(ext_labels, stacker_ext_extra.coef_.tolist())),
        )
        print(f"{stack_ext_label:32s} RMSE(log10 sigma)={stack_ext_rmse:.3f}  R2={stack_ext_r2:.3f}  "
              f"extrapolation RMSE={stack_ext_extra_rmse:.3f}")
        print("  Stacked weights (standard):     "
              + ", ".join(f"{l}={w:.2f}" for l, w in zip(ext_labels, stacker_ext.coef_)))
        print("  Stacked weights (extrapolation): "
              + ", ".join(f"{l}={w:.2f}" for l, w in zip(ext_labels, stacker_ext_extra.coef_)))

        # ---------------- Validation-RMSE-weighted average, 10 models ----------------
        # A seventh experiment: the same inverse-variance weighting scheme
        # as above, now over the ten-model pool, to see whether a
        # low-parameter combination rule is any more sensitive to the
        # candidate pool than the simple average (Section 5.6) or the
        # stacked Ridge were.
        wavg_ext_label = "Ensemble (10 models, weighted by validation RMSE)"
        weights_ext = inverse_rmse_weights(ext_val_preds, ext_labels, y_val)
        wavg_ext_pred_test = np.sum([w * ext_test_preds[l] for w, l in zip(weights_ext, ext_labels)], axis=0)
        wavg_ext_rmse, wavg_ext_r2 = rmse_r2(wavg_ext_pred_test, true_log10_test)

        weights_ext_lowT = inverse_rmse_weights(ext_val_preds_lowT, ext_labels, y_val_lowT)
        wavg_ext_pred_highT = np.sum(
            [w * ext_test_preds_highT[l] for w, l in zip(weights_ext_lowT, ext_labels)], axis=0)
        wavg_ext_extra_rmse, _ = rmse_r2(wavg_ext_pred_highT, true_log10_highT)

        results[wavg_ext_label] = dict(
            rmse_log10=wavg_ext_rmse, r2=wavg_ext_r2, consistency_violation=float("nan"),
            n_checkable_curves=0, extrapolation_rmse=wavg_ext_extra_rmse,
            validation_weights=dict(zip(ext_labels, weights_ext.tolist())),
            validation_weights_extrapolation=dict(zip(ext_labels, weights_ext_lowT.tolist())),
        )
        print(f"{wavg_ext_label:32s} RMSE(log10 sigma)={wavg_ext_rmse:.3f}  R2={wavg_ext_r2:.3f}  "
              f"extrapolation RMSE={wavg_ext_extra_rmse:.3f}")
        print("  Validation-RMSE weights (standard):     "
              + ", ".join(f"{l}={w:.2f}" for l, w in zip(ext_labels, weights_ext)))
        print("  Validation-RMSE weights (extrapolation): "
              + ", ".join(f"{l}={w:.2f}" for l, w in zip(ext_labels, weights_ext_lowT)))

        # ---------------- Per-curve RMSE, for the material-level bootstrap ----------------
        # Only the standard (in-distribution) split and only the models
        # bootstrap_tests.py actually compares -- the same set already
        # covered by significance_tests.py's paired seed-level tests -- to
        # keep the results file from growing unboundedly.
        per_curve_preds = dict(test_preds)
        per_curve_preds[avg_label] = avg_pred_test
        per_curve_preds[stack_label] = stack_pred_test
        per_curve_preds[wavg_label] = wavg_pred_test
        per_curve_preds[hybrid_label] = hybrid_pred_test
        results["_per_curve_rmse"] = {
            label: per_curve_rmse_records(pred, true_log10_test, test_flat_ids)
            for label, pred in per_curve_preds.items()
        }
    except ImportError:
        print("scikit-learn not installed; skipping tree-ensemble baselines.")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--out", type=str, default="results.json")
    args = parser.parse_args()
    run(seed=args.seed, epochs=args.epochs, patience=args.patience, out_path=args.out)
