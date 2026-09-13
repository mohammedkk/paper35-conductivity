"""
Physics-informed vs. black-box deep learning for electrical (ionic) conductivity
prediction in solid-state materials.

Generates a physically-grounded synthetic benchmark (structure descriptors ->
Arrhenius parameters -> temperature-dependent conductivity), trains three
models from scratch with a hand-written NumPy MLP + Adam optimizer:

  A. Black-box MLP        : (T, x) -> log10(sigma), pure data loss
  B. Soft physics-informed: same architecture as A, + a physics-residual loss
                             evaluated at temperature-collocation pairs, penalizing
                             violation of the Arrhenius differential relation
  C. Physics-embedded     : x -> (Ea, log A); log10(sigma) computed analytically
                             from the Arrhenius law (physics baked into the
                             architecture, not just the loss)

and evaluates all three on three held-out conditions: in-distribution,
temperature extrapolation, and material-family (compositional) extrapolation.

No experimental data is used; everything below is derived from a stated,
reproducible generative model (see `references.bib` for the physical
parameter ranges' inspiration, e.g. Famprikis et al. 2019). Fixed seed = 42.
"""
import json
import numpy as np

RNG_SEED = 42
KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K
LN10 = np.log(10.0)

rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Synthetic, physically-grounded structure -> Arrhenius-parameter model
# ---------------------------------------------------------------------------

FAMILIES = {
    "sulfide_like":    dict(ea_base=0.25, loga_base=4.5),
    "garnet_like":     dict(ea_base=0.35, loga_base=4.0),
    "nasicon_like":    dict(ea_base=0.30, loga_base=4.2),
    "perovskite_like": dict(ea_base=0.50, loga_base=3.7),  # held out entirely for family extrapolation
}
TRAIN_FAMILIES = ["sulfide_like", "garnet_like", "nasicon_like"]
HOLDOUT_FAMILY = "perovskite_like"


def sample_descriptors(n, rng_):
    """x1..x5 in [0,1]: bottleneck radius ratio, free volume fraction,
    carrier concentration, lattice rigidity/covalency, ionic radius mismatch."""
    return rng_.uniform(0.0, 1.0, size=(n, 5))


def structure_to_arrhenius(x, family, rng_, noise=True):
    ea_base = FAMILIES[family]["ea_base"]
    loga_base = FAMILIES[family]["loga_base"]
    x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

    ea = (ea_base
          + 0.15 * (1 - x1)
          + 0.10 * (1 - x2)
          + 0.12 * x4
          + 0.10 * x5
          - 0.05 * x3
          + 0.10 * x1 * x4)
    loga = (loga_base
            + 0.8 * x2
            + 0.6 * x3
            - 0.4 * x5
            - 0.3 * x4
            - 0.25 * x2 * x5)

    if noise:
        ea = ea + rng_.normal(0.0, 0.015, size=ea.shape)
        loga = loga + rng_.normal(0.0, 0.08, size=loga.shape)

    ea = np.clip(ea, 0.05, 1.2)
    loga = np.clip(loga, 1.0, 7.0)
    return ea, loga


def arrhenius_log10sigma(ea, loga, T):
    return loga - ea / (KB_EV * LN10 * T)


def make_material_pool(n_materials, families, rng_):
    fam_choice = rng_.choice(families, size=n_materials)
    x = sample_descriptors(n_materials, rng_)
    ea = np.zeros(n_materials)
    loga = np.zeros(n_materials)
    for fam in set(families):
        mask = fam_choice == fam
        if mask.sum() == 0:
            continue
        ea[mask], loga[mask] = structure_to_arrhenius(x[mask], fam, rng_)
    return x, ea, loga, fam_choice


def expand_over_temperatures(x, ea, loga, T_values, rng_, sigma_noise=0.03):
    """Expand a material pool into (T, descriptors) rows with noisy log10(sigma)."""
    n_mat = x.shape[0]
    rows_x, rows_T, rows_y = [], [], []
    for t in T_values:
        y_true = arrhenius_log10sigma(ea, loga, t)
        y_noisy = y_true + rng_.normal(0.0, sigma_noise, size=y_true.shape)
        rows_x.append(x)
        rows_T.append(np.full(n_mat, t))
        rows_y.append(y_noisy)
    return (np.concatenate(rows_x), np.concatenate(rows_T), np.concatenate(rows_y))


# ---------------------------------------------------------------------------
# 2. Dataset construction
# ---------------------------------------------------------------------------

N_TRAIN_MATERIALS = 480   # optimization split
N_VAL_MATERIALS = 120     # early-stopping validation split (disjoint materials, same families)
N_ID_TEST_MATERIALS = 150
N_EXTRAP_MATERIALS = 150  # reused for temperature-extrapolation test (same families)
N_FAMILY_MATERIALS = 150  # held-out family, structure extrapolation

T_TRAIN = np.array([250.0, 280.0, 310.0, 340.0, 370.0, 400.0])
T_EXTRAP = np.array([420.0, 450.0, 480.0, 500.0])

x_train_mat, ea_train_mat, loga_train_mat, fam_train = make_material_pool(
    N_TRAIN_MATERIALS, TRAIN_FAMILIES, rng)
x_val_mat, ea_val_mat, loga_val_mat, fam_val = make_material_pool(
    N_VAL_MATERIALS, TRAIN_FAMILIES, rng)
x_idtest_mat, ea_idtest_mat, loga_idtest_mat, fam_idtest = make_material_pool(
    N_ID_TEST_MATERIALS, TRAIN_FAMILIES, rng)
x_text_mat, ea_text_mat, loga_text_mat, fam_text = make_material_pool(
    N_EXTRAP_MATERIALS, TRAIN_FAMILIES, rng)
x_fam_mat, ea_fam_mat, loga_fam_mat, fam_fam = make_material_pool(
    N_FAMILY_MATERIALS, [HOLDOUT_FAMILY], rng)

X_train, T_train, y_train = expand_over_temperatures(x_train_mat, ea_train_mat, loga_train_mat, T_TRAIN, rng)
X_val, T_val, y_val = expand_over_temperatures(x_val_mat, ea_val_mat, loga_val_mat, T_TRAIN, rng)
X_id, T_id, y_id = expand_over_temperatures(x_idtest_mat, ea_idtest_mat, loga_idtest_mat, T_TRAIN, rng)
X_text, T_text, y_text = expand_over_temperatures(x_text_mat, ea_text_mat, loga_text_mat, T_EXTRAP, rng)
X_fam, T_fam, y_fam = expand_over_temperatures(x_fam_mat, ea_fam_mat, loga_fam_mat, T_TRAIN, rng)

# feature normalization (fit on optimization split only)
T_MEAN, T_STD = T_train.mean(), T_train.std()


def featurize(X, T):
    Tn = (T - T_MEAN) / T_STD
    return np.concatenate([Tn.reshape(-1, 1), X], axis=1)


F_train = featurize(X_train, T_train)
F_val = featurize(X_val, T_val)
F_id = featurize(X_id, T_id)
F_text = featurize(X_text, T_text)
F_fam = featurize(X_fam, T_fam)

# ---------------------------------------------------------------------------
# 3. Minimal NumPy MLP with manual backprop + Adam
# ---------------------------------------------------------------------------


class MLP:
    def __init__(self, sizes, seed):
        r = np.random.default_rng(seed)
        self.sizes = sizes
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            self.W.append(r.normal(0, scale, size=(sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self.m = {"W": [np.zeros_like(w) for w in self.W], "b": [np.zeros_like(bb) for bb in self.b]}
        self.v = {"W": [np.zeros_like(w) for w in self.W], "b": [np.zeros_like(bb) for bb in self.b]}
        self.t = 0

    def forward(self, X):
        A = [X]
        Z = []
        n_layers = len(self.W)
        for i in range(n_layers):
            z = A[-1] @ self.W[i] + self.b[i]
            Z.append(z)
            if i < n_layers - 1:
                A.append(np.tanh(z))
            else:
                A.append(z)  # linear output layer
        return A[-1], (A, Z)

    def backward(self, cache, dOut):
        A, Z = cache
        n_layers = len(self.W)
        dW = [None] * n_layers
        db = [None] * n_layers
        dA = dOut
        for i in reversed(range(n_layers)):
            if i < n_layers - 1:
                dz = dA * (1 - np.tanh(Z[i]) ** 2)
            else:
                dz = dA
            dW[i] = A[i].T @ dz
            db[i] = dz.sum(axis=0)
            dA = dz @ self.W[i].T
        return dW, db

    def adam_step(self, dW, db, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0):
        self.t += 1
        for i in range(len(self.W)):
            self.m["W"][i] = beta1 * self.m["W"][i] + (1 - beta1) * dW[i]
            self.v["W"][i] = beta2 * self.v["W"][i] + (1 - beta2) * dW[i] ** 2
            mhat = self.m["W"][i] / (1 - beta1 ** self.t)
            vhat = self.v["W"][i] / (1 - beta2 ** self.t)
            self.W[i] -= lr * (mhat / (np.sqrt(vhat) + eps) + wd * self.W[i])  # decoupled weight decay (AdamW)

            self.m["b"][i] = beta1 * self.m["b"][i] + (1 - beta1) * db[i]
            self.v["b"][i] = beta2 * self.v["b"][i] + (1 - beta2) * db[i] ** 2
            mhat_b = self.m["b"][i] / (1 - beta1 ** self.t)
            vhat_b = self.v["b"][i] / (1 - beta2 ** self.t)
            self.b[i] -= lr * mhat_b / (np.sqrt(vhat_b) + eps)

    def snapshot(self):
        return ([w.copy() for w in self.W], [bb.copy() for bb in self.b])

    def load(self, snap):
        W, b = snap
        self.W = [w.copy() for w in W]
        self.b = [bb.copy() for bb in b]


HIDDEN = [32, 32]
N_IN = F_train.shape[1]  # T + 5 descriptors = 6
EPOCHS = 2000
BATCH = 64
LR_MAX = 3e-3
LR_MIN = 5e-5
WD = 2e-4  # AdamW weight decay
VAL_EVERY = 20
DT_COLLOC = 6.0  # K, finite-difference step for physics residual
EA_MIN_PHYS, EA_MAX_PHYS = 0.05, 1.2  # eV, physically admissible activation-energy window


def cosine_lr(epoch, epochs, lr_max, lr_min):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / epochs))


def iterate_minibatches(n, batch, rng_):
    idx = rng_.permutation(n)
    for start in range(0, n, batch):
        yield idx[start:start + batch]


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def metrics(y_true, y_pred):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse_log10sigma": rmse, "mae_log10sigma": mae, "r2": r2}


# ---------------------------------------------------------------------------
# 4. Model A: black-box MLP, pure data loss
# ---------------------------------------------------------------------------

netA = MLP([N_IN] + HIDDEN + [1], seed=1)
trainrng = np.random.default_rng(RNG_SEED + 1)
lossA_hist, valA_hist = [], []
bestA_val, bestA_snap = np.inf, netA.snapshot()
for epoch in range(EPOCHS):
    lr_t = cosine_lr(epoch, EPOCHS, LR_MAX, LR_MIN)
    for batch_idx in iterate_minibatches(F_train.shape[0], BATCH, trainrng):
        Xb, yb = F_train[batch_idx], y_train[batch_idx].reshape(-1, 1)
        pred, cache = netA.forward(Xb)
        dOut = 2 * (pred - yb) / Xb.shape[0]
        dW, db = netA.backward(cache, dOut)
        netA.adam_step(dW, db, lr=lr_t, wd=WD)
    if epoch % VAL_EVERY == 0 or epoch == EPOCHS - 1:
        pred_all, _ = netA.forward(F_train)
        lossA_hist.append(float(np.mean((pred_all.ravel() - y_train) ** 2)))
        pred_val, _ = netA.forward(F_val)
        val_mse = float(np.mean((pred_val.ravel() - y_val) ** 2))
        valA_hist.append(val_mse)
        if val_mse < bestA_val:
            bestA_val, bestA_snap = val_mse, netA.snapshot()
netA.load(bestA_snap)


def predict_A(F):
    pred, _ = netA.forward(F)
    return pred.ravel()


# ---------------------------------------------------------------------------
# 5. Model B: soft physics-informed MLP (same architecture, + physics residual)
# ---------------------------------------------------------------------------

netB = MLP([N_IN] + HIDDEN + [1], seed=1)  # same init seed as A for a fair comparison
trainrng = np.random.default_rng(RNG_SEED + 2)
LAMBDA_PHYS = 0.5
lossB_hist, valB_hist = [], []
bestB_val, bestB_snap = np.inf, netB.snapshot()

# collocation pool: reuse training material descriptors, sample random T within [T_min-40, T_max+40]
colloc_x = X_train  # (n_train_rows, 5) -- includes repeats across T, fine as a descriptor pool
colloc_T_low, colloc_T_high = 230.0, 420.0

for epoch in range(EPOCHS):
    lr_t = cosine_lr(epoch, EPOCHS, LR_MAX, LR_MIN)
    for batch_idx in iterate_minibatches(F_train.shape[0], BATCH, trainrng):
        Xb, yb = F_train[batch_idx], y_train[batch_idx].reshape(-1, 1)
        pred, cache = netB.forward(Xb)
        dOut_data = 2 * (pred - yb) / Xb.shape[0]

        # --- physics-residual collocation batch ---
        colloc_idx = trainrng.choice(colloc_x.shape[0], size=Xb.shape[0], replace=True)
        xc = colloc_x[colloc_idx]
        Tc = trainrng.uniform(colloc_T_low, colloc_T_high, size=xc.shape[0])
        Fc1 = featurize(xc, Tc)
        Fc2 = featurize(xc, Tc + DT_COLLOC)

        pred1, cache1 = netB.forward(Fc1)
        pred2, cache2 = netB.forward(Fc2)
        inv_T1 = 1.0 / Tc
        inv_T2 = 1.0 / (Tc + DT_COLLOC)
        d_inv_T = (inv_T2 - inv_T1).reshape(-1, 1)  # negative
        slope = (pred2 - pred1) / d_inv_T  # d(log10 sigma)/d(1/T)
        ea_local = -KB_EV * LN10 * slope  # implied local activation energy (eV)

        hinge_neg = np.maximum(0.0, -ea_local)  # penalize wrong-sign temperature dependence
        hinge_hi = np.maximum(0.0, ea_local - EA_MAX_PHYS)
        hinge_lo = np.maximum(0.0, EA_MIN_PHYS - ea_local)
        # d(loss_phys)/d(ea_local)
        n_c = xc.shape[0]
        dL_dea = (2 * (-hinge_neg) * (-1.0) + 2 * hinge_hi - 2 * hinge_lo) / n_c
        dL_dslope = dL_dea * (-KB_EV * LN10)
        dpred2 = LAMBDA_PHYS * dL_dslope / d_inv_T
        dpred1 = LAMBDA_PHYS * dL_dslope * (-1.0 / d_inv_T)

        dW1, db1 = netB.backward(cache1, dpred1)
        dW2, db2 = netB.backward(cache2, dpred2)
        dW_data, db_data = netB.backward(cache, dOut_data)

        dW = [a + b + c for a, b, c in zip(dW_data, dW1, dW2)]
        db = [a + b + c for a, b, c in zip(db_data, db1, db2)]
        netB.adam_step(dW, db, lr=lr_t, wd=WD)
    if epoch % VAL_EVERY == 0 or epoch == EPOCHS - 1:
        pred_all, _ = netB.forward(F_train)
        lossB_hist.append(float(np.mean((pred_all.ravel() - y_train) ** 2)))
        pred_val, _ = netB.forward(F_val)
        val_mse = float(np.mean((pred_val.ravel() - y_val) ** 2))
        valB_hist.append(val_mse)
        if val_mse < bestB_val:
            bestB_val, bestB_snap = val_mse, netB.snapshot()
netB.load(bestB_snap)


def predict_B(F):
    pred, _ = netB.forward(F)
    return pred.ravel()


# ---------------------------------------------------------------------------
# 6. Model C: physics-embedded (hard-constrained) architecture
#    x (descriptors only) -> (Ea_hat, logA_hat) -> analytic Arrhenius decoder
# ---------------------------------------------------------------------------

netC = MLP([5] + HIDDEN + [2], seed=1)
trainrng = np.random.default_rng(RNG_SEED + 3)
lossC_hist, valC_hist = [], []
bestC_val, bestC_snap = np.inf, netC.snapshot()

X_only_train = X_train  # descriptors only, no T
for epoch in range(EPOCHS):
    lr_t = cosine_lr(epoch, EPOCHS, LR_MAX, LR_MIN)
    for batch_idx in iterate_minibatches(X_only_train.shape[0], BATCH, trainrng):
        Xb = X_only_train[batch_idx]
        Tb = T_train[batch_idx]
        yb = y_train[batch_idx]

        raw, cache = netC.forward(Xb)
        ea_hat = raw[:, 0]
        loga_hat = raw[:, 1]
        pred = loga_hat - ea_hat / (KB_EV * LN10 * Tb)

        n_b = Xb.shape[0]
        dpred = 2 * (pred - yb) / n_b
        dea = dpred * (-1.0 / (KB_EV * LN10 * Tb))
        dloga = dpred * 1.0
        dOut = np.stack([dea, dloga], axis=1)

        dW, db = netC.backward(cache, dOut)
        netC.adam_step(dW, db, lr=lr_t, wd=WD)
    if epoch % VAL_EVERY == 0 or epoch == EPOCHS - 1:
        raw, _ = netC.forward(X_only_train)
        pred_all = raw[:, 1] - raw[:, 0] / (KB_EV * LN10 * T_train)
        lossC_hist.append(float(np.mean((pred_all - y_train) ** 2)))

        raw_v, _ = netC.forward(X_val)
        pred_v = raw_v[:, 1] - raw_v[:, 0] / (KB_EV * LN10 * T_val)
        val_mse = float(np.mean((pred_v - y_val) ** 2))
        valC_hist.append(val_mse)
        if val_mse < bestC_val:
            bestC_val, bestC_snap = val_mse, netC.snapshot()
netC.load(bestC_snap)


def predict_C(X, T):
    raw, _ = netC.forward(X)
    ea_hat, loga_hat = raw[:, 0], raw[:, 1]
    return loga_hat - ea_hat / (KB_EV * LN10 * T), ea_hat, loga_hat


# ---------------------------------------------------------------------------
# 7. Evaluation across the three held-out conditions
# ---------------------------------------------------------------------------

results = {
    "loss_curves": {"A": lossA_hist, "B": lossB_hist, "C": lossC_hist},
    "val_curves": {"A": valA_hist, "B": valB_hist, "C": valC_hist},
    "best_val_mse": {"A": bestA_val, "B": bestB_val, "C": bestC_val},
    "test_sets": {},
}

conditions = {
    "in_distribution": (F_id, X_id, T_id, y_id),
    "temperature_extrapolation": (F_text, X_text, T_text, y_text),
    "family_extrapolation": (F_fam, X_fam, T_fam, y_fam),
}

for name, (F, X, T, y) in conditions.items():
    predA = predict_A(F)
    predB = predict_B(F)
    predC, _, _ = predict_C(X, T)
    results["test_sets"][name] = {
        "n_points": int(F.shape[0]),
        "black_box_A": metrics(y, predA),
        "soft_pinn_B": metrics(y, predB),
        "physics_embedded_C": metrics(y, predC),
    }

# physical interpretability check for model C: recovered Ea vs ground-truth Ea
raw_fam, _ = netC.forward(x_fam_mat)
ea_hat_fam = raw_fam[:, 0]
ea_corr = float(np.corrcoef(ea_hat_fam, ea_fam_mat)[0, 1])
ea_mae = float(np.mean(np.abs(ea_hat_fam - ea_fam_mat)))
results["physics_embedded_Ea_recovery_holdout_family"] = {
    "pearson_r": ea_corr,
    "mae_eV": ea_mae,
}

with open("/home/user/paper/results/results.json", "w") as f:
    json.dump(results, f, indent=2)

# --- illustrative Arrhenius sweeps for two example materials (one in-family, one held-out-family) ---
T_sweep = np.linspace(240.0, 510.0, 60)


def sweep_material(x_row, ea_row, loga_row):
    n = T_sweep.shape[0]
    x_rep = np.tile(x_row, (n, 1))
    F_sweep = featurize(x_rep, T_sweep)
    y_true_sweep = arrhenius_log10sigma(np.full(n, ea_row), np.full(n, loga_row), T_sweep)
    predA_sweep = predict_A(F_sweep)
    predB_sweep = predict_B(F_sweep)
    predC_sweep, _, _ = predict_C(x_rep, T_sweep)
    return dict(T=T_sweep, y_true=y_true_sweep, predA=predA_sweep, predB=predB_sweep, predC=predC_sweep)


example_infamily = sweep_material(x_text_mat[0], ea_text_mat[0], loga_text_mat[0])
example_heldout = sweep_material(x_fam_mat[0], ea_fam_mat[0], loga_fam_mat[0])

# per-point predictions on the temperature-extrapolation test set (for the parity plot)
predA_text = predict_A(F_text)
predB_text = predict_B(F_text)
predC_text, _, _ = predict_C(X_text, T_text)

# save a few arrays for plotting
np.savez(
    "/home/user/paper/results/arrays.npz",
    x_fam_mat=x_fam_mat, ea_fam_mat=ea_fam_mat, loga_fam_mat=loga_fam_mat,
    ea_hat_fam=ea_hat_fam,
    y_text=y_text, predA_text=predA_text, predB_text=predB_text, predC_text=predC_text,
    T_TRAIN=T_TRAIN, T_EXTRAP=T_EXTRAP,
    ex_infamily_T=example_infamily["T"], ex_infamily_true=example_infamily["y_true"],
    ex_infamily_A=example_infamily["predA"], ex_infamily_B=example_infamily["predB"],
    ex_infamily_C=example_infamily["predC"],
    ex_heldout_T=example_heldout["T"], ex_heldout_true=example_heldout["y_true"],
    ex_heldout_A=example_heldout["predA"], ex_heldout_B=example_heldout["predB"],
    ex_heldout_C=example_heldout["predC"],
    lossA_hist=np.array(lossA_hist), valA_hist=np.array(valA_hist),
    lossB_hist=np.array(lossB_hist), valB_hist=np.array(valB_hist),
    lossC_hist=np.array(lossC_hist), valC_hist=np.array(valC_hist),
)

print(json.dumps(results, indent=2))
