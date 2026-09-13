"""Generate publication figures from results/results.json and results/arrays.npz."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLORS = {"A": "#7A7A7A", "B": "#2C7FB8", "C": "#D95F02"}
LABELS = {"A": "Black-box MLP", "B": "Soft physics-informed (loss residual)", "C": "Physics-embedded (hard constraint)"}

with open("/home/user/paper/results/results.json") as f:
    R = json.load(f)
D = np.load("/home/user/paper/results/arrays.npz")

# ---------------------------------------------------------------------------
# Figure 1: RMSE across the three generalization conditions
# ---------------------------------------------------------------------------
conditions = ["in_distribution", "temperature_extrapolation", "family_extrapolation"]
cond_labels = ["In-distribution", "Temperature\nextrapolation", "Material-family\nextrapolation"]
model_keys = [("black_box_A", "A"), ("soft_pinn_B", "B"), ("physics_embedded_C", "C")]

fig, ax = plt.subplots(figsize=(6.2, 4.0))
x = np.arange(len(conditions))
width = 0.25
for i, (mk, short) in enumerate(model_keys):
    vals = [R["test_sets"][c][mk]["rmse_log10sigma"] for c in conditions]
    ax.bar(x + (i - 1) * width, vals, width, label=LABELS[short], color=COLORS[short])
ax.set_xticks(x)
ax.set_xticklabels(cond_labels)
ax.set_ylabel(r"RMSE, $\log_{10}\sigma$ (S/cm)")
ax.set_title("Generalization error across three held-out conditions")
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig("/home/user/paper/figures/fig1_rmse_by_condition.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: parity plot on the temperature-extrapolation test set
# ---------------------------------------------------------------------------
y_text = D["y_text"]
preds = {"A": D["predA_text"], "B": D["predB_text"], "C": D["predC_text"]}

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
lo, hi = y_text.min() - 0.3, y_text.max() + 0.3
for ax, short in zip(axes, ["A", "B", "C"]):
    ax.scatter(y_text, preds[short], s=6, alpha=0.35, color=COLORS[short], linewidths=0)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(LABELS[short], fontsize=9)
    ax.set_xlabel(r"True $\log_{10}\sigma$")
    r2 = R["test_sets"]["temperature_extrapolation"][
        {"A": "black_box_A", "B": "soft_pinn_B", "C": "physics_embedded_C"}[short]]["r2"]
    ax.text(0.05, 0.90, f"$R^2$={r2:.2f}", transform=ax.transAxes, fontsize=8)
axes[0].set_ylabel(r"Predicted $\log_{10}\sigma$")
fig.suptitle("Temperature-extrapolation test set (T = 420-500 K, unseen during training)", fontsize=10)
fig.tight_layout()
fig.savefig("/home/user/paper/figures/fig2_parity_temperature_extrapolation.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: Arrhenius plots for two example materials
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=False)

for ax, prefix, title in zip(
    axes,
    ["ex_infamily", "ex_heldout"],
    ["Example material, in-distribution family", "Example material, held-out family\n(compositional extrapolation)"],
):
    T = D[f"{prefix}_T"]
    inv_T_1000 = 1000.0 / T
    ax.plot(inv_T_1000, D[f"{prefix}_true"], "k-", lw=2, label="Ground truth (Arrhenius law)")
    ax.plot(inv_T_1000, D[f"{prefix}_A"], "--", color=COLORS["A"], label=LABELS["A"])
    ax.plot(inv_T_1000, D[f"{prefix}_B"], "--", color=COLORS["B"], label=LABELS["B"])
    ax.plot(inv_T_1000, D[f"{prefix}_C"], "--", color=COLORS["C"], label=LABELS["C"])
    ax.axvspan(1000.0 / 500.0, 1000.0 / 400.0, color="gray", alpha=0.12, label="Extrapolation region (T > 400 K)")
    ax.set_xlabel(r"$1000/T$ (K$^{-1}$)")
    ax.set_ylabel(r"$\log_{10}\sigma$ (S/cm)")
    ax.set_title(title, fontsize=9.5)

axes[0].legend(fontsize=6.8, loc="upper left")
fig.tight_layout()
fig.savefig("/home/user/paper/figures/fig3_arrhenius_examples.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: recovered vs true activation energy (physics-embedded model),
#           held-out family
# ---------------------------------------------------------------------------
ea_true = D["ea_fam_mat"]
ea_hat = D["ea_hat_fam"]
fig, ax = plt.subplots(figsize=(4.6, 4.3))
ax.scatter(ea_true, ea_hat, s=10, alpha=0.5, color=COLORS["C"], linewidths=0)
lo, hi = min(ea_true.min(), ea_hat.min()) - 0.02, max(ea_true.max(), ea_hat.max()) + 0.02
ax.plot([lo, hi], [lo, hi], "k--", lw=1)
r = R["physics_embedded_Ea_recovery_holdout_family"]["pearson_r"]
mae = R["physics_embedded_Ea_recovery_holdout_family"]["mae_eV"]
ax.set_xlabel(r"True activation energy $E_a$ (eV)")
ax.set_ylabel(r"Recovered $\hat{E}_a$ (eV)")
ax.set_title(f"Physics-embedded model: recovered $E_a$\nheld-out family, Pearson $r$={r:.2f}, MAE={mae:.2f} eV", fontsize=9.5)
fig.tight_layout()
fig.savefig("/home/user/paper/figures/fig4_ea_recovery.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5: training/validation loss curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.0))
val_every = 20
for short in ["A", "B", "C"]:
    hist = D[f"val{short}_hist"]
    epochs = np.arange(len(hist)) * val_every
    ax.plot(epochs, hist, color=COLORS[short], label=LABELS[short])
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation MSE (log10 sigma)")
ax.set_title("Validation loss curves (early-stopping checkpoint selection)")
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig("/home/user/paper/figures/fig5_validation_curves.png")
plt.close(fig)

print("Figures written to /home/user/paper/figures/")
