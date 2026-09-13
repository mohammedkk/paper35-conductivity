"""Paired significance tests across the 5 seeds for the paper's key model
comparisons, computed directly from the released results_seed*.json files
(same seeds, same splits per seed, so a paired test is appropriate).
Run with: python significance_tests.py
"""

import json

import numpy as np
from scipy import stats

SEEDS = [json.load(open(f"results_seed{i}.json")) for i in range(5)]

COMPARISONS = [
    ("Ensemble (weighted by validation RMSE)", "Gradient-Boosted Trees", "rmse_log10"),
    ("Ensemble (weighted by validation RMSE)", "Gradient-Boosted Trees", "r2"),
    ("Ensemble (weighted by validation RMSE)", "Gradient-Boosted Trees", "extrapolation_rmse"),
    ("Ensemble (simple average)", "Gradient-Boosted Trees", "rmse_log10"),
    ("Ensemble (simple average)", "Gradient-Boosted Trees", "extrapolation_rmse"),
    ("Ensemble (weighted by validation RMSE)", "Ensemble (simple average)", "rmse_log10"),
    ("Ensemble (weighted by validation RMSE)", "Ensemble (simple average)", "extrapolation_rmse"),
    ("Ensemble (weighted by validation RMSE)", "Ensemble (stacked, Ridge)", "rmse_log10"),
    ("Ensemble (weighted by validation RMSE)", "Ensemble (stacked, Ridge)", "extrapolation_rmse"),
    ("PGML (2-channel)", "PGML (2-channel + RCN)", "rmse_log10"),
    ("PGML (2-channel)", "PGML (2-channel + RCN)", "extrapolation_rmse"),
    ("PGML (2-channel) + GBT residual", "Gradient-Boosted Trees", "extrapolation_rmse"),
    ("PGML (2-channel) + GBT residual", "Gradient-Boosted Trees", "rmse_log10"),
    ("Ensemble (10 models, simple average)", "Ensemble (simple average)", "rmse_log10"),
    ("Ensemble (10 models, simple average)", "Ensemble (simple average)", "extrapolation_rmse"),
    ("Ensemble (10 models, stacked Ridge)", "Ensemble (stacked, Ridge)", "rmse_log10"),
    ("Ensemble (10 models, stacked Ridge)", "Ensemble (stacked, Ridge)", "extrapolation_rmse"),
    ("Gradient-Boosted Trees", "Random Forest", "rmse_log10"),
]


def paired(model_a, model_b, metric):
    a = np.array([s[model_a][metric] for s in SEEDS])
    b = np.array([s[model_b][metric] for s in SEEDS])
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    return dict(
        a_mean=float(a.mean()), b_mean=float(b.mean()),
        diff_mean=float(diff.mean()), diff_std=float(diff.std(ddof=1)),
        t_stat=float(t_stat), t_p=float(t_p), w_stat=float(w_stat), w_p=float(w_p),
    )


if __name__ == "__main__":
    results = {}
    for a, b, m in COMPARISONS:
        r = paired(a, b, m)
        key = f"{a} vs {b} [{m}]"
        results[key] = r
        print(f"{key}\n  means: {r['a_mean']:.4f} vs {r['b_mean']:.4f}  diff={r['diff_mean']:+.4f}"
              f"\n  paired t-test: t={r['t_stat']:.3f}, p={r['t_p']:.4f}"
              f"\n  Wilcoxon signed-rank: W={r['w_stat']:.1f}, p={r['w_p']:.4f}\n")
    with open("significance_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved significance_tests.json")
