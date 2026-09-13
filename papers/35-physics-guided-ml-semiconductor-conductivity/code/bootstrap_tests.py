"""Material-level bootstrap for the paper's key RMSE comparisons, as a
complement to significance_tests.py's seed-level paired tests.

significance_tests.py treats each of the 5 seeds as one paired observation
(n=5), which is a weak basis for a significance claim on its own. This script
instead resamples over the 24 benchmark *materials* -- the more natural unit
of statistical independence here, since curves from the same material share
descriptors and are not independent draws -- pooling each model's per-curve
squared error across all 5 seeds' test sets (results_seed*.json's
"_per_curve_rmse", written by train.py) into a single RMSE the same way the
paper's own headline numbers are computed: sqrt(total squared error / total
point count) over every selected curve's points, not an average of
per-curve RMSEs (which would be a different, smaller-magnitude statistic
whenever error variance differs a lot across curves, as it does here).

A given material can appear as a test material in more than one seed, so
this is not a bootstrap over fully independent units either; we report it
as exploratory for exactly that reason (see the paper's Sections 5.5/5.7
and Limitations).

Run with: python bootstrap_tests.py
"""

import json

import numpy as np

N_BOOTSTRAP = 10000
SEED = 0

SEEDS_DATA = [json.load(open(f"results_seed{i}.json")) for i in range(5)]

COMPARISONS = [
    ("Ensemble (weighted by validation RMSE)", "Gradient-Boosted Trees"),
    ("Ensemble (simple average)", "Gradient-Boosted Trees"),
    ("Ensemble (weighted by validation RMSE)", "Ensemble (simple average)"),
    ("Ensemble (weighted by validation RMSE)", "Ensemble (stacked, Ridge)"),
    ("PGML (2-channel)", "PGML (2-channel + RCN)"),
    ("PGML (2-channel) + GBT residual", "Gradient-Boosted Trees"),
    ("Gradient-Boosted Trees", "Random Forest"),
]


def pooled_records(model):
    """material -> list of (sse, n) tuples, pooled across all 5 seeds."""
    by_material = {}
    for seed_data in SEEDS_DATA:
        for rec in seed_data["_per_curve_rmse"][model]:
            by_material.setdefault(rec["material"], []).append((rec["sse"], rec["n"]))
    return by_material


def pooled_rmse(materials, by_mat):
    sse = sum(s for m in materials for s, _ in by_mat.get(m, []))
    n = sum(k for m in materials for _, k in by_mat.get(m, []))
    return float(np.sqrt(sse / n)) if n > 0 else float("nan")


def bootstrap_compare(model_a, model_b, rng):
    a_by_mat = pooled_records(model_a)
    b_by_mat = pooled_records(model_b)
    materials = sorted(set(a_by_mat) | set(b_by_mat))
    n = len(materials)

    observed_a = pooled_rmse(materials, a_by_mat)
    observed_b = pooled_rmse(materials, b_by_mat)

    diffs = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = rng.choice(materials, size=n, replace=True)
        diffs[i] = pooled_rmse(sample, a_by_mat) - pooled_rmse(sample, b_by_mat)

    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_le = np.mean(diffs <= 0.0)
    p_ge = np.mean(diffs >= 0.0)
    p_two_sided = min(1.0, 2.0 * min(p_le, p_ge))

    return dict(
        n_materials=n,
        a_mean=observed_a, b_mean=observed_b,
        diff_mean=float(observed_a - observed_b),
        bootstrap_diff_mean=float(diffs.mean()), bootstrap_diff_std=float(diffs.std(ddof=1)),
        ci95_lo=float(ci_lo), ci95_hi=float(ci_hi), p_two_sided=float(p_two_sided),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    results = {}
    for a, b in COMPARISONS:
        r = bootstrap_compare(a, b, rng)
        key = f"{a} vs {b}"
        results[key] = r
        print(f"{key}  (n={r['n_materials']} materials, {N_BOOTSTRAP} resamples)")
        print(f"  pooled RMSE: {r['a_mean']:.4f} vs {r['b_mean']:.4f}  diff={r['diff_mean']:+.4f}")
        print(f"  bootstrap diff: mean={r['bootstrap_diff_mean']:+.4f} std={r['bootstrap_diff_std']:.4f}"
              f"  95% CI=[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]  p={r['p_two_sided']:.4f}\n")
    with open("bootstrap_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved bootstrap_tests.json")
