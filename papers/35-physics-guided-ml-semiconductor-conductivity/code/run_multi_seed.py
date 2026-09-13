"""Runs the full comparison (Section 5) over multiple seeds -- each seed
reshuffles both weight initialization and the material-level train/val/test
split -- and reports mean +/- std per model per metric, as described in
Section 4.3 of the paper. Also aggregates the per-seed count of physically
checkable test curves, since any single seed's consistency-violation rate is
computed over a small number of curves and is noisy on its own.

Usage: python run_multi_seed.py [--seeds 5] [--epochs 2000]
"""

import argparse
import json

import numpy as np

from train import run


def summarize(all_results, n_seeds):
    # Leading-underscore keys (e.g. "_per_curve_rmse", written by train.py
    # for bootstrap_tests.py) are per-curve breakdowns, not per-model
    # metrics, and are intentionally excluded from this seed-level summary.
    labels = [l for l in all_results[0].keys() if not l.startswith("_")]
    metrics = ["rmse_log10", "r2", "consistency_violation", "extrapolation_rmse"]
    summary = {}
    for label in labels:
        summary[label] = {}
        for metric in metrics:
            vals = [r[label][metric] for r in all_results if not np.isnan(r[label].get(metric, np.nan))]
            if vals:
                summary[label][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        n_checkable_total = sum(r[label].get("n_checkable_curves", 0) for r in all_results)
        summary[label]["n_checkable_curves_total"] = n_checkable_total
    return summary


def print_summary(summary):
    header = f"{'Model':32s} {'RMSE(log10 sigma)':>20s} {'R2':>14s} {'Extrap. RMSE':>16s} {'Consistency viol. (%)':>24s}"
    print(header)
    print("-" * len(header))
    for label, m in summary.items():
        def fmt(key, scale=1.0):
            if key not in m:
                return "n/a"
            return f"{m[key]['mean']*scale:.3f} +/- {m[key]['std']*scale:.3f}"
        print(f"{label:32s} {fmt('rmse_log10'):>20s} {fmt('r2'):>14s} "
              f"{fmt('extrapolation_rmse'):>16s} {fmt('consistency_violation', 100.0):>24s} "
              f"(n_total={m.get('n_checkable_curves_total', 0)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--out", type=str, default="results_multiseed.json")
    args = parser.parse_args()

    all_results = []
    for seed in range(args.seeds):
        print(f"\n=========== Seed {seed} ===========")
        all_results.append(run(seed=seed, epochs=args.epochs, patience=args.patience,
                                out_path=f"results_seed{seed}.json"))

    summary = summarize(all_results, args.seeds)
    print("\n=========== Summary over {} seeds ===========".format(args.seeds))
    print_summary(summary)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved multi-seed summary to {args.out}")
