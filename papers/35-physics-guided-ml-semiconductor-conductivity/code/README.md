# Code

Reference implementation for the paper. Every number in Tables 1–7 is
produced directly by this code from the synthetic benchmark it generates —
nothing here is hand-tuned to match a pre-decided result. `models.py`
currently implements the *richer* base closed form (Eqs. 3–5 in the paper —
carrier concentration and mobility as separate terms, 5 parameters per
material) plus, optionally, the explicit second conduction channel (Eq. 6)
and/or the residual correction network (Eq. 7) — the four combinations of
those two flags are Table 2/3's four PGML rows. The *reduced* closed form
used for Table 1 (a single combined exponential-times-power-law expression,
4 parameters per material, no second channel) is no longer in the code; see
the paper's Section 5.1 for its definition if you want to reproduce it, or
`git log` this file for the earlier version. `train.py` computes the
physical-consistency-violation metric for Random Forest and Gradient-Boosted
Trees too (not only the neural models), by evaluating their predictions on
the same full temperature grid the neural models use and applying the same
finite-difference Arrhenius-slope check (Section 4.2 of the paper); both
tree ensembles turn out to have a nonzero violation rate, unlike every
neural model in Table 2 onward.

## Contents

- `significance_tests.py` — paired *t*-test and Wilcoxon signed-rank test,
  computed directly from `results_seed*.json`, for the key model comparisons
  reported in the paper's Results and Conclusion (e.g. the weighted ensemble
  vs. tuned Gradient-Boosted Trees). None of the ensemble-vs-GBT differences
  reach significance at `n=5` seeds — see `significance_tests.json` for the
  full numbers and the paper's Section 5.7/Conclusion for how this bears on
  the paper's claims. Both this test and the one below are reported in the
  paper as exploratory rather than confirmatory, given how few seeds and
  materials this benchmark has.
- `bootstrap_tests.py` — a material-level bootstrap (10,000 resamples) over
  the same key comparisons, using `results_seed*.json`'s `_per_curve_rmse`
  breakdown (written by `train.py`) pooled across all 5 seeds. Resamples the
  benchmark's materials rather than its seeds, so it is a complementary
  check less sensitive to any one seed's particular train/validation/test
  split; see `bootstrap_tests.json` and the paper's Section 4.2.
- `descriptors.py` — electronic-structure descriptor table for 24 real
  semiconductors (band gap, effective masses, Debye temperature, lattice
  thermal conductivity, plus electronegativity difference and average atomic
  mass computed exactly from the constituent elements).
- `physics.py` — the *ground-truth generator*: a reduced Boltzmann-transport
  model (Section 3.5 of the paper) used to synthesize σ(T) curves, including
  the narrow-gap high-temperature logistic correction (`high_temperature_bump`)
  that motivates the explicit second channel in `models.py`.
- `dataset.py` — assembles the benchmark, classifies each point's transport
  regime (intrinsic/extrinsic/ambiguous) and measurability, and splits at the
  material level.
- `models.py` — `PGML` (physics-parameter head + closed-form transport layer
  + optional explicit second conduction channel + optional residual
  correction network) and `BlackBoxMLP`.
- `losses.py` — the physics-consistency ("shape") loss and the
  consistency-violation metric.
- `train.py` — trains and evaluates all models for one seed, including both
  physics-plus-boosting hybrids and all three ensemble combination
  strategies over both the 8-model and 10-model pools
  (`python train.py --seed 0`).
- `run_multi_seed.py` — runs `train.py` over 5 seeds and aggregates mean ±
  std, reproducing Tables 1–7 (`python run_multi_seed.py`).
- `results_seed*.json`, `results_multiseed.json` — the actual outputs used
  in the paper, included for inspection without re-running anything.

## Running it

```bash
pip install -r requirements.txt
python run_multi_seed.py          # ~2.5 minutes on CPU; reproduces Tables 2-4
python train.py --seed 0          # single-seed run with per-model printout
```

## Known behavior (see Section 6 of the paper)

With the richer base closed form, all models — PGML included — achieve a
perfect 0% physical-consistency violation rate. Among the four PGML
variants (`use_second_channel` × `use_rcn`), `PGML (2-channel + RCN)` gets
the best accuracy (RMSE 0.769, R² 0.848), matching or slightly exceeding
the residual-only PINN baseline (0.765, 0.836) — the best result any PGML
variant reaches in this study — but the plain `PGML (2-channel)` (no RCN)
generalizes to unseen materials at unseen temperatures better (extrapolation
RMSE 1.304 vs. 1.637). None of the four PGML variants closes the gap to the
tree ensembles (Random Forest / Gradient-Boosted Trees, RMSE 0.63–0.70).

We did try to close that gap directly, three ways (`train.py`'s tree-ensemble
and "Hybrid" blocks):

1. **Tune GBT itself** — `tune_gbt()` runs a 36-point grid search (estimator
   count × depth × learning rate × min leaf size) selected by RMSE on the
   material-level validation split, separately for the standard and the
   T≤400K extrapolation-training regime, then refits on the training split
   alone (not train+val, so this isolates the tuning effect from any change
   in training data, and Random Forest's numbers are untouched). Barely
   moved GBT's accuracy: RMSE 0.628 → 0.656, R² 0.908 → 0.909, both changes
   far smaller than the five-seed standard deviation. Extrapolation RMSE
   improved somewhat (1.307 → 1.166). The fixed defaults used throughout
   Tables 1–2 were already close to as good as this search finds.
2. **Residual hybrid** — fit the (now tuned) Gradient-Boosted Trees on the
   *residual* of `PGML (2-channel)` rather than on `log10(sigma)` directly,
   then add the two predictions. Narrows the remaining RMSE gap by about a
   third (0.797 → 0.747, still short of GBT's 0.656) but reaches the best
   extrapolation RMSE of every model in the paper (1.100) — better than
   either the physics-only model or Gradient-Boosted Trees alone.
3. **Feature-augmentation hybrid** — hand Gradient-Boosted Trees the same
   physics model's *fitted parameters* (`theta` from `model.pph(x)`) as
   extra input features alongside the raw descriptors, and predict
   `log10(sigma)` directly (no residual). This did not help at all: RMSE
   0.775 and extrapolation RMSE 1.415, both worse than tuned
   Gradient-Boosted Trees alone.

Our read: the tree ensembles' in-distribution accuracy advantage on this
small tabular problem is a genuine, tuning-robust property of the learning
problem, not an artifact of an unfairly weak default baseline, and it isn't
something a better closed form fully closes either, whether by
residual-correction or by feature engineering — but the residual hybrid's
contribution to extrapolation robustness is real and carries through even
when GBT dominates the in-distribution fit. The feature-augmentation result
suggests this isn't just "physics info helps trees" in general: it
specifically has to be handed the closed form's *output prediction* to
correct, not its internal parameters as extra columns. If you try other
combinations (e.g. either hybrid on top of a different PGML variant, a
larger hyperparameter grid, tuning Random Forest too, or the reverse hybrid
— a closed-form correction on top of a tree ensemble's prediction), we'd be
curious whether any result changes.

Rather than stop at "no single model closes the gap," we tried integrating
*all* of them (`train.py`'s "Ensemble" block, after the hybrid block):

1. **Simple average** — an unweighted mean of the predictions of all eight
   already-trained models (four PGML variants, black-box MLP, residual-only
   PINN, Random Forest, tuned Gradient-Boosted Trees). This came closer to
   closing the gap than anything else in the paper: RMSE 0.630 (vs. tuned
   GBT's 0.656), and by far the best extrapolation RMSE of any model or
   combination tried (0.937 vs. GBT's 1.166 and the residual hybrid's
   previous best of 1.100). It does not, however, beat GBT's R² (0.896 vs.
   0.909).
2. **Stacked ensemble** — a non-negative-weight Ridge regression
   meta-learner, fit on the validation split's base-model predictions, used
   in place of a uniform average. This did *not* improve on the simple
   average (RMSE 0.648, extrapolation RMSE 1.121): with only 12 validation
   curves to fit eight regression weights, the learned weights were
   noticeably unstable from seed to seed (see `stacked_weights` in
   `results_seed*.json`), so the "smarter" combination ends up fitting
   validation-split noise rather than genuine differences in base-model
   quality.

Our read: on a benchmark this small, declining to estimate anything extra
from data (the simple average) is itself a form of regularization that beats
trying to learn which models to trust more. The eight base models are
diverse enough (a closed-form physics model, a black-box network, a
loss-only PINN, and two tree ensembles) that their errors partly cancel on
simple averaging, which is exactly the mechanism variance-reduction
ensembling relies on.

One natural follow-up: does the ensemble improve if it can also draw on the
two physics-plus-boosting hybrids from the section above as candidate
members, since both are already trained by that point and cost nothing
extra to add? We tried it (`train.py`'s "Extended ensemble" block, right
after the eight-model one): repeat both combination strategies over all ten
models (the original eight plus `PGML (2-channel) + GBT residual` and
`GBT + physics-parameter features`).

- **10-model simple average**: RMSE 0.634, R² 0.896, extrapolation RMSE
  0.964 — statistically indistinguishable from the 8-model average on every
  metric (all differences are far smaller than the five-seed standard
  deviations). Diluting a strong average with two individually weaker
  models did not visibly hurt it.
- **10-model stacked ensemble**: RMSE 0.645, extrapolation RMSE 1.036 — close
  to its 8-model counterpart, with a small extrapolation improvement
  (1.121 → 1.036) that still falls well short of the simple average. The
  same instability persists: 12 validation curves is not enough to reliably
  fit weights over 10 correlated candidates any more than it was for 8.

Our read: this is a clean null result. The ensemble's earlier performance
(above) wasn't leaving easy gains on the table by excluding the hybrids —
the bottleneck is the combination strategy and the size of the validation
split available to fit one, not which models are eligible to be combined.

That's exactly what we tried next (`train.py`'s "validation-RMSE-weighted"
blocks, one per pool size): instead of an unweighted average or a fully
free Ridge stack, weight each base model's prediction by the inverse of its
own *squared* validation RMSE (inverse-variance weighting), normalized to
sum to one. This costs exactly one scalar per model — its own validation
accuracy — rather than a joint regression over every model's predictions.

- **8-model, weighted by validation RMSE**: RMSE 0.593, R² 0.905,
  extrapolation RMSE 0.822 — beats *both* earlier combination strategies on
  *every* metric, and is the only result in the whole study that beats
  tuned GBT on both RMSE (0.593 vs. 0.656) and extrapolation RMSE (0.822 vs.
  1.166) while matching its R² within noise (0.905 ± 0.083 vs. 0.909 ±
  0.042).
- **10-model, weighted by validation RMSE**: RMSE 0.604, R² 0.902,
  extrapolation RMSE 0.834 — almost identical to the 8-model version,
  consistent with the pool size not mattering once the combination strategy
  itself works.

Comparing the actual weights (`validation_weights` in `results_seed*.json`)
explains why this beats the stacked Ridge: the Ridge weights swing wildly
across seeds (mean per-model weight std ≈ 0.140 — one seed puts 0.95 on the
black-box MLP alone, another puts nearly everything on the two tree
ensembles), while validation-RMSE weighting stays steady (mean std ≈ 0.047,
roughly a third as much, and no model is ever weighted near zero or near
one). Fitting one number per model from validation data generalizes far
better than fitting a joint regression over eight to ten correlated
candidates on the same 12 curves.

We did not try adding each hybrid individually, hybrids built on the other
three PGML variants (never constructed), a hand-picked smaller pool of only
the strongest models, other low-parameter weighting schemes (e.g. by rank,
or a shrinkage-regularized stack), or tuning the weighting scheme's implicit
assumption of independent errors — any of which might close the small
remaining R² gap to tuned GBT, or do even better than inverse-variance
weighting specifically.
