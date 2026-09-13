# Physics-Guided Machine Learning for Prediction of Dielectric Properties of Functional Materials

> **This repository hosts more than one independent paper.** This README describes the dielectric-constant paper below (root-level files: `main.tex`, `analysis/`, `data/`, `figures/`, root `references.bib`). Two other, unrelated papers live alongside it: **"Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials"** in [`paper-32-conductivity/`](paper-32-conductivity/) (see [`paper-32-conductivity/README.md`](paper-32-conductivity/README.md)), and **"Physics-Guided Learning for Synthetic Temperature-Dependent Semiconductor Conductivity: Effects of Closed-Form Model Mismatch"** (paper 35) in [`papers/35-physics-guided-ml-semiconductor-conductivity/`](papers/35-physics-guided-ml-semiconductor-conductivity/) (see that folder's own README). Each paper has its own manuscript, code, data/results, and bibliography.

Original-research manuscript: an empirical test of whether embedding a
classical physics equation (the Clausius–Mossotti relation) improves
machine-learning prediction of the static dielectric constant of real,
experimentally measured inorganic-oxide microwave dielectrics, compared
against a purely data-driven model trained on the same descriptors.

**Headline result:** on this dataset, physics guidance did not help. A
physics-only Clausius–Mossotti baseline performs far worse than predicting
the mean (R² ≈ −2), a data-driven gradient-boosted model achieves R² ≈ 0.66,
and combining the two (physics baseline + ML residual correction) is
statistically indistinguishable from the data-driven model alone
(R² ≈ 0.63). We trace the physics baseline's failure to a sign reversal:
composition-weighted polarizability normalized by an ionic-volume proxy is
*negatively* correlated with the measured dielectric constant, because the
highest-permittivity compounds in this dataset owe their response to
structural/lattice-dynamical mechanisms that a static, electronic
ionic-polarizability additivity rule does not capture.

**We then tried to fix it.** Five further physics-baseline variants were
built and cross-validated under the identical protocol — enriching the
physics equation's input with the two descriptors most correlated with the
target, separating an explicit electronic term from an additive
lattice-contrast term, and re-pairing each repaired baseline with the
residual-correction architecture. The best repair matches but does not
exceed the original hybrid (R² ≈ 0.63), and one repair that enriches the
equation's input without preserving its physical boundedness makes the
hybrid measurably *worse* (R² ≈ 0.34), traced to 84% of compounds being
pushed past the Clausius–Mossotti equation's pole.

**A simulated peer review then caught two real methodological gaps, both now
fixed.** (1) The paper originally called M2-vs-M3 "statistically
indistinguishable" based on overlapping mean±s.d. error bars — but the
cross-validation design is paired (identical folds for every model), and a
correct paired test on that same data shows the gap *is* statistically
significant (p < 10⁻⁹ throughout), which if anything sharpens the paper's
claim: physics guidance is measurably, not just nominally, worse here. (2)
33% of the dataset consists of near-duplicate compositional "families"
(e.g. `Mg0.8Zn0.2Al2O4` vs `Mg0.6Zn0.4Al2O4`) that plain random k-fold
splitting can leak across train/test. A family-grouped re-evaluation
(`analysis/grouped_cv.py`) shows this does inflate both M2 and M3's absolute
R² by ~0.04 — but by an almost identical amount for both, so the M2-vs-M3
gap and its significance survive intact (p ≈ 10⁻¹²).

**The review's two remaining (recommended, not required) suggestions were
then also implemented.** (3) Substituting a structurally different ML
algorithm (`RandomForestRegressor` in place of gradient boosting,
`analysis/model_rf.py`) reproduces essentially the same significant M2-vs-M3
gap (p ≈ 7×10⁻¹³) — the finding isn't an artifact of one algorithm family.
(4) The paper's proposed "lone-pair cation" mechanism (Bi³⁺/Pb²⁺/Sn²⁺/Tl⁺
compounds drive the failure) was directly tested by stratifying the dataset
(`analysis/subgroup_analysis.py`), using genuinely held-out repeated-CV
predictions rather than an in-sample fit. The result is a nuanced, honest
correction rather than a clean confirmation: the physics-vs-ML *gap* turns
out to be a dataset-wide effect, not concentrated in lone-pair chemistry —
but lone-pair-cation compounds are independently confirmed to be the
hardest subgroup to predict for *every* model tested (data-driven model R²
collapses from 0.76 to 0.15 on that subgroup alone), which is itself a
new, informative finding about what these composition-level descriptors
are missing.

**Finally, the paper's own most concrete future-work suggestion — a
genuine structural-instability descriptor — was built and tested, closing
that loop too.** Inspecting the raw data revealed that one "site" in both
the ternary and quaternary tables is actually the oxygen anion sublattice
(fixed coordination number 2, ionic radius 1.35 Å in every single
compound), not a fourth cation — previously folded indiscriminately into
the same composition-weighted features as the real cation sites. Separating
them (`analysis/tolerance_factor.py`) enabled a genuine, Goldschmidt-style
tolerance-factor descriptor built entirely from existing columns. It also
failed to help — not because the idea was wrong, but because the resulting
descriptor turned out to be nearly redundant (r = 0.964) with the
ionic-radius-heterogeneity feature the data-driven model already had
access to. A mechanistically clear negative result, not just one more
failed attempt. See `main.pdf` / `main.tex` for the full paper.

## Contents

- `main.tex`, `references.bib`, `main.pdf` — the manuscript.
- `data/ExpDiele/` — the experimental dataset analyzed, redistributed with
  attribution from the source repository (see **Data provenance** below).
- `analysis/` — all Python code used to produce every number, table, and
  figure in the paper:
  - `load_data.py` — loads and unifies the ternary/quaternary raw data into
    a single arity-independent feature table (composition-weighted mean and
    standard deviation of five per-site ionic descriptors).
  - `model.py` — defines the three compared models (physics-only
    Clausius–Mossotti, data-driven gradient boosting, physics-guided
    hybrid) and runs the repeated 5-fold × 20-repeat cross-validation.
    Running it (`python3 model.py`) regenerates `cv_results_raw.csv`,
    `cv_results_summary*.csv`, and `full_fit_extra.json`.
  - `make_figures.py` — regenerates the first four figures in `../figures/`
    from the results above.
  - `model_v2.py` — the repair attempts: five further physics-baseline
    variants (M1c enriched-b, M5 two-mechanism, and hybrids M3b/M4/M6)
    targeting the specific failure mode diagnosed in the paper, evaluated
    under the identical cross-validation protocol.
  - `make_figures_v2.py` — regenerates the two repair-attempt figures.
  - `significance_tests.py` — **required fix #1**: paired *t*-test and
    Wilcoxon signed-rank test on the per-fold R² differences for every
    headline model comparison (the design is paired, so this is the
    correct test — not comparing each model's own mean±s.d. by eye).
  - `grouped_cv.py` — **required fix #2**: compositional-family-grouped
    repeated cross-validation (via `load_data.formula_skeleton`), so that
    near-duplicate solid-solution-family members are never split across
    train/test; reports the family-overlap statistic and re-evaluates
    M1/M1b/M2/M3, confirming the M2-vs-M3 gap and its significance survive.
  - `make_figures_v3.py` — regenerates the plain-vs-grouped-CV figure.
  - `model_rf.py` — **recommended fix #1**: re-evaluates M2/M3 with
    `RandomForestRegressor` in place of gradient boosting, to check the
    M2-vs-M3 gap isn't specific to one ML algorithm family.
  - `subgroup_analysis.py` — **recommended fix #2**: flags lone-pair-cation
    compounds (Bi/Pb/Sn/Tl) and compares out-of-fold accuracy within each
    subgroup, directly testing the paper's proposed failure mechanism
    instead of only asserting it.
  - `make_figures_v4.py` — regenerates the RF-comparison and subgroup
    figures.
  - `tolerance_factor.py` — closes the paper's own flagged future-work
    item: separates the dataset's previously unexploited invariant oxygen
    site from the true cation sites, builds a genuine Goldschmidt-style
    tolerance-factor descriptor, and tests it as a physics-only model
    (M7), a physics-guided hybrid (M8), and a plain added ML feature —
    finding it nearly redundant (r=0.964) with a feature already in use.
  - `requirements.txt` — Python dependencies (numpy, pandas, scipy,
    scikit-learn, matplotlib).
- `figures/` — the nine generated figures used in the manuscript.

## Reproducing the results

```bash
cd analysis
pip install -r requirements.txt
python3 model.py             # regenerates cv_results_*.csv and full_fit_extra.json
python3 make_figures.py      # regenerates ../figures/fig_{cv_comparison,physics_diagnostic,parity,feature_importance}.png
python3 model_v2.py          # regenerates cv_results_v2_*.csv (the repair attempts)
python3 make_figures_v2.py   # regenerates ../figures/fig_{repair_comparison,pole_saturation}.png
python3 significance_tests.py # regenerates significance_tests.csv (required fix #1)
python3 grouped_cv.py         # regenerates cv_results_grouped_*.csv, family_overlap_stats.csv (required fix #2)
python3 make_figures_v3.py    # regenerates ../figures/fig_grouped_cv.png
python3 model_rf.py           # regenerates cv_results_rf_*.csv, significance_tests_rf.csv (recommended fix #1)
python3 subgroup_analysis.py  # regenerates subgroup_analysis.csv, subgroup_per_compound.csv (recommended fix #2)
python3 make_figures_v4.py    # regenerates ../figures/fig_{rf_comparison,subgroup}.png
python3 tolerance_factor.py   # regenerates cv_results_tf_*.csv, significance_tests_tf.csv
```

Then rebuild the PDF:

```bash
latexmk -pdf main.tex
```

## Data provenance

The dataset in `data/ExpDiele/` is **not original to this repository**. It
is redistributed, with full attribution, from the public GitHub repository
[`yabeiwu/DielectricProperties`](https://github.com/yabeiwu/DielectricProperties),
which accompanies:

> Wu, Y., Ye, C., & Zhang, W. (2025). Interpretable model of dielectric
> constant for rational design of microwave dielectric materials: a machine
> learning study. *Journal of Materials Informatics*, 5, 7.
> https://doi.org/10.20517/jmi.2024.75

All credit for collecting and curating the underlying experimental
measurements belongs to those authors. The source repository does not state
an explicit license; the data are redistributed here solely to make the
present paper's analysis exactly reproducible, consistent with the source
repository's own description of itself as a "Machine Learning for
Dielectric Property Database." Anyone wishing to reuse the data beyond
reproducing this paper should contact the original authors.

Every model, evaluation, and figure built from this data in `analysis/` was
written and executed for this paper and is original.
