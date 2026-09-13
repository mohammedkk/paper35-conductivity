# Paper 35 — Physics-Guided Machine Learning for Temperature-Dependent Electrical Conductivity Prediction in Semiconductor Materials Using Electronic-Structure Descriptors

A physics-guided machine learning (PGML) architecture that predicts the temperature-dependent
electrical conductivity of semiconductors from electronic-structure descriptors (band gap,
effective masses, density of states at the Fermi level, doping concentration, Debye
temperature) by combining a Boltzmann-transport-derived closed-form layer with a correction
for higher-order effects — tested against a black-box MLP, tree ensembles, and a loss-only
physics-informed baseline on a released, fully reproducible synthetic benchmark. The paper
reports seven successive experiments, not just the one that worked best: an initial, more
reduced closed form was the *worst*-performing model on every metric, including the
physical-consistency violation rate it was designed to improve; revising it into a richer
closed form that matches the benchmark generator's functional family resolved the consistency
problem entirely and closed most of the accuracy gap; replacing the generic residual
correction network with an explicit second conduction channel, structurally matched to the
benchmark's own narrow-gap high-temperature correction, closed nearly all of the rest —
matching the strongest neural baseline, though not the tree ensembles; a physics-plus-
boosting hybrid (gradient-boosted trees fit on the closed form's residual) narrowed the
remaining gap to the tree ensembles without closing it, while reaching the best extrapolation
result of any single model tested; integrating all eight models tried into one
predictor — a simple unweighted average — came closer to closing the gap than anything else in
the paper, matching tuned gradient-boosted trees on RMSE and clearly beating it on
extrapolation, while a "smarter" stacked variant that learned combination weights from data
underperformed the plain average because of weight instability on the small validation split;
extending both ensemble strategies to also include the two physics-plus-boosting
hybrids as candidate members left both essentially unchanged, showing the bottleneck was the
combination strategy and the small validation split, not which models were eligible to be
combined; and finally, weighting each base model's prediction by the inverse of its own squared
validation error, rather than averaging uniformly or fitting a fully free regression, beat both
earlier combination strategies on every metric and is the only result in the paper that is
simultaneously more accurate and more extrapolation-robust than the strongest tree-ensemble
baseline while matching its R² within noise. See `code/README.md` and Section 6 of the paper
for the full comparison and the likely mechanism.

## Contents

- `Physics-Guided Learning for Synthetic Temperature-Dependent Semiconductor Conductivity - Effects of Closed-Form Model Mismatch.tex` — paper source (LaTeX), with an English abstract and a bilingual Arabic abstract.
- `references.bib` — bibliography (BibTeX).
- `Physics-Guided Learning for Synthetic Temperature-Dependent Semiconductor Conductivity - Effects of Closed-Form Model Mismatch.pdf` — compiled PDF.
- `code/` — full reference implementation (benchmark generator, PGML, all baselines, training
  and multi-seed evaluation scripts) that reproduces every number in the paper; see
  `code/README.md`.

## Compiling

The Arabic abstract requires XeLaTeX (via `polyglossia`/`fontspec`) and the **Amiri** font
(`fonts-hosny-amiri` on Debian/Ubuntu, or install from https://www.amirifont.org/).

```bash
xelatex "Physics-Guided Learning for Synthetic Temperature-Dependent Semiconductor Conductivity - Effects of Closed-Form Model Mismatch.tex"
bibtex "Physics-Guided Learning for Synthetic Temperature-Dependent Semiconductor Conductivity - Effects of Closed-Form Model Mismatch"
xelatex "Physics-Guided Learning for Synthetic Temperature-Dependent Semiconductor Conductivity - Effects of Closed-Form Model Mismatch.tex"
xelatex "Physics-Guided Learning for Synthetic Temperature-Dependent Semiconductor Conductivity - Effects of Closed-Form Model Mismatch.tex"
```
