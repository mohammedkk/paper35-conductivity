# Paper 32 — Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials

This repository contains the manuscript, code, and results for:

> **Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials**
> Mohammed Hamzah Khudhur

## Contents

- [`Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials.pdf`](Physics-Informed%20Deep%20Learning%20for%20Prediction%20of%20Electrical%20Conductivity%20in%20Solid-State%20Materials.pdf) / [`.tex`](Physics-Informed%20Deep%20Learning%20for%20Prediction%20of%20Electrical%20Conductivity%20in%20Solid-State%20Materials.tex) — typeset manuscript (LaTeX, compiles with `pdflatex` + `bibtex`).
- [`Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials.md`](Physics-Informed%20Deep%20Learning%20for%20Prediction%20of%20Electrical%20Conductivity%20in%20Solid-State%20Materials.md) — Markdown rendition of the same content, for reading directly on GitHub.
- [`Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials.docx`](Physics-Informed%20Deep%20Learning%20for%20Prediction%20of%20Electrical%20Conductivity%20in%20Solid-State%20Materials.docx) — Word rendition (generated from the `.md` file via `pandoc`; see below to rebuild).
- [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) — AI-tool-use statement, kept as a standalone repository file rather than inside the manuscript body, at the author's request.
- [`references.bib`](references.bib) — bibliography (18 references, each independently verified against publisher/preprint records; 8 of them from the last 5 years, 2023–2025).
- [`code/experiment.py`](code/experiment.py) — self-contained NumPy implementation: synthetic Arrhenius-law benchmark generator, hand-written MLP with manual backpropagation and Adam/AdamW, and the three compared models (black-box, soft physics-informed, physics-embedded). Produces `results/results.json` and `results/arrays.npz`.
- [`code/figures.py`](code/figures.py) — generates all manuscript figures from the saved results.
- [`figures/`](figures) — the five figures used in the manuscript (PNG).
- [`results/`](results) — `results.json` (all reported metrics) and `arrays.npz` (raw arrays used for plotting).

## Reproducing the results

```bash
pip install numpy matplotlib scikit-learn
python3 code/experiment.py   # trains all three models, writes results/
python3 code/figures.py      # regenerates figures/ from results/
```

Everything is generated from a fixed random seed (`42`); see the manuscript's *Data and Code Availability* and *Limitations* sections for a note on residual floating-point run-to-run variation. No experimental or proprietary data are used anywhere in this repository — the benchmark is a fully specified, physically grounded synthetic generative model (Section 4 of the paper).

## Rebuilding the PDF

```bash
pdflatex "Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials.tex" && bibtex "Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials" && pdflatex "Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials.tex" && pdflatex "Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials.tex"
```

Requires a LaTeX distribution with `natbib`, `booktabs`, and `multirow` (e.g. `texlive-latex-extra`, `texlive-bibtex-extra`, `texlive-science`).

## Rebuilding the Word document

```bash
pandoc "Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials.md" -o "Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials.docx" --resource-path=.:figures -s
```

Requires `pandoc`. Converting from the `.md` file (plain text, real embedded figures) rather than the `.tex` source avoids inline-equation-to-OOXML conversion issues; the LaTeX source remains the authoritative typeset version.

## Summary of findings

Three deep learning models — a black-box MLP, a soft physics-informed MLP (Arrhenius-residual loss), and a physics-embedded MLP (Arrhenius law as the output decoder) — are compared on a synthetic, physically grounded conductivity-prediction benchmark across three generalization conditions:

| Condition | Black-box | Soft PINN | Physics-embedded |
|---|---|---|---|
| In-distribution (R²) | 0.816 | 0.822 | 0.802 |
| Temperature extrapolation (R²) | 0.703 | 0.714 | **0.736** |
| Material-family extrapolation (R²) | -1.419 | -1.443 | -1.483 |

Embedding the Arrhenius law improves generalization specifically along the temperature axis (the variable that law governs) but confers no benefit — and a small cost — for extrapolation to an unseen structural family, a distinct generalization axis the embedded law does not constrain. Full discussion in `Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials.md` / `Physics-Informed Deep Learning for Prediction of Electrical Conductivity in Solid-State Materials.pdf`.
