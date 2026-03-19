# landscape-evolution-ml

**Numerical modeling and machine learning pipeline for parameter estimation, hypothesis testing, and discovery in landscape evolution**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Landlab 2.9.2](https://img.shields.io/badge/landlab-2.9.2-green.svg)](https://landlab.readthedocs.io/)
[![scikit-learn 1.6.1](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![networkx 3.4.2](https://img.shields.io/badge/networkx-3.4.2-red.svg)](https://networkx.org/)

---

## Overview

This repository contains the code accompanying:

> **Saturay, R.M., Ramos, N.T., & Bantang, J.Y.** (2025). *[Title]*. 
> Journal of Geophysical Research: Earth Surface. [DOI — to be added upon publication]

Inferring landscape evolution parameters from topography is a fundamental inverse problem in geomorphology, made difficult by the scarcity of field sites with well-constrained process rates and the non-uniqueness of inverse problem solutions. 

This repository implements a **landscape evolution – machine learning (LE-ML) pipeline** that addresses this by:

1. **Generating synthetic training landscapes** using a numerical landscape evolution model (LEM) implemented in [Landlab](https://landlab.readthedocs.io/), with parameter sets constrained by physically meaningful bounds
2. **Extracting topographic features** — 28 raster-based and 11 drainage network features — from the synthetic elevation grids
3. **Training machine learning models** using nested cross-validation across eight scikit-learn algorithms to estimate LEM parameters from those features
4. **Testing two key hypotheses** about what topography can and cannot tell us about landscape processes

### Key Findings

- **Parameter ratios, not individual parameters, are recoverable from topography.** All eight ML algorithms achieve R² > 0.9 for the parameter ratios U/K_s and Kh/K_s, while individual parameters (U, Kh, Ks) remain poorly constrained (R² < 0.2). This demonstrates that landscapes reflect the *relative rates* of competing geomorphic processes rather than their absolute magnitudes.

- **Zero-order drainage structures are powerful, underexplored indicators of landscape dynamics.** Bifurcation and length ratios of drainages nearest to hilltops emerge as among the most important estimators of Kh/K_s — features overlooked in conventional drainage analysis.

---

## Pipeline Structure

The pipeline consists of four scripts, each corresponding to a section of the paper:

| Script | Paper Section | Description |
|--------|--------------|-------------|
| `pipeline/01_generate_landscapes.py` | Sec. 2: Landscapes from LEM | Forward model: sample LE parameters, evolve landscapes to steady state, save elevation grids |
| `pipeline/02_extract_features.py` | Sec. 3.1: Feature generation | Extract 39 raster- and network-based topographic features from elevation grids |
| `pipeline/03_train_models.py` | Sec. 3.2–3.3: ML model development | Train eight ML algorithms with nested CV and hyperparameter tuning |
| `pipeline/04_test_hypotheses.py` | Sec. 3.4–3.5: Results & Synthesis | Compare individual vs ratio estimation; permutation-based feature importance analysis |

---

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/[username]/landscape-evolution-ml.git
cd landscape-evolution-ml
conda env create -f environment.yml
conda activate leml
```

### 2. Run the minimal working example

The quickstart notebook walks through the full pipeline end-to-end on a small sample dataset (5 pre-generated landscapes included in `data/sample/`). It demonstrates both key findings without requiring you to generate new training data.

```bash
jupyter notebook notebooks/00_quickstart.ipynb
```

### 3. Reproduce the paper's results

The full trained models are provided in `trained_models/`. To apply the best-performing model (MLP) to your own DEM:

```python
import pickle
import numpy as np

# Load trained model
with open('trained_models/mlp_ratios.pkl', 'rb') as f:
    model = pickle.load(f)

# Extract features from your DEM (see pipeline/02_extract_features.py)
# features shape: (1, 39)
log_U_Ks, log_Kh_Ks = model.predict(features)[0]
```

To reproduce individual figures from the paper, see the figure notebooks in `notebooks/`.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_quickstart.ipynb` | Full pipeline demo on 5 sample landscapes — start here |
| `01_parameter_space.ipynb` | Reproduces Fig. 3: constrained LE parameter space |
| `02_sample_landscapes.ipynb` | Reproduces Fig. 4: sample landscape elevation maps |
| `03_model_performance.ipynb` | Reproduces Figs. 6–9: individual vs ratio estimation performance |
| `04_feature_importance.ipynb` | Reproduces Figs. 10–12: feature clustering and zero-order drainage finding |

---

## Dataset

The full synthetic landscape dataset used to train the models in the paper is archived at:

> **[Zenodo DOI — to be added upon publication]**

It contains elevation grid snapshots and parameter files for all generated landscapes. See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) for complete documentation of the dataset structure, file naming conventions, and seed conventions needed to reproduce individual landscapes.

A small sample dataset (5 landscapes) is included in `data/sample/` for running the quickstart notebook without downloading the full archive.

---

## Generating New Training Data

To generate a new batch of synthetic landscapes (e.g., to extend the training set or explore a different parameter space):

```bash
python pipeline/01_generate_landscapes.py \
    --job-id 0 \
    --n-landscapes 10 \
    --output-dir data/my_landscapes/
```

Multiple jobs can be run in parallel with different `--job-id` values (0–999). Each `job-id` acts as the random seed for parameter sampling, ensuring reproducibility. See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) for full details.

> **Note:** Generating the full training dataset used in the paper required substantial compute resources (University of the Philippines Data Commons HPC cluster). Generating 10 landscapes on a standard laptop takes approximately [X] hours. We recommend downloading the published dataset from Zenodo for reproducing paper results.

---

## Applying the Model to Real Topography

The trained models estimate log(U/Ks) and log(Kh/Ks) for any landscape whose evolution is reasonably governed by the stream power + linear diffusion LEM:

∂z/∂t = U + Kh∇²z − Ks · A^0.5 · |∇z|

**Before applying the model**, ensure your landscape meets these assumptions:
- Spatially uniform material properties
- Approximately uniform climatic conditions
- Topographic steady state (or near-steady state)
- Similar tectonic setting to the training data (convergent mountain range)

The model estimates **parameter ratios**, not individual parameters. To recover individual values of U, Kh, or Ks, at least one parameter must be independently constrained (e.g., from cosmogenic nuclide analysis or thermochronology).

See `notebooks/00_quickstart.ipynb` for a worked example of the full feature extraction and estimation workflow.

---

## Landscape Evolution Model

The forward model simulates the growth of a linear mountain range under tectonic uplift, hillslope diffusion, and stream incision:

∂z/∂t = U + Kh∇²z − Ks · A^0.5 · |∇z|

Implemented using [Landlab v2.9.2](https://landlab.readthedocs.io/) components:

| Process | Landlab Component |
|---------|------------------|
| Hillslope diffusion (Kh∇²z) | `LinearDiffuser` |
| Flow accumulation | `FlowAccumulator` (D8) + `DepressionFinderAndRouter` |
| Stream power incision (Ks · A^0.5 · \|∇z\|) | `StreamPowerEroder` |

Parameter ranges and physical constraints follow Table 1 and Section 2.2 of the paper. See `pipeline/01_generate_landscapes.py` and `DATA_PROVENANCE.md` for complete documentation.

---

## Repository Structure

```
landscape-evolution-ml/
│
├── README.md
├── DATA_PROVENANCE.md          # Dataset structure, seed conventions, file naming
├── environment.yml             # Pinned conda environment
├── LICENSE
│
├── pipeline/
│   ├── 01_generate_landscapes.py
│   ├── 02_extract_features.py
│   ├── 03_train_models.py
│   └── 04_test_hypotheses.py
│
├── notebooks/
│   ├── 00_quickstart.ipynb
│   ├── 01_parameter_space.ipynb
│   ├── 02_sample_landscapes.ipynb
│   ├── 03_model_performance.ipynb
│   └── 04_feature_importance.ipynb
│
├── data/
│   ├── sample/                 # 5-landscape sample dataset
│   │   ├── params-sample.pkl
│   │   └── elevts-sample-*.npy
│   └── README.md               # Link to full dataset on Zenodo
│
├── trained_models/
│   ├── mlp_ratios.pkl          # Best model (MLP) for log(U/Ks) and log(Kh/Ks)
│   └── README.md               # Notes on model versions and training data
│
└── tests/
    └── test_smoke.py           # Smoke test: pipeline runs on sample data
```

---

## Citation

If you use this code or the accompanying dataset, please cite:

```bibtex
@article{saturay2025,
  author  = {Saturay, Ricarido M. and Ramos, Noelynna T. and Bantang, Johnrob Y.},
  title   = {[Title]},
  journal = {Journal of Geophysical Research: Earth Surface},
  year    = {2025},
  doi     = {[DOI — to be added upon publication]}
}
```

If you use the dataset specifically, please also cite the Zenodo archive:

```bibtex
@dataset{saturay2025data,
  author    = {Saturay, Ricarido M. and Ramos, Noelynna T. and Bantang, Johnrob Y.},
  title     = {Synthetic landscape evolution dataset for LE-ML pipeline},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {[DOI — to be added upon publication]}
}
```

---

## Acknowledgements

Funding support was provided by the Department of Science and Technology (Philippines) – Human Resource Development Program, and the University of the Philippines Diliman Office of the Chancellor, through the Office of the Vice Chancellor for Research and Development (Open Grant 242402). Computing facilities were provided by the University of the Philippines Data Commons.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
