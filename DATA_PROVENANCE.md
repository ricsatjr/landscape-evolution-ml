# Data Provenance

## Overview

This document describes how the synthetic landscape dataset accompanying
Saturay et al. (2025, JGR: Earth Surface) was generated, including the
conventions used to ensure reproducibility of each individual landscape.

The dataset consists of synthetic elevation grids and their corresponding
landscape evolution (LE) parameter sets, generated using the forward model
pipeline in `01_generate_landscapes.py`. The topographic feature vectors used
for machine learning model training are extracted from these grids using
`02_extract_features.py`, which operates in two stages: rasnet extraction
(flow routing and drainage network construction) and feature computation
(raster and network feature derivation).

---

## Parallel Job Convention

Landscape generation was distributed across independent parallel jobs, each
identified by an integer **job ID** (originally referred to as `tmux_sess`
in the development version of the code, after the terminal multiplexer used
to manage parallel processes on the computing cluster).

Each job:
- Is uniquely identified by an integer `job_id` in the range [0, 999]
- Generates a fixed number of landscapes (`n_landscapes` per job)
- Produces one parameter file and a set of elevation grid snapshots

The refactored code uses `--job-id` as the CLI argument. The original
development code used `tmux_sess` for the same purpose. These are
equivalent: `job_id == tmux_sess`.

---

## Dataset Composition

The published dataset was generated across 95 independent parallel jobs in
five batches, producing 950 synthetic landscapes in total (10 landscapes per
job). All jobs were run under identical conditions; the only difference between
jobs is the random seed, which is deterministically derived from `job_id` as
described in the Random Seed Convention below.

| Batch | Job ID range    | Jobs |
|-------|-----------------|------|
| 1     | 121701–121725   | 25   |
| 2     | 122001–122025   | 25   |
| 3     | 122601–122615   | 15   |
| 4     | 123101–123120   | 20   |
| 5     | 130301–130310   | 10   |
| **Total** |             | **95 (950 landscapes)** |

---

## Random Seed Convention

Two levels of random seeds are used, both deterministically derived from
`job_id` and the landscape index `landscape_idx` (originally `dfi`):

### 1. Parameter Sampling Seed

Controls which LE parameter sets (U, Kh, Ks, Ly) are drawn for a given job.

```python
np.random.seed(job_id)
```

This seed is set once at the start of each job, before calling
`sample_le_parameters()`. All parameter sets for a given job are therefore
fully determined by `job_id` alone.

### 2. Elevation Grid Seed

Controls the initial random noise added to the elevation field of each
individual landscape before time-stepping begins.

```python
elev_seed = 100 * job_id + landscape_idx
```

where `landscape_idx` is the zero-based index of the landscape within its
job (0, 1, 2, ..., n_landscapes - 1). This formula ensures that every
landscape in the entire dataset has a unique, reproducible seed.

**Example:** The 3rd landscape (index 2) of job 7 uses `elev_seed = 702`.

### 3. Feature Extraction Noise Seed

Controls the random elevation measurement noise added to each landscape
during **Stage 1 of `02_extract_features.py`** (rasnet extraction). This
noise simulates DEM measurement uncertainty (±`elev_err` meters) applied
before flow routing and feature extraction.

```python
elev_seed = int(f"{job_id}{landscape_idx}{ts_index}") + 1
```

This is constructed by string-concatenating `job_id`, `landscape_idx`, and
`ts_index` as integers, converting to int, and adding 1.

**Example:** For job 7, landscape index 2, time step 99:
```python
elev_seed = int("7299") + 1 = 7300
```

**Why a different formula from the landscape generation seed?**

The landscape generation seed (`100 * job_id + landscape_idx`) controls the
initial surface noise that seeds the geomorphic evolution itself — it
determines the early drainage divide positions and flow routing structure.
The feature extraction seed controls noise added *after* evolution is
complete, simulating DEM observation error.

Using the same formula for both would create a correlation between a
landscape's evolutionary history and its simulated measurement noise. For
example, landscapes with similar `job_id` and `landscape_idx` values could
share measurement noise patterns that also resemble their initial conditions,
introducing a subtle form of data leakage into the feature vectors. The
string-concatenation formula ensures the feature extraction seeds are
numerically distinct from — and uncorrelated with — the landscape generation
seeds across the full dataset.

---

## File Naming Convention

### Parameter File

One parameter file is saved per job, containing the LE parameter sets and
time-series summary statistics for all landscapes in that job:

```
params-{job_id}.pkl
```

This is a pandas DataFrame saved as a pickle file. Columns include:
`u`, `kh`, `ks`, `dx`, `ly`, `T`, `hc`, `lc`, `nX`, `nY`,
`ts_mean_elevs`, `ts_max_elevs`, and the job/index identifiers
`job_id`, `landscape_idx`.

### Elevation Grid Snapshots

For each landscape, elevation grids are saved at every 5th time step
(i.e., at time steps 4, 9, 14, ..., 99; zero-indexed), producing
20 snapshots per landscape out of 100 total time steps:

```
elevts-{job_id}-{landscape_idx}-{timestep_index}.npy
```

where `timestep_index` is the zero-based time step index (0–99).
Only indices that are multiples of 5, minus 1 (i.e., 4, 9, 14, ... 99)
are saved, corresponding to `(i+1) % 5 == 0` in the loop.

Each `.npy` file contains a 2D float64 array of shape `(nY, nX)`,
representing the elevation (in meters, rounded to 2 decimal places)
of the core nodes of the grid at that time step.

**Example:** `elevts-7-2-99.npy` is the final (steady-state) snapshot
of the 3rd landscape (index 2) of job 7.

### Rasnet Intermediate Files

Stage 1 of `02_extract_features.py` produces intermediate rasnet files
containing the processed Landlab grid, watershed mask, and NetworkX drainage
network for each landscape. These are saved before feature computation
because flow routing and network construction are computationally expensive.
Saving the intermediate objects allows Stage 2 (feature computation) to be
rerun independently — for example, when experimenting with alternative
feature definitions — without repeating the flow routing.

```
rasnet-n{elev_err}-{job_id}-{landscape_idx}-{ts_index}.pkl
```

where `elev_err` is the elevation noise magnitude in meters (integer).

**Example:** `rasnet-n10-7-2-99.pkl` is the rasnet file for the 3rd
landscape of job 7, with 10 m elevation noise, at steady-state (ts=99).

> **Note on published dataset naming:** The rasnet files in the published
> dataset (Zenodo archive) were generated before varying noise levels was
> considered, and do not include the `n{elev_err}` prefix. Their naming
> convention is:
> ```
> rasnet-{job_id}-{landscape_idx}-{ts_index}.pkl
> ```
> All published rasnet files used `elev_err = 10 m`.
> The refactored `02_extract_features.py` uses the `n{elev_err}` prefix for
> newly generated files to support future experiments with different noise
> levels. If you are loading the published dataset with the refactored code,
> you will need to adjust the glob pattern in `run_stage2_features()` from
> `rasnet-n*-{job_id}-*-*.pkl` to `rasnet-{job_id}-*-*.pkl`.

Each rasnet `.pkl` file contains a list:
```python
[le_params, mg, mask, chNet, wsOutlets, wsOutletsDA]
```

| Element      | Type                  | Description                                          |
|--------------|-----------------------|------------------------------------------------------|
| `le_params`  | dict                  | LE parameter labels (u, kh, ks, ly) and identifiers |
| `mg`         | RasterModelGrid       | Landlab grid with elevation and drainage fields      |
| `mask`       | np.ndarray (bool)     | Valid node mask, excluding boundary-connected cells  |
| `chNet`      | nx.DiGraph            | Drainage network with modified Strahler orders       |
| `wsOutlets`  | np.ndarray (int)      | Node IDs of valid watershed outlets                  |
| `wsOutletsDA`| list of int           | Drainage areas (cells) of valid watershed outlets    |

### Feature DataFrame Files

Stage 2 of `02_extract_features.py` produces the final feature DataFrames
used for ML model training:

```
features-{job_id}.pkl
```

Each `.pkl` file is a pandas DataFrame with one row per landscape and
columns for LE parameter labels, derived ratio labels (`u_ks`, `kh_ks`),
and the 39 topographic features described in Tables 2–3 of the paper.

> **Note on log transformation:** `u_ks` and `kh_ks` are stored as raw
> (untransformed) ratios U/Ks and Kh/Ks respectively. The log₁₀ transform
> is applied at training time inside `split_features_labels()` in
> `pipeline_utils.py`, keeping the DataFrame self-describing.

---

## Exact Reproduction of a Specific Landscape

To exactly reproduce landscape `landscape_idx` from job `job_id`:

```python
import numpy as np

# All functions below are defined in 01_generate_landscapes.py.
# This is illustrative pseudocode; run 01_generate_landscapes.py directly
# to reproduce the full dataset.

# 1. Reproduce the parameter set
np.random.seed(job_id)
df = sample_le_parameters(...)
df = apply_constraints(df)
params = df.iloc[landscape_idx]

# 2. Reproduce the elevation grid with the correct seed
elev_seed = 100 * job_id + landscape_idx
mg = initialize_grid(ly=params['ly'], dx=params['dx'], elev_seed=elev_seed)

# 3. Evolve the landscape
evolve_landscape(mg, u=params['u'], kh=params['kh'], ks=params['ks'])
```

See `01_generate_landscapes.py` for the full working implementation.

To reproduce the feature extraction noise applied to the same landscape:

```python
# Elevation noise seed for feature extraction (Stage 1 of 02_extract_features.py)
ts_index = 99  # steady-state snapshot
elev_seed_features = int(f"{job_id}{landscape_idx}{ts_index}") + 1

# Load and add noise (matches 02_extract_features.py exactly)
np.random.seed(elev_seed_features)
noise = (np.random.rand(mg.number_of_nodes) - 0.5) * elev_err * 2
mg.at_node['topographic__elevation'] += noise
```

See Seed Convention 3 above for the full rationale behind using a separate
formula for the feature extraction seed.

---

## Physical Parameters and Fixed Constants

The following constants were fixed across all landscape generation runs.
They are defined in `01_generate_landscapes.py` and documented here for
reference.

| Constant       | Value                  | Description                                         |
|----------------|------------------------|-----------------------------------------------------|
| `T_TC`         | 25                     | Steady-state time as multiple of tc = 1/Ks          |
| `NUM_TS`       | 100                    | Number of time steps per model run                  |
| `SAVE_EVERY`   | 5                      | Snapshot saved every N time steps                   |
| `LY_MULT`      | 5                      | Grid length multiplier: Lx = LY_MULT * Ly           |
| `DX`           | 30 m                   | Grid cell resolution                                |
| `LY_MIN`       | 5,000 m                | Minimum grid width                                  |
| `LY_MAX`       | 20,000 m               | Maximum grid width                                  |
| `MINLC_X_DX`   | 5                      | Minimum lc as multiple of dx (lc >= 5*dx)           |
| `LC_MAXLC`     | 5                      | lc <= (Ly/2) / LC_MAXLC = Ly/10; Ly/2 is the divide-to-outlet distance |
| `KZ0`          | 0.001                  | Initial noise scale factor: noise in [0, KZ0*dx]   |
| `COHESION`     | 150,000 Pa             | Rock cohesion for Schmidt & Montgomery (1995) limit |
| `FRICTION_ANGLE` | 20 degrees           | Friction angle for stability envelope               |
| `ROCK_DENSITY` | 2,660 kg/m³            | Rock density for stability envelope                 |
| `REG_SLOPE_M`  | 0.9898557446591553     | Slope of log(hc) vs log(zmax) regression            |
| `REG_INTERCEPT_B` | 0.8182467270202705  | Intercept of log(hc) vs log(zmax) regression        |
| `REG_RSQ`      | 0.9985214859381484     | R² of the hc-zmax regression                        |

The regression coefficients `REG_SLOPE_M` and `REG_INTERCEPT_B` were
derived from a preliminary experiment (internal reference: experiment
2024-10-27a) establishing the empirical linear relationship between
log(characteristic height, hc = U/Ks) and log(maximum grid elevation,
zmax) across a sample of evolved landscapes. These coefficients are used
in the mechanical constraint filter (`evaluate_relief_feasibility()`).

---

## Software Environment

See `environment.yml` for the exact software versions used during dataset
generation. Key dependencies:

| Package      | Version  |
|--------------|----------|
| python       | 3.12.2   |
| landlab      | 2.9.2    |
| numpy        | 2.0.1    |
| pandas       | 2.2.3    |
| scikit-learn | 1.6.1    |
| scipy        | 1.15.1   |
| networkx     | 3.4.2    |
| matplotlib   | 3.10.0   |
| richdem      | 2.3.0    |
| kneed        | 0.8.5    |
| statsmodels  | 0.14.4   |

---

## Citation

If you use this dataset, please cite:

> Saturay, R.L., Ramos, N.T., & Bantang, J.Y. (2025). [Title]. 
> Journal of Geophysical Research: Earth Surface. [DOI]

And the underlying modeling framework:

> Barnhart, K.R., et al. (2020). Landlab v2.0: A software package for 
> Earth surface dynamics. Earth Surface Dynamics, 8(2), 379–397.
