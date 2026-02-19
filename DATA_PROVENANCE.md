# Data Provenance

## Overview

This document describes how the synthetic landscape dataset accompanying
Saturay et al. (2025, JGR: Earth Surface) was generated, including the
conventions used to ensure reproducibility of each individual landscape.

The dataset consists of synthetic elevation grids and their corresponding
landscape evolution (LE) parameter sets, generated using the forward model
pipeline in `01_generate_landscapes.py`.

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

---

## Exact Reproduction of a Specific Landscape

To exactly reproduce landscape `landscape_idx` from job `job_id`:

```python
import numpy as np
from leml.params import sample_le_parameters, apply_constraints
from leml.forward import initialize_grid, evolve_landscape

# 1. Reproduce the parameter set
np.random.seed(job_id)
df = sample_le_parameters(...)
df = apply_constraints(df, ...)
params = df.iloc[landscape_idx]

# 2. Reproduce the elevation grid with the correct seed
elev_seed = 100 * job_id + landscape_idx
mg = initialize_grid(ly=params['ly'], dx=params['dx'], elev_seed=elev_seed)

# 3. Evolve the landscape
evolve_landscape(mg, u=params['u'], kh=params['kh'], ks=params['ks'])
```

See `01_generate_landscapes.py` for the full working implementation.

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
| `LC_MAXLC`     | 5                      | Maximum lc as fraction of Ly (lc <= Ly/10)          |
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

| Package     | Version |
|-------------|---------|
| landlab     | 2.8.0   |
| numpy       | 2.0.1   |
| pandas      | 2.2.2   |
| scikit-learn| 1.5.1   |
| scipy       | 1.15.1  |
| networkx    | 3.4.2   |
| python      | 3.11    |

---

## Citation

If you use this dataset, please cite:

> Saturay, R.L., Ramos, N.T., & Bantang, J.Y. (2025). [Title]. 
> Journal of Geophysical Research: Earth Surface. [DOI]

And the underlying modeling framework:

> Barnhart, K.R., et al. (2020). Landlab v2.0: A software package for 
> Earth surface dynamics. Earth Surface Dynamics, 8(2), 379–397.
