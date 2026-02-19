#!/usr/bin/env python
# coding: utf-8
"""
01_generate_landscapes.py
=========================
Generates synthetic landscape elevation grids for training the LE-ML inverse
model pipeline described in:

    Saturay et al. (2025). [Title]. JGR: Earth Surface. [DOI]

This script implements the forward model (Section 2, Landscapes from LEM).
For each valid LE parameter set, it evolves a numerical landscape to
topographic steady state using Landlab, and saves periodic elevation grid
snapshots as training data.

Usage
-----
    python 01_generate_landscapes.py --job-id <int> --n-landscapes <int> [--output-dir <path>]

Arguments
---------
    --job-id        Integer identifying this parallel job (0–999). Also used
                    as the random seed for parameter sampling. Equivalent to
                    `tmux_sess` in the original development code.
    --n-landscapes  Number of landscapes to generate in this job.
    --output-dir    Directory for output files (default: current directory).

Output
------
    params-{job_id}.pkl
        Pandas DataFrame with LE parameter sets and elevation time-series
        summary statistics for all landscapes in this job.

    elevts-{job_id}-{landscape_idx}-{timestep_index}.npy
        2D float64 elevation array (nY x nX) of core grid nodes at each
        saved time step. Snapshots saved every SAVE_EVERY time steps.

See DATA_PROVENANCE.md for full details on file naming, seed conventions,
and exact reproduction of individual landscapes.

Dependencies
------------
    landlab >= 2.8.0, numpy, pandas
    See environment.yml for pinned versions.
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from landlab import RasterModelGrid
from landlab.components import (
    LinearDiffuser,
    FlowAccumulator,
    DepressionFinderAndRouter,
    StreamPowerEroder,
)


# =============================================================================
# FIXED CONSTANTS
# These values were used to generate the published dataset and should not be
# changed if you intend to reproduce results from Saturay et al. (2025).
# See DATA_PROVENANCE.md for physical justifications and sources.
# =============================================================================

# --- Temporal controls ---
T_TC = 25          # Steady-state time as a multiple of tc = 1/Ks
                   # (Theodoratos et al., 2018; Eq. 2 in paper)
NUM_TS = 100       # Number of time steps per model run; dt = T_ss / NUM_TS
SAVE_EVERY = 5     # Save an elevation snapshot every N time steps
                   # (produces 20 snapshots per landscape)

# --- Spatial controls ---
LY_MULT = 5        # Grid length multiplier: Lx = LY_MULT * Ly
                   # Elongated geometry enforces linear mountain range
                   # (Section 2.1, Fig. 1 in paper)
DX = 30            # Grid cell resolution in meters

# --- Parameter space bounds (Table 1 in paper) ---
U_MIN,  U_MAX  = 1e-4, 1e0    # Uplift rate (m/yr)
KH_MIN, KH_MAX = 1e-5, 1e2    # Hillslope diffusion coefficient (m²/yr)
KS_MIN, KS_MAX = 1e-7, 1e-2   # Stream incision coefficient (1/yr)
LY_MIN, LY_MAX = 5e3,  20e3   # Grid width Ly (m)

# --- Characteristic length scale constraints (Theodoratos et al., 2018) ---
# lc = sqrt(Kh/Ks) must satisfy: MINLC_X_DX * dx <= lc <= Ly / LC_MAXLC
MINLC_X_DX = 5    # lc must be at least 5x the grid resolution
LC_MAXLC   = 5    # lc must be no more than 1/5 of Ly
                  # (ensures hillslope-valley transition is resolved by grid
                  #  but not larger than the domain; Section 2.2 in paper)

# --- Initial elevation noise ---
KZ0 = 0.001       # Noise scale: initial elevations drawn from [0, KZ0 * dx]
                  # Represents a nearly flat initial surface (Section 2.1)

# --- Mechanical constraint: Schmidt & Montgomery (1995) stability envelope ---
# (Eq. 5 in paper; used in evaluate_relief_feasibility())
COHESION      = 150e3   # Rock cohesion, c (Pa = kg m⁻¹ s⁻²)
FRICTION_ANGLE = 20     # Internal friction angle, phi (degrees)
ROCK_DENSITY  = 2660    # Rock density, rho (kg/m³)
GRAVITY       = 9.81    # Gravitational acceleration (m/s²)
MIN_RELIEF    = 100     # Minimum acceptable landscape relief (m)
MIN_GRADIENT  = 5       # Minimum acceptable landscape gradient (degrees)

# --- Empirical hc-zmax regression (internal experiment 2024-10-27a) ---
# log(zmax) = REG_M * log(hc) + REG_B, where hc = U/Ks
# Used to estimate expected maximum relief before running the full LEM,
# enabling pre-filtering of parameter sets (Section 2.2 in paper).
REG_M   = 0.9898557446591553
REG_B   = 0.8182467270202705
REG_RSQ = 0.9985214859381484   # R² of regression (for reference)

# --- Internal sampling pool size ---
N_CANDIDATE_SETS = int(1e4)  # Candidate parameter sets generated before
                              # filtering; must be >> n_landscapes


# =============================================================================
# UTILITY
# =============================================================================

def round_to_sigfigs(x, sigfigs=3):
    """
    Round x to a given number of significant figures.

    Parameters
    ----------
    x : float
        Value to round. Must be positive and non-zero.
    sigfigs : int
        Number of significant figures.

    Returns
    -------
    float
    """
    return np.round(x, -int(np.floor(np.log10(x))) + (sigfigs - 1))


# =============================================================================
# PHYSICAL CONSTRAINTS
# =============================================================================

def compute_critical_height(slope_deg,
                             cohesion=COHESION,
                             friction_angle=FRICTION_ANGLE,
                             density=ROCK_DENSITY,
                             g=GRAVITY):
    """
    Compute the critical (limiting) hillslope height after Schmidt &
    Montgomery (1995), Eq. 5 in Saturay et al. (2025).

    Parameters
    ----------
    slope_deg : float
        Hillslope gradient in degrees.
    cohesion : float
        Rock cohesion, c (Pa). Default: 150,000 Pa.
    friction_angle : float
        Internal friction angle, phi (degrees). Default: 20°.
    density : float
        Rock density (kg/m³). Default: 2,660 kg/m³.
    g : float
        Gravitational acceleration (m/s²). Default: 9.81 m/s².

    Returns
    -------
    float or None
        Critical height Hc (m), or None if slope_deg <= friction_angle
        (slope is below the angle of repose; no mechanical limit applies).

    Notes
    -----
    Unit weight: gamma = density * g  (kg m⁻² s⁻²)
    Hc = (4c / gamma) * (sin(B) * cos(phi)) / (1 - cos(B - phi))
    where B = slope_deg, phi = friction_angle.
    """
    if slope_deg <= friction_angle:
        return None

    gamma = density * g
    B   = np.radians(slope_deg)
    phi = np.radians(friction_angle)

    return (4 * cohesion / gamma) * (np.sin(B) * np.cos(phi)) / \
           (1 - np.cos(B - phi))


def estimate_max_relief(u, ks, ly, reg_m=REG_M, reg_b=REG_B):
    """
    Estimate expected maximum grid elevation and landscape gradient using
    the empirical hc-zmax regression (Eq. 19–20 in Saturay et al., 2025).

    Parameters
    ----------
    u : float
        Uplift rate (m/yr).
    ks : float
        Stream incision coefficient (1/yr).
    ly : float
        Grid width Ly (m).
    reg_m : float
        Regression slope (log-log). Default: REG_M.
    reg_b : float
        Regression intercept (log-log). Default: REG_B.

    Returns
    -------
    zmax : float
        Estimated maximum grid elevation (m).
    slope_deg : float
        Estimated landscape gradient (degrees), computed as
        arctan(zmax / (Ly/2)).
    """
    hc = u / ks                               # Characteristic height (Eq. 4)
    zmax = 10 ** (reg_m * np.log10(hc) + reg_b)
    slope_deg = np.degrees(np.arctan(zmax / (ly / 2)))
    return zmax, slope_deg


def is_relief_feasible(u, ks, ly,
                       min_relief=MIN_RELIEF,
                       min_gradient=MIN_GRADIENT):
    """
    Check whether a parameter set produces a geomorphically and mechanically
    feasible landscape relief (Section 2.2 in Saturay et al., 2025).

    A parameter set passes if:
      (1) The estimated maximum relief is below the Schmidt & Montgomery
          (1995) critical height (mechanical stability), AND
      (2) The landscape has sufficient relief and gradient to be
          geomorphically meaningful.

    Parameters
    ----------
    u : float
        Uplift rate (m/yr).
    ks : float
        Stream incision coefficient (1/yr).
    ly : float
        Grid width Ly (m).
    min_relief : float
        Minimum acceptable maximum relief (m). Default: 100 m.
    min_gradient : float
        Minimum acceptable landscape gradient (degrees). Default: 5°.

    Returns
    -------
    bool
        True if the parameter set is feasible, False otherwise.
    """
    zmax, slope_deg = estimate_max_relief(u, ks, ly)
    hc = compute_critical_height(slope_deg)

    # Case 1: slope below friction angle — no mechanical limit
    if hc is None:
        return zmax >= min_relief and slope_deg >= min_gradient

    # Case 2: slope above friction angle — check against stability envelope
    return zmax < hc and zmax >= min_relief and slope_deg >= min_gradient


# =============================================================================
# PARAMETER SAMPLING
# =============================================================================

def sample_le_parameters(n_sets=N_CANDIDATE_SETS):
    """
    Sample landscape evolution (LE) parameter sets from the defined
    parameter space (Table 1 in Saturay et al., 2025).

    Parameters are drawn randomly and independently:
      - U, Kh, Ks: uniform sampling on log scale
      - Ly: uniform sampling on linear scale
      - dx: fixed at DX (30 m)

    Parameters
    ----------
    n_sets : int
        Number of candidate parameter sets to sample. Should be much larger
        than the number of landscapes needed, to allow for constraint
        filtering. Default: N_CANDIDATE_SETS (10,000).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: u, kh, ks, dx, ly, T, hc, lc, minlc, maxlc,
        lc_in_range.
        Includes characteristic scales (Theodoratos et al., 2018):
          - T  = T_TC / ks       (characteristic time to steady state, yr)
          - hc = u / ks          (characteristic height, m; Eq. 4)
          - lc = sqrt(kh / ks)   (characteristic length, m; Eq. 3)

    Notes
    -----
    np.random.seed() must be set by the caller before invoking this function
    to ensure reproducibility. See DATA_PROVENANCE.md.
    """
    pool = n_sets * 1000   # Oversampling pool for np.random.choice

    u  = np.array([round_to_sigfigs(x) for x in
                   np.random.choice(np.logspace(np.log10(U_MIN),
                                                np.log10(U_MAX), pool),
                                    n_sets, replace=False)])
    kh = np.array([round_to_sigfigs(x) for x in
                   np.random.choice(np.logspace(np.log10(KH_MIN),
                                                np.log10(KH_MAX), pool),
                                    n_sets, replace=False)])
    ks = np.array([round_to_sigfigs(x) for x in
                   np.random.choice(np.logspace(np.log10(KS_MIN),
                                                np.log10(KS_MAX), pool),
                                    n_sets, replace=False)])
    ly = np.array([round_to_sigfigs(x) for x in
                   np.random.choice(np.linspace(LY_MIN, LY_MAX, pool),
                                    n_sets, replace=False)])
    dx = np.full(n_sets, DX)

    df = pd.DataFrame({'u': u, 'kh': kh, 'ks': ks, 'dx': dx, 'ly': ly})

    # Characteristic scales (Theodoratos et al., 2018)
    df['T']    = T_TC / df['ks']                    # Steady-state time (yr)
    df['hc']   = df['u'] / df['ks']                 # Characteristic height (m)
    df['lc']   = (df['kh'] / df['ks']) ** 0.5       # Characteristic length (m)
    df['minlc'] = MINLC_X_DX * df['dx']             # Lower bound on lc (m)
    df['maxlc'] = (df['ly'] / 2) / LC_MAXLC         # Upper bound on lc (m)

    # Characteristic length constraint (Section 2.2 in paper)
    df['lc_in_range'] = (df['lc'] >= df['minlc']) & (df['lc'] <= df['maxlc'])

    return df


def apply_constraints(df):
    """
    Filter a parameter DataFrame to retain only physically feasible sets.

    Applies two constraints (Section 2.2 in Saturay et al., 2025):
      1. Characteristic length scale constraint: lc_in_range == True
      2. Mechanical relief constraint: is_relief_feasible() == True

    Parameters
    ----------
    df : pd.DataFrame
        Output of sample_le_parameters().

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame, reset index, with added column `relief_ok`.
    """
    df['relief_ok'] = [
        is_relief_feasible(row['u'], row['ks'], row['ly'])
        for _, row in df.iterrows()
    ]
    mask = df['lc_in_range'] & df['relief_ok']
    return df[mask].reset_index(drop=True)


# =============================================================================
# GRID INITIALIZATION
# =============================================================================

def initialize_grid(ly, dx, ly_mult=LY_MULT, elev_seed=None, kz0=KZ0):
    """
    Initialize a Landlab RasterModelGrid for a linear mountain range
    (Fig. 1 in Saturay et al., 2025).

    Grid dimensions:
      - nY = int(Ly / dx) + 2  (includes 1-cell buffer rows for open BCs)
      - nX = LY_MULT * (nY-2) + 2  (elongated in x-direction)

    Boundary conditions:
      - Top and bottom edges (parallel to x-axis): open (material exits here)
      - Left and right edges (parallel to y-axis): closed

    Initial elevation field: uniform random noise in [0, kz0 * dx],
    representing a nearly flat initial surface with maximum gradient kz0.
    Open boundary cells are fixed at zero elevation.

    Parameters
    ----------
    ly : float
        Grid width in the y-direction (m).
    dx : float
        Grid cell resolution (m).
    ly_mult : int
        Grid length multiplier: Lx = ly_mult * Ly. Default: LY_MULT (5).
    elev_seed : int or None
        Random seed for initial elevation noise. See DATA_PROVENANCE.md for
        the seed convention used in the published dataset.
    kz0 : float
        Noise scale factor. Initial elevations drawn from [0, kz0 * dx].
        Default: KZ0 (0.001).

    Returns
    -------
    mg : RasterModelGrid
        Initialized Landlab grid with 'topographic__elevation' field.
    nX : int
        Number of core node columns.
    nY : int
        Number of core node rows.
    """
    nY = int(ly / dx) + 2
    nX = ly_mult * (nY - 2) + 2

    mg = RasterModelGrid((nY, nX), dx)
    mg.set_closed_boundaries_at_grid_edges(
        right_is_closed=True,
        top_is_closed=False,
        left_is_closed=True,
        bottom_is_closed=False,
    )

    zr = mg.add_zeros("topographic__elevation", at="node")
    np.random.seed(elev_seed)
    zr += np.random.rand(mg.number_of_nodes) * kz0 * dx
    zr[mg.status_at_node != 0] = 0   # Fix boundary nodes at zero

    return mg, nX - 2, nY - 2


# =============================================================================
# LANDSCAPE EVOLUTION
# =============================================================================

def evolve_landscape(mg, u, kh, ks,
                     t_tc=T_TC, num_ts=NUM_TS, save_every=SAVE_EVERY,
                     job_id=None, landscape_idx=None, output_dir=Path('.')):
    """
    Evolve a Landlab grid to topographic steady state using the LEM:

        dz/dt = U + Kh * ∇²z - Ks * A^0.5 * |∇z|    (Eq. 1 in paper)

    The landscape is evolved for T_ss = t_tc / Ks years, with time step
    dt = T_ss / num_ts. Elevation snapshots of the core nodes are saved
    every `save_every` time steps.

    Landlab components used (Table A1 in paper):
      - LinearDiffuser         → Kh * ∇²z  (hillslope diffusion)
      - FlowAccumulator (D8)   → computes drainage area A
      - DepressionFinderAndRouter → fills pits before flow routing
      - StreamPowerEroder      → Ks * A^0.5 * |∇z|  (stream incision)

    Parameters
    ----------
    mg : RasterModelGrid
        Initialized grid from initialize_grid().
    u : float
        Uplift rate (m/yr).
    kh : float
        Hillslope diffusion coefficient (m²/yr).
    ks : float
        Stream incision coefficient (1/yr).
    t_tc : float
        Steady-state time as multiple of tc = 1/Ks. Default: T_TC (25).
    num_ts : int
        Number of time steps. Default: NUM_TS (100).
    save_every : int
        Save a snapshot every N time steps. Default: SAVE_EVERY (5).
    job_id : int or None
        Job identifier used in snapshot filenames. Required if saving.
    landscape_idx : int or None
        Landscape index within job, used in snapshot filenames.
    output_dir : Path
        Directory where snapshot .npy files are written.

    Returns
    -------
    mean_elevs : np.ndarray, shape (num_ts,)
        Mean elevation of core nodes at each time step (m).
    max_elevs : np.ndarray, shape (num_ts,)
        Maximum elevation of core nodes at each time step (m).

    Notes
    -----
    Snapshot files are named: elevts-{job_id}-{landscape_idx}-{i}.npy
    where i is the zero-based time step index (0 to num_ts-1).
    Only time steps where (i+1) % save_every == 0 are saved.
    See DATA_PROVENANCE.md for full file naming convention.
    """
    zr = mg.at_node["topographic__elevation"]
    core = mg.core_nodes

    # Characteristic time and derived temporal controls (Theodoratos et al. 2018)
    tc = 1.0 / ks          # Characteristic time (yr); Eq. 2 in paper
    T_ss = t_tc * tc       # Total model time to steady state (yr)
    dt = T_ss / num_ts     # Time step size (yr)

    # Instantiate Landlab components
    diffuser = LinearDiffuser(mg, linear_diffusivity=kh,
                              method="simple", deposit=False)
    flow_acc = FlowAccumulator(mg,
                               flow_director="FlowDirectorD8",
                               depression_finder=DepressionFinderAndRouter,
                               routing='D8')
    inciser  = StreamPowerEroder(mg, K_sp=ks, m_sp=0.5, n_sp=1,
                                 threshold_sp=0.0)

    mean_elevs = np.zeros(num_ts)
    max_elevs  = np.zeros(num_ts)

    for i in range(num_ts):

        # --- Uplift ---
        zr[core] += u * dt

        # --- Hillslope diffusion (Kh * ∇²z) ---
        diffuser.run_one_step(dt)

        # --- Flow accumulation (D8 with depression routing) ---
        flow_acc.accumulate_flow()

        # --- Stream power incision (Ks * A^0.5 * |∇z|) ---
        inciser.run_one_step(dt)

        # Record summary statistics
        mean_elevs[i] = np.round(np.mean(zr[core]), 3)
        max_elevs[i]  = np.round(np.max(zr[core]), 3)

        # Save elevation snapshot
        if (i + 1) % save_every == 0:
            nY_core = mg.number_of_node_rows - 2
            nX_core = mg.number_of_node_columns - 2
            snapshot = np.round(
                zr[core].reshape(nY_core, nX_core).astype(np.float64), 2
            )
            fname = output_dir / f'elevts-{job_id}-{landscape_idx}-{i}.npy'
            np.save(fname, snapshot)
            print(f'{i + 1}', end=' ', flush=True)

    return mean_elevs, max_elevs


# =============================================================================
# MAIN
# =============================================================================

def main(job_id, n_landscapes, output_dir=Path('.')):
    """
    Run the full landscape generation pipeline for one parallel job.

    Steps:
      1. Set random seed to job_id for reproducible parameter sampling.
      2. Sample N_CANDIDATE_SETS parameter sets from the LE parameter space.
      3. Apply characteristic length and mechanical relief constraints.
      4. Take the first n_landscapes valid parameter sets.
      5. For each parameter set, initialize a grid and evolve to steady state.
      6. Save elevation snapshots and parameter file incrementally.

    Parameters
    ----------
    job_id : int
        Unique job identifier (0–999). Used as np.random.seed() for parameter
        sampling, and in file naming. See DATA_PROVENANCE.md.
    n_landscapes : int
        Number of landscapes to generate in this job.
    output_dir : Path
        Directory for all output files.

    Output files
    ------------
    params-{job_id}.pkl
        DataFrame with LE parameter sets and time-series summary statistics.
    elevts-{job_id}-{landscape_idx}-{timestep_index}.npy
        Elevation grid snapshots for each landscape.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Job ID: {job_id} | Landscapes requested: {n_landscapes}")

    # --- Step 1-3: Sample and filter parameter sets ---
    np.random.seed(job_id)   # See DATA_PROVENANCE.md
    df_candidates = sample_le_parameters(n_sets=N_CANDIDATE_SETS)
    df_valid = apply_constraints(df_candidates)

    if len(df_valid) < n_landscapes:
        raise ValueError(
            f"Only {len(df_valid)} valid parameter sets found after filtering; "
            f"{n_landscapes} requested. Increase N_CANDIDATE_SETS or widen "
            f"parameter bounds."
        )

    # --- Step 4: Select the first n_landscapes valid sets ---
    df = df_valid.iloc[:n_landscapes].copy().reset_index(drop=True)
    df['job_id'] = job_id
    df['landscape_idx'] = df.index.values
    df['nX'] = 0
    df['nY'] = 0
    df['ts_mean_elevs'] = None
    df['ts_max_elevs']  = None

    params_file = output_dir / f'params-{job_id}.pkl'
    df.to_pickle(params_file)
    print(f"Parameter sets saved: {params_file}")

    # --- Step 5-6: Evolve each landscape and save snapshots ---
    t_start_all = time.time()

    for landscape_idx in df.index:
        u, kh, ks, dx, ly = df.loc[landscape_idx, ['u', 'kh', 'ks', 'dx', 'ly']]

        print(f"\n  Landscape {landscape_idx:>3d}/{n_landscapes-1} "
              f"(U={u:.2e}, Kh={kh:.2e}, Ks={ks:.2e}, Ly={ly:.0f}m) | "
              f"Snapshots: ", end='')

        t_start = time.time()

        # Elevation seed convention: see DATA_PROVENANCE.md
        elev_seed = 100 * job_id + landscape_idx

        mg, nX, nY = initialize_grid(ly=ly, dx=dx, elev_seed=elev_seed)

        mean_elevs, max_elevs = evolve_landscape(
            mg, u=u, kh=kh, ks=ks,
            job_id=job_id,
            landscape_idx=landscape_idx,
            output_dir=output_dir,
        )

        # Update DataFrame with results
        df.at[landscape_idx, 'nX'] = int(nX)
        df.at[landscape_idx, 'nY'] = int(nY)
        df.at[landscape_idx, 'ts_mean_elevs'] = mean_elevs.astype(np.float64)
        df.at[landscape_idx, 'ts_max_elevs']  = max_elevs.astype(np.float64)

        # Save incrementally after each landscape (guards against job failure)
        df.to_pickle(params_file)

        t_elapsed = (time.time() - t_start) / 3600
        t_total   = (time.time() - t_start_all) / 3600
        t_remain  = t_total / (landscape_idx + 1) * (n_landscapes - landscape_idx - 1)
        print(f"  {t_elapsed:.2f} hrs elapsed | ETF: {t_remain:.2f} hrs")

    t_total = (time.time() - t_start_all) / 3600
    print(f"\nAll landscapes complete.")
    print(f"Total time: {t_total:.2f} hrs | "
          f"Mean per landscape: {t_total / n_landscapes:.2f} hrs")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic landscape elevation grids for the LE-ML "
            "pipeline (Saturay et al., 2025). See DATA_PROVENANCE.md for "
            "full documentation of output conventions."
        )
    )
    parser.add_argument(
        '--job-id', type=int, required=True,
        help="Unique integer job identifier (0-999). Used as the random seed "
             "for parameter sampling. Equivalent to `tmux_sess` in the "
             "original development code."
    )
    parser.add_argument(
        '--n-landscapes', type=int, required=True,
        help="Number of landscapes to generate in this job."
    )
    parser.add_argument(
        '--output-dir', type=str, default='.',
        help="Directory for output files (default: current directory)."
    )

    args = parser.parse_args()
    main(
        job_id=args.job_id,
        n_landscapes=args.n_landscapes,
        output_dir=Path(args.output_dir),
    )
