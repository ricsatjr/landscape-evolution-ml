"""
01c_compute_erosion_rates.py
============================
Compute instantaneous bulk erosion rates and total erosion from landscape
evolution simulation params pkl files.

The erosion rate at each time step is derived from the spatially averaged
mean elevation time series (ts_mean_elevs) stored in the params pkl files
produced by 01_generate_landscapes.py. Using the landscape-averaged form of
the governing LEM equation:

    d<z>/dt = U - E

where U is the (constant) uplift rate and E is the bulk erosion rate
aggregating both hillslope diffusion flux and stream incision. Rearranging:

    E = U - d<z>/dt

The time derivative is approximated by a forward finite difference:

    E_i = U - (<z>_{i+1} - <z>_i) / dt     i = 0, 1, ..., 98

where dt = T / NUM_TS is the simulation time step size (yr).

The result is stored as a 100-element array (ts_erosion_rates) where the
array position corresponds directly to ts_index:

    ts_erosion_rates[0]    = NaN    (no prior state; forward diff undefined)
    ts_erosion_rates[k]    = E_{k-1}    k = 1, ..., 99

This matches the ts_index convention in features-*.pkl: to look up the
erosion rate for a given row, use row['ts_erosion_rates'][ts_index].
The full 100-element array also supports smooth erosion rate curve plotting.

Cumulative erosion (m) is the running discrete integral of the erosion rate:

    ts_cumulative_erosion[k] = sum(ts_erosion_rates[1:k+1]) * dt
                             = sum(E_i for i=0..k-1) * dt

Array position k holds the total elevation removed by erosion from t=0 to
t_k. Index 0 is NaN, consistent with ts_erosion_rates. The final value
ts_cumulative_erosion[99] is the total erosion over the full simulation.

Output is a wide-format DataFrame (one row per landscape) saved as a pkl
with columns: job_id, landscape_idx, ts_erosion_rates, ts_cumulative_erosion.

Usage
-----
Single job:
    python 01c_compute_erosion_rates.py \\
        --params-dir outputs/ \\
        --output-dir outputs/erosion_rates/ \\
        --job-id 121701

All jobs:
    python 01c_compute_erosion_rates.py \\
        --params-dir outputs/ \\
        --output-dir outputs/erosion_rates/ \\
        --job-id all

Notes
-----
- Handles legacy params pkl column names: 'df-ind' -> 'landscape_idx';
  job_id is always taken from the filename, not the dataframe.
- Output filename: erosion_rates-{job_id}-{git_hash}.pkl
                or erosion_rates-all-{git_hash}.pkl
- To query erosion rate for a features-*.pkl row:
      er  = row['ts_erosion_rates'][ts_index]
      cum = row['ts_cumulative_erosion'][ts_index]
- To plot the smooth erosion rate curve for one landscape:
      plt.plot(range(1, 100), row['ts_erosion_rates'][1:])
- To plot cumulative erosion:
      plt.plot(range(1, 100), row['ts_cumulative_erosion'][1:])

Dependencies
------------
    numpy, pandas
    pipeline_utils (_git_hash)
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import _git_hash

# =============================================================================
# CONSTANTS
# =============================================================================

NUM_TS = 100    # Number of time steps per simulation run; must match
                # NUM_TS in 01_generate_landscapes.py

# Legacy params pkl column name mapping.
# Older params pkl files (pre-refactor) used different column names.
LEGACY_COLUMN_MAP = {
    'df-ind'   : 'landscape_idx',   # landscape index within job
    'tmux-sess': 'job_id',          # job identifier (overridden by filename)
}


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def compute_erosion_rates(ts_mean_elevs, u, T):
    """
    Compute the instantaneous bulk erosion rate array for one landscape.

    Uses a forward finite difference on the mean elevation time series:

        E_i = U - (<z>_{i+1} - <z>_i) / dt    i = 0, ..., NUM_TS-2

    Result is a 100-element array where array position equals ts_index:

        ts_erosion_rates[0]   = NaN
        ts_erosion_rates[k]   = E_{k-1}    k = 1, ..., 99

    Total erosion is the running discrete integral of the erosion rate:

        ts_cumulative_erosion[k] = sum(ts_erosion_rates[1:k+1]) * dt

    Index 0 is NaN. ts_cumulative_erosion[99] is total erosion over the
    full simulation.

    Parameters
    ----------
    ts_mean_elevs : array-like, shape (NUM_TS,)
        Mean core-node elevation at each time step (m).
    u : float
        Uplift rate (m/yr).
    T : float
        Total simulation time (yr). dt = T / NUM_TS.

    Returns
    -------
    ts_erosion_rates : np.ndarray, shape (NUM_TS,)
        Bulk erosion rate (m/yr) at each ts_index.
        Index 0 is NaN; indices 1-99 are computed values.
    ts_cumulative_erosion : np.ndarray, shape (NUM_TS,)
        Cumulative erosion (m) at each ts_index.
        Index 0 is NaN; ts_cumulative_erosion[99] is total erosion.
    """
    z = np.asarray(ts_mean_elevs, dtype=float)

    if len(z) != NUM_TS:
        raise ValueError(
            f"ts_mean_elevs has {len(z)} values; expected {NUM_TS}."
        )

    dt = T / NUM_TS

    # Forward differences: E[i] = u - (z[i+1] - z[i]) / dt, i = 0..NUM_TS-2
    dz = np.diff(z)       # shape (NUM_TS-1,)
    E  = u - dz / dt      # shape (NUM_TS-1,); E[i] assigned to ts_index i+1

    # Build 100-element array aligned to ts_index
    ts_erosion_rates     = np.empty(NUM_TS, dtype=float)
    ts_erosion_rates[0]  = np.nan
    ts_erosion_rates[1:] = E

    # Cumulative erosion: running discrete integral of E
    # np.nancumsum treats NaN at index 0 as 0, giving correct running sum;
    # index 0 is then reset to NaN for consistency.
    ts_cumulative_erosion    = np.nancumsum(ts_erosion_rates) * dt
    ts_cumulative_erosion[0] = np.nan

    return ts_erosion_rates, ts_cumulative_erosion


# =============================================================================
# PIPELINE FUNCTION
# =============================================================================

def compute_erosion_rates_from_params(params_dir, job_id='all'):
    """
    Read params pkl files and compute erosion rates for all landscapes.

    Parameters
    ----------
    params_dir : Path
        Directory containing params-*.pkl files.
    job_id : int or 'all'
        If int, process only params-{job_id}.pkl.
        If 'all', process all params-*.pkl files in params_dir.

    Returns
    -------
    df_out : pd.DataFrame
        Wide-format DataFrame with one row per landscape and columns:
        job_id, landscape_idx, ts_erosion_rates, total_erosion
    n_skipped : int
        Number of landscapes skipped due to missing ts_mean_elevs.
    """
    params_dir = Path(params_dir)

    if str(job_id) == 'all':
        pkl_files = sorted(params_dir.glob('params-*.pkl'))
    else:
        pkl_files = sorted(params_dir.glob(f'params-{int(job_id)}.pkl'))

    if not pkl_files:
        raise FileNotFoundError(
            f"No params pkl files found in {params_dir} "
            f"for job_id={job_id}."
        )

    print(f"Found {len(pkl_files)} params file(s) in {params_dir}")

    rows      = []
    n_skipped = 0
    t_start   = time.time()

    for pkl_path in pkl_files:

        # --- Extract job_id from filename (authoritative; ignores tmux-sess) ---
        file_job_id = int(pkl_path.stem.split('-')[1])

        df_params = pd.read_pickle(pkl_path)

        # --- Legacy column mapping ---
        # Remap old column names to current conventions where present.
        # job_id is always derived from the filename, not the dataframe.
        rename = {
            old: new
            for old, new in LEGACY_COLUMN_MAP.items()
            if old in df_params.columns and new not in df_params.columns
        }
        if rename:
            df_params = df_params.rename(columns=rename)

        # Validate required columns after remapping
        required     = {'landscape_idx', 'u', 'T', 'ts_mean_elevs'}
        missing_cols = required - set(df_params.columns)
        if missing_cols:
            raise KeyError(
                f"{pkl_path.name}: missing required column(s) "
                f"{sorted(missing_cols)} after legacy remapping. "
                f"Columns present: {df_params.columns.tolist()}"
            )

        # --- Guard: skip rows missing ts_mean_elevs ---
        missing_data = df_params['ts_mean_elevs'].isna()
        if missing_data.any():
            warnings.warn(
                f"{pkl_path.name}: {missing_data.sum()} landscape(s) have "
                f"missing ts_mean_elevs and will be skipped."
            )
            df_params  = df_params[~missing_data]
            n_skipped += int(missing_data.sum())

        for _, row in df_params.iterrows():

            landscape_idx = int(row['landscape_idx'])
            u             = float(row['u'])
            T             = float(row['T'])
            ts_mean_elevs = row['ts_mean_elevs']

            ts_erosion_rates, ts_cumulative_erosion = compute_erosion_rates(
                ts_mean_elevs, u, T
            )

            rows.append({
                'job_id'               : file_job_id,
                'landscape_idx'        : landscape_idx,
                'ts_erosion_rates'     : ts_erosion_rates,
                'ts_cumulative_erosion': ts_cumulative_erosion,
            })

        print(f"  {pkl_path.name}: {len(df_params)} landscape(s) processed.")

    df_out = pd.DataFrame(rows, columns=[
        'job_id', 'landscape_idx', 'ts_erosion_rates', 'ts_cumulative_erosion'
    ])

    elapsed = time.time() - t_start
    print(f"\nDone. {len(df_out)} landscape(s). Elapsed: {elapsed:.1f}s")

    return df_out, n_skipped


# =============================================================================
# MAIN
# =============================================================================

def main(params_dir, output_dir, job_id):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    git_hash = _git_hash()

    df_out, n_skipped = compute_erosion_rates_from_params(
        params_dir=params_dir,
        job_id=job_id,
    )

    if str(job_id) == 'all':
        out_file = output_dir / f'erosion_rates-all-{git_hash}.pkl'
    else:
        out_file = output_dir / f'erosion_rates-{int(job_id)}-{git_hash}.pkl'

    df_out.to_pickle(out_file)
    print(f"Saved: {out_file}")

    if n_skipped > 0:
        print(f"Warning: {n_skipped} landscape(s) skipped "
              f"(missing ts_mean_elevs).")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            "Compute instantaneous bulk erosion rates and total erosion "
            "from landscape evolution params pkl files. Output is a "
            "wide-format DataFrame (one row per landscape) with columns: "
            "job_id, landscape_idx, ts_erosion_rates (array, len=100), "
            "ts_cumulative_erosion (array, len=100). Saved as "
            "erosion_rates-{job_id}-{git_hash}.pkl. "
            "To query erosion rate for a features-*.pkl row: "
            "er = row['ts_erosion_rates'][ts_index]."
        )
    )
    parser.add_argument(
        '--params-dir', type=str, required=True,
        help="Directory containing params-*.pkl files produced by "
             "01_generate_landscapes.py."
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help="Directory for output erosion_rates pkl file."
    )
    parser.add_argument(
        '--job-id', type=str, required=True,
        help="Job identifier: integer to process a single params pkl, "
             "or 'all' to process all params pkl files in --params-dir."
    )

    args = parser.parse_args()

    job_id = args.job_id if args.job_id == 'all' else int(args.job_id)

    main(
        params_dir = args.params_dir,
        output_dir = args.output_dir,
        job_id     = job_id,
    )
