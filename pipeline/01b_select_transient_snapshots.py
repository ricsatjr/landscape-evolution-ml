#!/usr/bin/env python
# coding: utf-8
"""
01b_select_transient_snapshots.py
==================================
Selects one transient-stage snapshot per landscape from the elevation
time-series summary statistics saved by 01_generate_landscapes.py.

A snapshot at time step index i is eligible if:
  (1) It corresponds to a saved .npy file (i.e., (i+1) % SAVE_EVERY == 0)
  (2) It is not the final time step (explicitly excluded)
  (3) ts_mean_elevs[i] >= MIN_ELEV_FRAC * ts_mean_elevs[-1]
      (landscape is not too young / topography not too subdued)
  (4) abs(ts_mean_elevs[i] / ts_mean_elevs[-1] - 1) > NEAR_SS_THRESH
      (landscape has not yet reached near-steady-state;
       the > 1 case captures time series with overshoot behaviour)

One eligible index is drawn at random per landscape. Landscapes with no
eligible snapshots are reported and excluded from the output.

This script is part of the transient-stage analysis branch. Its output
(a parquet lookup table) is consumed by 02_extract_features.py via the
--transient-map argument.

Usage
-----
    python 01b_select_transient_snapshots.py \\
        --params-dir <path> \\
        --output-dir <path> \\
        [--min-elev-frac <float>] \\
        [--near-ss-thresh <float>] \\
        [--final-window <int>]

Arguments
---------
    --params-dir        Directory containing params-{job_id}.pkl files
                        produced by 01_generate_landscapes.py.
    --output-dir        Directory for the output parquet file
                        (default: current directory).
    --min-elev-frac     Minimum ratio of snapshot mean elevation to final
                        mean elevation for eligibility (default: 0.50).
    --near-ss-thresh    Fractional deviation from final mean elevation
                        below which a snapshot is considered near steady
                        state and excluded (default: 0.05). Applied as
                        abs(ratio - 1) > near_ss_thresh.
    --final-window      Number of trailing time steps used to compute the
                        reference final mean elevation. These steps are also
                        excluded from the candidate pool (default: 5).

Output
------
    transient_map.csv
        DataFrame with one row per landscape, columns:
            job_id              int   Job identifier
            landscape_idx       int   Landscape index within job
            selected_ts_index   int   Time step index of selected snapshot
            mean_elev_selected  float Mean elevation at selected snapshot (m)
            mean_elev_final     float Reference final mean elevation (m);
                                      mean of last --final-window time steps
            ratio               float mean_elev_selected / mean_elev_final
            n_eligible          int   Number of eligible snapshots available

Dependencies
------------
    numpy, pandas
    See environment.yml for pinned versions.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS (must match 01_generate_landscapes.py)
# =============================================================================

NUM_TS     = 100   # Total number of time steps per model run
SAVE_EVERY = 5     # Snapshot saved every N time steps

# Indices of saved snapshots: 4, 9, 14, ..., 99
SAVED_INDICES = np.arange(SAVE_EVERY - 1, NUM_TS, SAVE_EVERY)  # (i+1) % 5 == 0



# =============================================================================
# CORE SELECTION LOGIC
# =============================================================================

def select_transient_snapshot(ts_mean_elevs,
                               min_elev_frac=0.50,
                               near_ss_thresh=0.05,
                               final_window=5,
                               selection_seed=0):
    """
    Select one transient-stage snapshot index from a landscape's elevation
    time series.

    The reference final mean elevation is computed as the mean of the last
    `final_window` time step measurements. Saved snapshots whose time step
    index falls within that trailing window are excluded from the candidate
    pool (assumed to be at steady state).

    Parameters
    ----------
    ts_mean_elevs : array-like, shape (NUM_TS,)
        Mean elevation of core nodes at each time step, as saved in the
        ts_mean_elevs column of params-{job_id}.pkl.
    min_elev_frac : float
        Minimum ratio of snapshot mean elevation to final mean elevation.
        Default: 0.50.
    near_ss_thresh : float
        Exclusion threshold for near-steady-state snapshots. Snapshots
        where abs(ratio - 1) <= near_ss_thresh are excluded.
        Default: 0.05.
    final_window : int
        Number of trailing time steps used to compute the reference final
        mean elevation. These steps are also excluded from the candidate
        pool. Default: 5.
    selection_seed : int
        Per-landscape random seed derived as job_id * 100 + landscape_idx,
        following the same convention as elev_seed in 01_generate_landscapes.py.

    Returns
    -------
    selected_idx : int or None
        Time step index of the selected snapshot, or None if no eligible
        snapshots exist.
    eligible_indices : np.ndarray
        Array of all eligible time step indices (may be empty).
    """
    ts_mean_elevs = np.asarray(ts_mean_elevs, dtype=float)

    # Reference final mean: mean of last final_window time steps
    final_mean = ts_mean_elevs[-final_window:].mean()

    # Candidate pool: saved snapshots not in the trailing final_window
    final_window_start = NUM_TS - final_window          # e.g. 95 for window=5
    candidate_indices = SAVED_INDICES[
        SAVED_INDICES < final_window_start
    ]

    ratios = ts_mean_elevs[candidate_indices] / final_mean

    # Apply eligibility criteria
    not_too_young = ratios >= min_elev_frac
    not_near_ss   = np.abs(ratios - 1.0) > near_ss_thresh
    eligible_mask = not_too_young & not_near_ss

    eligible_indices = candidate_indices[eligible_mask]

    if len(eligible_indices) == 0:
        return None, eligible_indices

    np.random.seed(selection_seed)
    selected_idx = int(np.random.choice(eligible_indices))
    return selected_idx, eligible_indices


def build_transient_map(params_dir,
                        min_elev_frac=0.50,
                        near_ss_thresh=0.05,
                        final_window=5):
    """
    Build the transient snapshot lookup table from all params pkl files.

    Per-landscape selection seeds are derived as:
        selection_seed = seed * 100000 + job_id * 100 + landscape_idx

    This follows the same convention as 01_generate_landscapes.py, where
    elev_seed = 100 * job_id + landscape_idx. The base seed scales above
    both job_id (max ~999) and landscape_idx (max ~99), keeping each
    component in a distinct digit band.

    Parameters
    ----------
    params_dir : Path
        Directory containing params-{job_id}.pkl files.
    min_elev_frac : float
        Passed to select_transient_snapshot(). Default: 0.50.
    near_ss_thresh : float
        Passed to select_transient_snapshot(). Default: 0.05.
    final_window : int
        Passed to select_transient_snapshot(). Default: 5.

    Returns
    -------
    df_map : pd.DataFrame
        Transient snapshot lookup table.
    n_skipped : int
        Number of landscapes with no eligible snapshots.
    """
    pkl_files = sorted(params_dir.glob('params-*.pkl'))
    if not pkl_files:
        raise FileNotFoundError(
            f"No params-*.pkl files found in {params_dir}"
        )

    print(f"Found {len(pkl_files)} params file(s) in {params_dir}")

    rows = []
    n_skipped = 0

    for pkl_path in pkl_files:
        df_params = pd.read_pickle(pkl_path)


        # --- Legacy pkl fix: derive job_id from filename, map df-ind -> landscape_idx ---
        if 'job_id' not in df_params.columns and 'df-ind' in df_params.columns:
            job_id_from_file = int(pkl_path.stem.split('-')[1])
            df_params['job_id'] = job_id_from_file
            df_params = df_params.rename(columns={'df-ind': 'landscape_idx'})
        # -------------------------------------------------------------------------------



        # Guard: skip rows where ts_mean_elevs was not saved (e.g. failed runs)
        missing = df_params['ts_mean_elevs'].isna()
        if missing.any():
            warnings.warn(
                f"{pkl_path.name}: {missing.sum()} landscape(s) have missing "
                f"ts_mean_elevs and will be skipped."
            )
            df_params = df_params[~missing]

        for _, row in df_params.iterrows():
            job_id        = int(row['job_id'])
            landscape_idx = int(row['landscape_idx'])
            ts_mean_elevs = np.asarray(row['ts_mean_elevs'], dtype=float)

            # Per-landscape seed: job_id * 100 + landscape_idx
            # Same convention as elev_seed in 01_generate_landscapes.py
            selection_seed = job_id * 100 + landscape_idx

            selected_idx, eligible = select_transient_snapshot(
                ts_mean_elevs,
                min_elev_frac=min_elev_frac,
                near_ss_thresh=near_ss_thresh,
                final_window=final_window,
                selection_seed=selection_seed,
            )

            if selected_idx is None:
                warnings.warn(
                    f"job_id={job_id}, landscape_idx={landscape_idx}: "
                    f"no eligible transient snapshots. Skipping."
                )
                n_skipped += 1
                continue

            final_mean    = float(ts_mean_elevs[-final_window:].mean())
            selected_mean = float(ts_mean_elevs[selected_idx])

            rows.append({
                'job_id':             job_id,
                'landscape_idx':      landscape_idx,
                'selected_ts_index':  selected_idx,
                'mean_elev_selected': selected_mean,
                'mean_elev_final':    final_mean,
                'ratio':              selected_mean / final_mean,
                'n_eligible':         len(eligible),
            })

    df_map = pd.DataFrame(rows).astype({
        'job_id':            'int32',
        'landscape_idx':     'int32',
        'selected_ts_index': 'int32',
        'n_eligible':        'int32',
    })

    return df_map, n_skipped


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_transient_selection(params_dir,
                              job_id=None,
                              landscape_idx=None,
                              min_elev_frac=0.50,
                              near_ss_thresh=0.05,
                              final_window=5,
                              ax=None):
    """
    Plot the mean elevation time series for one landscape, annotated with
    the transient selection criteria and the selected snapshot.

    Intended for use in Jupyter notebooks to visually inspect the selection
    logic. Does not affect the pipeline.

    If job_id and landscape_idx are both None, a landscape is chosen at
    random from the available params files.

    Parameters
    ----------
    params_dir : str or Path
        Directory containing params-{job_id}.pkl files.
    job_id : int or None
        Job identifier. If None, a random landscape is selected.
    landscape_idx : int or None
        Landscape index within job. If None, a random landscape is selected.
    min_elev_frac : float
        Minimum ratio of snapshot mean elevation to final mean elevation
        for eligibility. Default: 0.50.
    near_ss_thresh : float
        Exclusion threshold for near-steady-state snapshots. Default: 0.05.
    final_window : int
        Number of trailing time steps used to compute the reference final
        mean elevation. Default: 5.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    info : dict
        Dictionary with keys: job_id, landscape_idx, selected_ts_index,
        final_mean, selected_mean, ratio, n_eligible, eligible_indices.
        Returns None for selected_ts_index and related fields if no
        eligible snapshots exist.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    params_dir = Path(params_dir)

    # --- Load landscape ---
    pkl_files = sorted(params_dir.glob('params-*.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f"No params-*.pkl files found in {params_dir}")

    if job_id is None or landscape_idx is None:
        # Random selection: pick a random pkl file, then a random row
        pkl_path = pkl_files[np.random.randint(len(pkl_files))]
        df_params = pd.read_pickle(pkl_path)
        df_params = df_params[df_params['ts_mean_elevs'].notna()]
        row = df_params.sample(1).iloc[0]
        job_id        = int(row['job_id'])
        landscape_idx = int(row['landscape_idx'])
    else:
        pkl_path = params_dir / f'params-{job_id}.pkl'
        df_params = pd.read_pickle(pkl_path)
        row = df_params[df_params['landscape_idx'] == landscape_idx].iloc[0]

    ts_mean_elevs = np.asarray(row['ts_mean_elevs'], dtype=float)

    # --- Run selection logic ---
    selection_seed = job_id * 100 + landscape_idx
    selected_idx, eligible_indices = select_transient_snapshot(
        ts_mean_elevs,
        min_elev_frac=min_elev_frac,
        near_ss_thresh=near_ss_thresh,
        final_window=final_window,
        selection_seed=selection_seed,
    )

    final_mean = ts_mean_elevs[-final_window:].mean()

    # Classify saved snapshots
    ineligible_saved = SAVED_INDICES[
        (SAVED_INDICES < NUM_TS - final_window) &
        ~np.isin(SAVED_INDICES, eligible_indices)
    ]
    final_window_indices = SAVED_INDICES[SAVED_INDICES >= NUM_TS - final_window]

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    ts_indices = np.arange(NUM_TS)

    # Full time series
    ax.plot(ts_indices, ts_mean_elevs, color='steelblue', lw=1.5,
            label='Mean elevation')

    # Reference lines
    ax.axhline(final_mean, color='black', lw=1.2, ls='--',
               label=f'Final mean (last {final_window} steps)')
    ax.axhline(min_elev_frac * final_mean, color='tomato', lw=1.0, ls=':',
               label=f'{min_elev_frac*100:.0f}% of final mean (lower cutoff)')
    ax.axhline((1.0 - near_ss_thresh) * final_mean, color='goldenrod',
               lw=1.0, ls=':', label=f'{(1-near_ss_thresh)*100:.0f}% of final mean')
    ax.axhline((1.0 + near_ss_thresh) * final_mean, color='goldenrod',
               lw=1.0, ls='--', label=f'{(1+near_ss_thresh)*100:.0f}% of final mean')

    # Shade the excluded near-steady-state band around final_mean
    ax.axhspan((1.0 - near_ss_thresh) * final_mean,
               (1.0 + near_ss_thresh) * final_mean,
               alpha=0.08, color='goldenrod', zorder=0)

    # Saved snapshots: final window (excluded by definition)
    ax.scatter(final_window_indices,
               ts_mean_elevs[final_window_indices],
               marker='o', s=40, color='gray', zorder=3,
               label='Saved (final window, excluded)')

    # Saved snapshots: ineligible (fail criteria)
    if len(ineligible_saved) > 0:
        ax.scatter(ineligible_saved,
                   ts_mean_elevs[ineligible_saved],
                   marker='o', s=40, color='lightcoral', zorder=3,
                   label='Saved (ineligible)')

    # Saved snapshots: eligible
    if len(eligible_indices) > 0:
        ax.scatter(eligible_indices,
                   ts_mean_elevs[eligible_indices],
                   marker='o', s=40, color='mediumseagreen', zorder=4,
                   label=f'Saved (eligible, n={len(eligible_indices)})')

    # Selected snapshot
    if selected_idx is not None:
        ax.scatter([selected_idx], [ts_mean_elevs[selected_idx]],
                   marker='*', s=200, color='crimson', zorder=5,
                   label=f'Selected (ts={selected_idx})')
        selected_mean = float(ts_mean_elevs[selected_idx])
        ratio = selected_mean / final_mean
    else:
        selected_mean = None
        ratio = None

    ax.set_xlabel('Time step index')
    ax.set_ylabel('Mean elevation (m)')
    ax.set_title(
        f'Transient snapshot selection  |  '
        f'job_id={job_id}, landscape_idx={landscape_idx}'
        + (f'\nSelected ts={selected_idx}  |  '
           f'ratio={ratio:.3f}  |  '
           f'eligible={len(eligible_indices)}'
           if selected_idx is not None
           else '\nNo eligible snapshots')
    )
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(-1, NUM_TS)

    info = {
        'job_id':             job_id,
        'landscape_idx':      landscape_idx,
        'selected_ts_index':  selected_idx,
        'final_mean':         float(final_mean),
        'selected_mean':      selected_mean,
        'ratio':              ratio,
        'n_eligible':         len(eligible_indices),
        'eligible_indices':   eligible_indices,
    }

    return ax, info


# =============================================================================
# MAIN
# =============================================================================

def main(params_dir, output_dir, min_elev_frac, near_ss_thresh, final_window):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_map, n_skipped = build_transient_map(
        params_dir=Path(params_dir),
        min_elev_frac=min_elev_frac,
        near_ss_thresh=near_ss_thresh,
        final_window=final_window,
    )

    out_path = output_dir / 'transient_map.csv'
    df_map.to_csv(out_path, index=False)

    n_total = len(df_map) + n_skipped
    print(f"\nTransient map saved: {out_path}")
    print(f"  Landscapes included : {len(df_map):>6d} / {n_total}")
    print(f"  Landscapes skipped  : {n_skipped:>6d} / {n_total}")
    print(f"  Selected ts_index   : min={df_map['selected_ts_index'].min()}, "
          f"max={df_map['selected_ts_index'].max()}, "
          f"mean={df_map['selected_ts_index'].mean():.1f}")
    print(f"  Ratio (sel/final)   : min={df_map['ratio'].min():.3f}, "
          f"max={df_map['ratio'].max():.3f}, "
          f"mean={df_map['ratio'].mean():.3f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Select one transient-stage snapshot per landscape from "
            "01_generate_landscapes.py output. Produces transient_map.csv "
            "for use with 02_extract_features.py --transient-map."
        )
    )
    parser.add_argument(
        '--params-dir', type=str, required=True,
        help="Directory containing params-{job_id}.pkl files."
    )
    parser.add_argument(
        '--output-dir', type=str, default='.',
        help="Directory for transient_map.parquet (default: current directory)."
    )
    parser.add_argument(
        '--min-elev-frac', type=float, default=0.50,
        help="Minimum ratio of snapshot to final mean elevation (default: 0.50)."
    )
    parser.add_argument(
        '--near-ss-thresh', type=float, default=0.05,
        help="Exclusion threshold for near-steady-state snapshots (default: 0.05)."
    )
    parser.add_argument(
        '--final-window', type=int, default=5,
        help="Number of trailing time steps used to compute the reference "
             "final mean elevation; these steps are excluded from the "
             "candidate pool (default: 5)."
    )

    args = parser.parse_args()
    main(
        params_dir=args.params_dir,
        output_dir=args.output_dir,
        min_elev_frac=args.min_elev_frac,
        near_ss_thresh=args.near_ss_thresh,
        final_window=args.final_window,
    )
