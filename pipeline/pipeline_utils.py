"""
pipeline_utils.py
-----------------
Shared data loading, splitting, and reproducibility utilities for the
landscape-evolution ML pipeline.

Imported by both ``03_train_models.py`` and ``04_feature_importance.py``.

Contents
--------
_git_hash()              Return current git commit hash for output filenames
load_features()          Load and concatenate features-{job_id}.pkl files
split_features_labels()  Split DataFrame into feature matrix X and label matrix y

Notes
-----
``split_features_labels`` applies log₁₀ transformation to all label columns
at this single location.  Label columns are stored as raw values in
features-*.pkl (written by 02_extract_features.py) so the DataFrame remains
self-describing.
"""

import glob
import os
import pickle
import subprocess

import numpy as np
import pandas as pd


# =============================================================================
# Reproducibility
# =============================================================================

def _git_hash(short: bool = True) -> str:
    """Return the current git commit hash, or ``'unknown'`` if unavailable.

    Parameters
    ----------
    short : bool
        If True, return the abbreviated 7-character hash.

    Returns
    -------
    str
    """
    try:
        cmd = ['git', 'rev-parse', '--short' if short else '', 'HEAD']
        return subprocess.check_output(
            [c for c in cmd if c], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


# =============================================================================
# Data loading
# =============================================================================

def load_features(data_dir: str, job_ids: list = None,
                  features_hash: str = None) -> pd.DataFrame:
    """Load and concatenate ``features-{job_id}-{hash}.pkl`` files.

    Parameters
    ----------
    data_dir : str
        Directory containing feature DataFrames written by
        ``02_extract_features.py``.
    job_ids : list of int or str, optional
        If given, load only the specified job IDs.  Otherwise all files
        matching ``features-*.pkl`` in ``data_dir`` are loaded.
    features_hash : str, optional
        Git hash suffix of the feature files to load (e.g. ``'abc1234'``).
        If None, the directory is scanned for a unique hash; a clear error
        is raised if multiple hashes are found.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with LE parameters and topographic features.

    Raises
    ------
    FileNotFoundError
        If specific job IDs are requested but their files are missing, or if
        no matching files are found.
    ValueError
        If ``features_hash`` is not specified and multiple hash versions are
        found in ``data_dir``.
    """
    def _file_hash(path):
        """Extract trailing hash from features-{job_id}-{hash}.pkl."""
        stem = os.path.basename(path)[len('features-'):-len('.pkl')]
        return stem.rsplit('-', 1)[-1]

    if job_ids:
        job_ids = sorted(job_ids)
        if features_hash:
            paths = [
                os.path.join(data_dir, f'features-{jid}-{features_hash}.pkl')
                for jid in job_ids
            ]
        else:
            # Find files for each job_id and check for unique hash
            paths = []
            for jid in job_ids:
                matches = sorted(glob.glob(
                    os.path.join(data_dir, f'features-{jid}-*.pkl')
                ))
                paths.extend(matches)
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Feature files not found: {missing}")
    else:
        if features_hash:
            pattern = os.path.join(data_dir, f'features-*-{features_hash}.pkl')
        else:
            pattern = os.path.join(data_dir, 'features-*.pkl')
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No feature files found matching '{pattern}'"
            )

    # If no hash was specified, verify all matched files share the same hash
    if not features_hash:
        hashes = sorted(set(_file_hash(p) for p in paths))
        if len(hashes) > 1:
            raise ValueError(
                f"Multiple feature file versions found in '{data_dir}':\n"
                f"  hashes: {hashes}\n"
                f"  Specify --features-hash to select one explicitly."
            )

    dfs = []
    for p in paths:
        with open(p, 'rb') as fh:
            dfs.append(pickle.load(fh))

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} landscapes from {len(paths)} file(s).")
    return df


# =============================================================================
# Feature / label splitting
# =============================================================================

# All columns in features-*.pkl that are LE parameters or derived quantities
# rather than topographic features.  These are excluded from X.
_NON_FEATURE_COLS = {'u', 'kh', 'ks', 'u_ks', 'kh_ks', 'u_kh',
                     'elev_err', 'job_id', 'landscape_idx', 'ts_index'}


def split_features_labels(df: pd.DataFrame, label_cols: list):
    """Split combined DataFrame into feature matrix X and label matrix y.

    All label columns are log₁₀-transformed at this single location, whether
    they are raw LE parameters (u, kh, ks) or raw ratios (u_ks, kh_ks, u_kh).
    Storing raw values in features-*.pkl keeps the DataFrame self-describing;
    the transform is applied here rather than being distributed across scripts.

    Parameters
    ----------
    df : pd.DataFrame
    label_cols : list of str
        Any subset of: ``u``, ``kh``, ``ks``, ``u_ks``, ``kh_ks``, ``u_kh``

    Returns
    -------
    X : pd.DataFrame
        Topographic feature matrix.
    y : pd.DataFrame
        log₁₀-transformed target matrix.

    Raises
    ------
    ValueError
        If any requested label column is not present in ``df``.
    """
    X = df[[c for c in df.columns if c not in _NON_FEATURE_COLS]]

    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Label columns not found in DataFrame: {missing}")

    y = np.log10(df[label_cols])
    return X, y
