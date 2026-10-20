"""
pipeline_utils.py
-----------------
Shared data loading, splitting, and reproducibility utilities for the
landscape-evolution ML pipeline.

Imported by both ``03_train_models.py`` and ``04_feature_importance.py``.

Contents
--------
_git_hash()              Return current git commit hash for output filenames
read_retained_ids()      Read a steady-state retained-landscape manifest
identity_keys()          Identifier columns present in a loaded frame
landscape_ids()          Per-row landscape identities, as tuples
select_by_ids()          Select rows by landscape identity, order preserved
load_features()          Load and concatenate features-{job_id}.pkl files
split_features_labels()  Split DataFrame into feature matrix X and label matrix y

Notes
-----
``load_features`` can restrict the loaded ensemble to the landscapes verified
as being at steady state by ``01c_check_steady_state.py``, via its
``retained_csv`` argument.  Feature files are read, never rewritten: the
filter is a mask applied to the concatenated frame, so omitting the argument
restores the full ensemble with no regeneration.

A train / test split is recorded and reapplied by landscape identity --- the
``job_id``, ``landscape_idx`` and ``ts_index`` columns that
``02_extract_features.py`` writes into every feature file --- using
``landscape_ids`` and ``select_by_ids``.  An identity denotes the same
landscape whatever else was loaded alongside it, so a split survives a
changed job selection, a changed glob order and a changed steady-state
filter, and a landscape that is absent raises rather than resolving to a
different row.  The index assigned by the concatenation is incidental and
carries no meaning across loads; nothing in the pipeline records or consumes
it.

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

# Identifier columns of a retained-landscape manifest, in the refactored
# naming convention. ts_index is used when both the manifest and the feature
# table carry it, so that a landscape is matched at a specific time step.
_RETAINED_KEYS = ['job_id', 'landscape_idx']

# Identifier columns that constitute a landscape's identity, in the
# refactored naming convention written by 02_extract_features.py. Columns
# absent from a given frame are dropped by identity_keys().
_IDENTITY_KEYS = ['job_id', 'landscape_idx', 'ts_index']


def read_retained_ids(path: str) -> pd.DataFrame:
    """Read a retained-landscape manifest from ``01c_check_steady_state``.

    Parameters
    ----------
    path : str
        Path to a ``*-retained.csv`` file.  One such file exists per series,
        window, reference and git hash, so name the intended one explicitly
        rather than globbing.

    Returns
    -------
    pd.DataFrame
        The identifier columns only: ``job_id``, ``landscape_idx``, and
        ``ts_index`` when present.

    Raises
    ------
    FileNotFoundError
        If the manifest does not exist.
    ValueError
        If it lacks the identifier columns, or contains duplicate keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Retained-landscape manifest not found: {path}")
    man = pd.read_csv(path)
    missing = [c for c in _RETAINED_KEYS if c not in man.columns]
    if missing:
        raise ValueError(
            f"{path} lacks {missing}. Expected a manifest written by "
            f"01c_check_steady_state.py, in the refactored identifier "
            f"convention (job_id, landscape_idx, ts_index)."
        )
    cols = _RETAINED_KEYS + (['ts_index'] if 'ts_index' in man.columns else [])
    man = man[cols]
    if man.duplicated().any():
        raise ValueError(
            f"{path} contains duplicate identifiers; the filter would be "
            f"ambiguous. Check for overlapping parameter directories in the "
            f"run that produced it."
        )
    return man



def identity_keys(df: pd.DataFrame) -> list:
    """Return the identifier columns of ``df``, in canonical order.

    Parameters
    ----------
    df : pd.DataFrame
        A frame loaded by :func:`load_features`.

    Returns
    -------
    list of str
        Those of ``job_id``, ``landscape_idx`` and ``ts_index`` that are
        present.  ``ts_index`` is absent from feature tables extracted at a
        single time step.

    Raises
    ------
    ValueError
        If neither ``job_id`` nor ``landscape_idx`` is present, so that no
        identity can be formed.
    """
    keys = [c for c in _IDENTITY_KEYS if c in df.columns]
    if not set(_RETAINED_KEYS) <= set(keys):
        legacy = {'job_id': 'tmux-sess', 'landscape_idx': 'df-ind',
                  'ts_index': 'ts-ind'}
        found = [legacy[c] for c in _RETAINED_KEYS
                 if c not in keys and legacy[c] in df.columns]
        hint = (f" The frame carries the legacy name(s) {found} instead; "
                f"re-run the feature-extraction stage so that the "
                f"identifiers are written in the refactored convention."
                if found else "")
        raise ValueError(
            f"No landscape identity available: the frame lacks "
            f"{[c for c in _RETAINED_KEYS if c not in keys]}.{hint}"
        )
    return keys


def landscape_ids(df: pd.DataFrame, keys: list = None) -> list:
    """Return the identity of each row of ``df``, in row order.

    Parameters
    ----------
    df : pd.DataFrame
        A frame loaded by :func:`load_features`.
    keys : list of str, optional
        Identifier columns to use.  Defaults to :func:`identity_keys`.

    Returns
    -------
    list of tuple
        One tuple per row, with plain Python scalars so that the result
        pickles and compares independently of numpy dtypes.
    """
    keys = keys or identity_keys(df)
    return [tuple(r) for r in df[keys].to_numpy().tolist()]


def select_by_ids(df: pd.DataFrame, ids: list, keys: list = None
                  ) -> pd.DataFrame:
    """Select the rows of ``df`` named by ``ids``, in the order given.

    The identity-based counterpart of ``df.loc[idx]``.  Unlike index labels,
    an identity denotes the same landscape whatever else was loaded alongside
    it, so a split recorded this way survives a changed job selection, glob
    order or steady-state filter.

    Parameters
    ----------
    df : pd.DataFrame
        A frame loaded by :func:`load_features`.
    ids : list of tuple
        Identities as returned by :func:`landscape_ids`, typically read from
        ``_meta['train_ids']`` or ``_meta['test_ids']``.
    keys : list of str, optional
        The identifier columns that ``ids`` was built from, typically
        ``_meta['id_keys']``.  Defaults to :func:`identity_keys`, which is
        correct only if the frame carries the same columns as at training
        time.

    Returns
    -------
    pd.DataFrame
        The selected rows, ordered as ``ids`` is, with the index of ``df``
        preserved.

    Raises
    ------
    ValueError
        If ``df`` does not identify its rows uniquely, or if any requested
        identity is absent from it.
    """
    keys = keys or identity_keys(df)
    missing_cols = [c for c in keys if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Cannot select by identity: the frame lacks {missing_cols}, "
            f"which the recorded identities were built from."
        )

    have = pd.MultiIndex.from_frame(df[keys])
    if have.has_duplicates:
        dup = df.loc[have.duplicated(keep=False), keys].drop_duplicates()
        raise ValueError(
            f"The frame does not identify its rows uniquely on "
            f"{'+'.join(keys)}: {len(dup):,} identit(y/ies) occur more than "
            f"once. Check for overlapping feature files in --data-dir."
        )

    want = pd.MultiIndex.from_tuples([tuple(i) for i in ids], names=keys)
    pos = have.get_indexer(want)
    if (pos == -1).any():
        n = int((pos == -1).sum())
        example = [tuple(t) for t in want[pos == -1][:3]]
        raise ValueError(
            f"{n:,} of {len(want):,} recorded landscape identit(y/ies) are "
            f"absent from the loaded features, e.g. {example}. The frame does "
            f"not contain the ensemble this split was recorded on: check "
            f"--features-dir, --job-ids and the steady-state filter."
        )
    return df.iloc[pos]

def load_features(data_dir: str, job_ids: list = None,
                  features_hash: str = None,
                  retained_csv: str = None) -> pd.DataFrame:
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
    retained_csv : str, optional
        Path to a ``*-retained.csv`` manifest from
        ``01c_check_steady_state.py``.  When given, only the landscapes it
        lists are kept.  The index is deliberately not reset; see the module
        Notes.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with LE parameters and topographic features,
        restricted to the retained landscapes when ``retained_csv`` is given.

    Raises
    ------
    FileNotFoundError
        If specific job IDs are requested but their files are missing, or if
        no matching files are found.
    ValueError
        If ``features_hash`` is not specified and multiple hash versions are
        found in ``data_dir``, or if ``retained_csv`` is given and the loaded
        frame lacks the identifier columns needed to apply it.
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

    if retained_csv:
        man = read_retained_ids(retained_csv)

        missing = [c for c in _RETAINED_KEYS if c not in df.columns]
        if missing:
            legacy = {'job_id': 'tmux-sess', 'landscape_idx': 'df-ind'}
            found = [legacy[c] for c in missing if legacy[c] in df.columns]
            hint = (
                f" The frame carries the legacy name(s) {found} instead; "
                f"re-run the feature-extraction stage so that the "
                f"identifiers are written in the refactored convention."
                if found else ""
            )
            raise ValueError(
                f"Cannot apply the retained-landscape filter: the loaded "
                f"features lack {missing}.{hint}"
            )

        # Match on ts_index as well when both sides carry it, so that a
        # landscape is matched at a specific time step rather than in
        # aggregate.
        keys = list(_RETAINED_KEYS)
        if 'ts_index' in man.columns and 'ts_index' in df.columns:
            keys.append('ts_index')

        want = pd.MultiIndex.from_frame(man[keys])
        have = pd.MultiIndex.from_frame(df[keys])

        # Boolean masking leaves the index assigned by the concatenation
        # above intact, so a filtered frame is a strict subset of the
        # unfiltered one. Splits are recorded by landscape identity rather
        # than by index label (see the module Notes), so this is provenance
        # only and nothing downstream depends on it.
        n_before = len(df)
        df = df[have.isin(want)]

        unmatched = len(want.difference(have))
        print(f"  Steady-state filter: {os.path.basename(retained_csv)}")
        print(f"    retained {len(df):,} of {n_before:,} landscapes "
              f"({n_before - len(df):,} excluded), matched on "
              f"{'+'.join(keys)}.")
        if unmatched:
            print(f"    NOTE: {unmatched:,} of {len(want):,} manifest entries "
                  f"are absent from the loaded features; check that the "
                  f"manifest and the feature files cover the same jobs.")
        if len(df) == 0:
            raise ValueError(
                "The retained-landscape filter left no rows. The manifest "
                "and the feature files appear to describe different "
                "ensembles."
            )

    return df


# =============================================================================
# Feature / label splitting
# =============================================================================

# All columns in features-*.pkl that are LE parameters or derived quantities
# rather than topographic features.  These are excluded from X.
# Columns of a feature table that are not predictors: the LE parameters and
# their ratios (candidate labels), the imposed elevation error, and the
# identifier columns. The identifiers are excluded from X but remain on the
# loaded frame, which is what makes identity-based selection possible — see
# select_by_ids.
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
        Topographic feature matrix: every column of ``df`` outside
        ``_NON_FEATURE_COLS``, so the LE parameters, ``elev_err`` and the
        identifier columns are all excluded.
    y : pd.DataFrame
        log₁₀-transformed target matrix.

    Raises
    ------
    ValueError
        If any requested label column is not present in ``df``.

    Notes
    -----
    Because ``X`` carries no identifier columns, a split recorded by
    landscape identity is reapplied to the loaded frame and split afterwards,
    not the other way round::

        df_train = select_by_ids(df, meta['train_ids'], meta['id_keys'])
        X_train, y_train = split_features_labels(df_train, label_names)
    """
    X = df[[c for c in df.columns if c not in _NON_FEATURE_COLS]]

    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Label columns not found in DataFrame: {missing}")

    y = np.log10(df[label_cols])
    return X, y
