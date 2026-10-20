#!/usr/bin/env python
"""
03b_compare_features.py
----------------------------
Paired per-feature comparison between two observational settings of the same
landscapes (elev_err = 10 m and 1 m).

Both ensembles are generated from the same parameter draws and the same
final landscape snapshots, differing only in the elevation error applied
before flow routing.  Each landscape therefore appears once in each table, and the
two values of a given feature form a matched pair.  This script quantifies,
for every feature, how far its value is a property of the landform and how
far it is a property of the observation.

Statistics reported per feature
-------------------------------
rho         Spearman rank correlation between the paired values.  Near 1
            means the feature orders landscapes identically in both settings.
pearson     Pearson correlation between the paired values.
shift       Standardized paired shift, mean(x01 - x10) / sd(x10).  A feature
            can have rho close to 1 and a nonzero shift: identical ordering
            at a displaced level.  That combination is what produces
            equivalent accuracy on retraining but failure on transfer.
med_rel     Median relative change, median(|x01 - x10| / |x10|).
n           Number of matched landscapes contributing.

Alignment is on (job_id, landscape_idx), not on row position: load_features
concatenates with ignore_index=True, so positional indices are an artifact of
file ordering rather than a landscape identifier.  The train/test split is
applied the same way, from the identities recorded in a results pkl _meta,
so --subset is unaffected by which jobs are loaded or in what order.

Usage
-----
Characterization defaults to the training subset, matching the subset on
which feature clustering and permutation importance are computed in
04_feature_importance.py, so that a feature's stability and its importance
describe the same landscapes.  The reason is not only parallelism: these
statistics sit alongside the clustering that selects features, so computing
them over the whole ensemble would let the held-out landscapes influence a
modelling decision.  --models-pkl supplies that subset and is therefore
required unless --subset all is given.

    python 03b_compare_features.py \
        --n10-features-dir <path-to-n10-features-dir> \
        --n01-features-dir <path-to-n01-features-dir> \
        --models-pkl <path-to-models-pkl> \
        --out <path-to-output-csv>

To characterize every landscape in the feature directories instead:

        --subset all

No steady-state manifest is read at any point.  The split identities in the
pkl _meta already name none but the landscapes the models were trained on,
so --subset train and --subset test are restricted to the retained ensemble
by construction.

Two implications of that.  --subset all is not restricted: it includes any
landscape in the feature directories that failed the steady-state check of
01c_check_steady_state.py, and so describes a different population from the
default.  And the retained ensemble itself is the union of the two splits
rather than a choice of its own, so characterizing it means taking --subset
train and --subset test separately.

--out is given explicitly and carries no subset tag of its own, so name the
runs apart if both are kept: a second invocation otherwise overwrites the
first with a table describing a different population.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_utils import (load_features, select_by_ids,  # noqa: E402
                            _NON_FEATURE_COLS)


KEY = ['job_id', 'landscape_idx']


def read_models_meta(models_pkl):
    """Return the ``_meta`` of a results pkl, or ``{}`` if none was given."""
    if not models_pkl:
        return {}
    with open(models_pkl, 'rb') as f:
        return pickle.load(f).get('_meta', {})


def subset_frame(df, meta, subset, models_pkl):
    """Restrict a freshly loaded feature frame to the train or test split.

    The split is applied by landscape identity, as the paired alignment is,
    so it does not depend on which jobs were loaded or in what order.
    """
    if subset == 'all':
        return df
    if not meta:
        sys.exit(
            f"ERROR: --subset {subset} requires --models-pkl, which supplies "
            f"the split identities.\n"
            f"       'train' is the default, matching the subset analysed by "
            f"04_feature_importance.py.\n"
            f"       Pass --subset all to characterize the whole ensemble "
            f"instead."
        )
    ids = meta.get(f'{subset}_ids')
    if not ids:
        sys.exit(
            f"ERROR: {subset}_ids not found in {models_pkl} _meta.\n"
            f"       The split is reapplied by landscape identity; retrain\n"
            f"       with the current 03_train_models.py to produce a pkl\n"
            f"       that records it."
        )
    try:
        return select_by_ids(df, ids, meta.get('id_keys'))
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--n10-features-dir', required=True)
    p.add_argument('--n01-features-dir', required=True)
    p.add_argument('--n10-features-hash', default=None)
    p.add_argument('--n01-features-hash', default=None)
    p.add_argument('--job-ids', nargs='+', type=int, default=None,
                   help='Restrict to these job IDs. Default: all found.')
    p.add_argument('--models-pkl', default=None,
                   help='Full-feature pkl supplying the split identities for '
                        '--subset, and the steady-state manifest its models '
                        'were trained under.')
    p.add_argument('--subset', choices=['all', 'train', 'test'], default='train',
                   help="Landscapes to include. The default, 'train', matches "
                        "the subset on which feature clustering and "
                        "permutation importance are computed in "
                        "04_feature_importance.py, so that a feature's "
                        "stability and its importance describe the same "
                        "landscapes. 'train' and 'test' require "
                        "--models-pkl.")
    p.add_argument('--features', nargs='+', default=None,
                   help='Restrict output to these features. Default: all.')
    p.add_argument('--out', required=True, help='Output CSV path.')
    args = p.parse_args()

    # The pkl supplies the split identities, which already name none but the
    # landscapes its models were trained on. No steady-state manifest is
    # read: applying one would change nothing for --subset train/test while
    # making the run depend on that path still resolving. The manifest
    # recorded in _meta is reported as provenance only.
    meta = read_models_meta(args.models_pkl)
    if meta.get('retained_csv') and args.subset != 'all':
        print(f"Split identities describe the steady-state subset of "
              f"{os.path.basename(meta['retained_csv'])}.")

    print("Loading 10 m ensemble...")
    df10 = load_features(args.n10_features_dir, args.job_ids,
                         args.n10_features_hash)
    print("Loading 1 m ensemble...")
    df01 = load_features(args.n01_features_dir, args.job_ids,
                         args.n01_features_hash)

    for name, df in (('n10', df10), ('n01', df01)):
        missing = [k for k in KEY if k not in df.columns]
        if missing:
            sys.exit(f"ERROR: {name} table lacks {missing}; cannot align pairs.")

    df10 = subset_frame(df10, meta, args.subset, args.models_pkl)
    df01 = subset_frame(df01, meta, args.subset, args.models_pkl)

    # Sanity check on the ensembles themselves before comparing features.
    if 'elev_err' in df10.columns and 'elev_err' in df01.columns:
        e10 = sorted(df10['elev_err'].unique())
        e01 = sorted(df01['elev_err'].unique())
        print(f"  elev_err  n10={e10}  n01={e01}")
        if e10 == e01:
            print("  WARNING: both tables report the same elev_err. "
                  "Check that the directories are not the same ensemble.")

    feat_cols = [c for c in df10.columns if c not in _NON_FEATURE_COLS]
    if args.features:
        unknown = [f for f in args.features if f not in feat_cols]
        if unknown:
            sys.exit(f"ERROR: unknown features: {unknown}")
        feat_cols = list(args.features)

    merged = df10[KEY + feat_cols].merge(
        df01[KEY + feat_cols], on=KEY, suffixes=('_n10', '_n01'),
        how='inner', validate='one_to_one',
    )
    print(f"\nMatched {len(merged):,} landscapes on {KEY} "
          f"(n10={len(df10):,}, n01={len(df01):,}).")
    if len(merged) < min(len(df10), len(df01)):
        print("  WARNING: some landscapes did not match; only pairs are used.")

    rows = []
    for f in feat_cols:
        a = merged[f + '_n10'].to_numpy(dtype=float)
        b = merged[f + '_n01'].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 3:
            rows.append({'feature': f, 'rho': np.nan, 'pearson': np.nan,
                         'shift': np.nan, 'med_rel': np.nan, 'n': len(a)})
            continue

        rho = stats.spearmanr(a, b).statistic
        r = stats.pearsonr(a, b).statistic if np.std(a) and np.std(b) else np.nan
        sd = np.std(a, ddof=1)
        shift = (np.mean(b - a) / sd) if sd > 0 else np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.abs(b - a) / np.abs(a)
        med_rel = np.nanmedian(rel[np.isfinite(rel)]) if np.isfinite(rel).any() else np.nan

        rows.append({'feature': f, 'rho': rho, 'pearson': r, 'shift': shift,
                     'med_rel': med_rel, 'n': len(a)})

    out = pd.DataFrame(rows).sort_values('rho', ascending=False)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out.to_csv(args.out, index=False)

    pd.set_option('display.width', 120)
    print(f"\nSubset: {args.subset}\n")
    print(out.to_string(index=False,
                        float_format=lambda v: f'{v: .4f}'))
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
