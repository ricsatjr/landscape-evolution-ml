#!/usr/bin/env python
"""
03b_compare_features.py
----------------------------
Paired per-feature comparison between two observational settings of the same
landscapes (elev_err = 10 m and 1 m).

Both ensembles are generated from the same parameter draws and the same
steady-state snapshots, differing only in the elevation error applied before
flow routing.  Each landscape therefore appears once in each table, and the
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
med_disp    Typical unsigned change per landscape, in units of the
            between-landscape spread: median(|x01 - x10|) / (1.4826 *
            MAD(x10)), where MAD is the median absolute deviation about the
            median.  The 1.4826 factor makes the denominator a consistent
            estimator of the standard deviation under normality, so med_disp
            and shift are on the same scale and a single threshold means the
            same thing for both.  This is the median of the absolute
            differences, not the difference of medians: it does not measure a
            change in level and nothing cancels.  Unlike med_rel it is
            defined for every feature, including those centred on zero, and
            is the measure used to decide value stability.
med_rel     Median relative change, median(|x01 - x10| / |x10|).  Retained
            because a percentage is easier to interpret, but it divides by
            the feature's own value and is therefore unreliable for features
            that pass through zero -- the centred moments (skewness,
            kurtosis) and mean or median curvature.  Read it together with
            min_abs before quoting it.
min_abs     min(|x10|) / (1.4826 * MAD(x10)).  A diagnostic for the above: a
            small value means some landscapes have a near-zero denominator,
            so a large med_rel for that feature is an artifact rather than a
            real change.
n           Number of matched landscapes contributing.

shift and med_disp are a matched pair: shift is the signed change in level,
med_disp the typical unsigned change per landscape, both in units of the
between-landscape spread.  Value stability is judged on the two together,
never on either alone.  A signed mean cancels direction-varying change --
htcrv_min shifts by 0.09 while individual landscapes move by 0.81, because
roughly half move each way -- and an unsigned median misses a coherent
displacement smaller than the ensemble spread.

Alignment is on (job_id, landscape_idx), not on row position: load_features
concatenates with ignore_index=True, so positional indices are an artifact of
file ordering rather than a landscape identifier.

Restricting the comparison
--------------------------
--restrict-by names a parameter column and --range gives bounds on it, as
values or, with --range-as-percentile, as percentiles.  The restriction is
applied once, immediately after the two ensembles are matched, so the table,
the figures and the printed summary all describe the same landscapes.  Output
filenames carry a tag recording the restriction.

Narrowing the range of the underlying signal attenuates correlation for every
feature, so a restricted run's rho is interpretable against the other features
in that run -- the value-stable ones supply the baseline -- and not against
the unrestricted table.

Usage
-----
    python 03_compare_features.py \
        --n10-features-dir data/features/n10 \
        --n01-features-dir data/features/n01 \
        --out outputs/stability/paired-feature-shift.csv

Optionally restrict to the training subset, to match the subset on which
feature clustering is performed:

        --models-pkl data/models/n10/train-1200/N10-nested-cv-results-full-u_ks-kh_ks-9b33cab.pkl \
        --subset train
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_utils import load_features, _NON_FEATURE_COLS  # noqa: E402


KEY = ['job_id', 'landscape_idx']

# Scales MAD to a consistent estimator of the standard deviation under
# normality, so that med_disp is comparable with shift.
MAD_TO_SD = 1.4826


FAMILIES = [
    ('elevation',         '#4C72B0', 'o', lambda f: f.startswith('Z_')),
    ('gradient',          '#DD8452', 's', lambda f: f.startswith('grd')),
    ('curvature',         '#55A868', '^', lambda f: f.startswith('crv')),
    ('hilltop curvature', '#C44E52', 'v', lambda f: f.startswith('htcrv')),
    ('relief',            '#8172B3', 'D', lambda f: f.startswith('rlf')),
    ('network',           '#937860', 'P', lambda f: f in {
        'Rb', 'Rl', 'Rb0', 'Rl0', 'n0', 'n_nodes',
        'L0', 'l0_mean', 'path_max'}),
]


def _family(feature):
    """Assign a feature to a morphometric family for plotting."""
    for name, colour, marker, test in FAMILIES:
        if test(feature):
            return name, colour, marker
    return 'other', '#808080', 'X'


def plot_stability(df, path, shift_max=0.1, rel_max=0.1, dpi=200):
    """Two-panel summary of the paired per-feature comparison.

    Both panels share one vertical axis: features sorted by Spearman rho, so
    each row is one feature and the panels can be read across.

    Panel (a) gives Spearman rho and Pearson r.  The span between the two
    markers is rho - r; a wide span means the ordering of landscapes survives
    the change of setting but the shape of the relationship does not, which
    is the signature of a feature that cannot be recovered by rescaling.

    Panel (b) gives the two magnitude statistics on a log axis: |shift|, the
    signed change in level, and med_disp, the typical unsigned change per
    landscape.  The two differ most for features whose landscapes move in
    both directions, which a signed mean hides.  Both are expressed in units of the spread of the
    feature across landscapes, so horizontal position means the same thing
    for every feature; shift_max and rel_max are drawn as dashed and dotted
    lines and coincide when they are equal.  med_rel is deliberately
    not plotted: it divides by the feature's own value, so features centred
    on zero are placed far to the right for a reason that is not instability.
    Features left of the dashed boundary are stable in value, not merely in
    ordering.

    Colour and marker encode the morphometric family, which is what makes the
    systematic behaviour visible: quantities determined by discrete routing
    decisions separate from averages taken over many cells.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = df.dropna(subset=['rho']).copy()
    d = d.sort_values('rho').reset_index(drop=True)
    meta = [_family(f) for f in d['feature']]
    d['fam'] = [m[0] for m in meta]
    d['colour'] = [m[1] for m in meta]
    d['marker'] = [m[2] for m in meta]

    n = len(d)
    y = np.arange(n)
    fig, (ax0, ax1) = plt.subplots(
        1, 2, sharey=True, figsize=(9.5, max(4.5, 0.235 * n)),
        gridspec_kw={'wspace': 0.06},
    )

    # -- (a) ordering and shape -------------------------------------------
    has_r = 'pearson' in d.columns
    if has_r:
        ax0.hlines(y, d['rho'], d['pearson'].fillna(d['rho']),
                   color='0.78', lw=1.3, zorder=1)
    for i, row in d.iterrows():
        ax0.plot(row['rho'], i, marker=row['marker'], ms=6,
                 mfc='none', mec=row['colour'], mew=1.3, ls='none', zorder=3)
        if has_r and np.isfinite(row.get('pearson', np.nan)):
            ax0.plot(row['pearson'], i, marker=row['marker'], ms=5,
                     color=row['colour'], ls='none', zorder=3)

    ax0.axvline(0.0, color='0.45', lw=0.8, ls=':')
    ax0.set_yticks(y)
    ax0.set_yticklabels(d['feature'], fontsize=8)
    for tick, colour in zip(ax0.get_yticklabels(), d['colour']):
        tick.set_color(colour)
    ax0.tick_params(axis='y', length=0)
    ax0.set_ylim(-0.8, n - 0.2)
    ax0.set_xlabel('correlation between settings')
    ax0.grid(axis='x', color='0.92', lw=0.6)
    ax0.set_axisbelow(True)
    ax0.set_title('(a) ordering and shape', loc='left', fontsize=10)
    ax0.legend(handles=[
        Line2D([], [], marker='o', ls='none', mfc='none', mec='0.35',
               ms=6, label=r'Spearman $\rho$'),
        Line2D([], [], marker='o', ls='none', color='0.35',
               ms=5, label='Pearson $r$'),
    ], loc='upper left', fontsize=8, frameon=True)

    # -- (b) magnitude of displacement ------------------------------------
    floor = 1e-5
    shift = d['shift'].abs().clip(lower=floor) if 'shift' in d else None
    rel = d['med_disp'].abs().clip(lower=floor) if 'med_disp' in d else None

    ax1.axvspan(floor, min(shift_max, rel_max), color='0.93', zorder=0)
    # A single line when the two thresholds coincide, so the figure does not
    # imply a distinction the caption would then have to explain away.
    if np.isclose(shift_max, rel_max):
        ax1.axvline(shift_max, color='0.6', lw=0.8, ls='--', zorder=1)
    else:
        for bound, style in ((shift_max, '--'), (rel_max, ':')):
            ax1.axvline(bound, color='0.6', lw=0.8, ls=style, zorder=1)

    if shift is not None and rel is not None:
        ax1.hlines(y, shift, rel, color='0.78', lw=1.3, zorder=2)
    for i, row in d.iterrows():
        if shift is not None:
            ax1.plot(shift[i], i, marker=row['marker'], ms=6, mfc='none',
                     mec=row['colour'], mew=1.3, ls='none', zorder=3)
        if rel is not None:
            ax1.plot(rel[i], i, marker=row['marker'], ms=5,
                     color=row['colour'], ls='none', zorder=3)

    ax1.set_xscale('log')
    # The clip floor only exists to keep zeros on a log axis; letting it set
    # the lower limit would spend several decades on empty space.
    finite = np.concatenate([
        v.to_numpy() for v in (shift, rel) if v is not None])
    finite = finite[finite > floor]
    if finite.size:
        ax1.set_xlim(finite.min() / 3.0, finite.max() * 3.0)

    ax1.set_xlabel('change relative to between-landscape spread')
    ax1.grid(axis='x', color='0.92', lw=0.6)
    ax1.set_axisbelow(True)
    ax1.set_title('(b) displacement of value', loc='left', fontsize=10)
    ax1.legend(handles=[
        Line2D([], [], marker='o', ls='none', mfc='none', mec='0.35',
               ms=6, label='|shift|'),
        Line2D([], [], marker='o', ls='none', color='0.35',
               ms=5, label='med_disp'),
    ], loc='upper right', fontsize=8, frameon=True)

    fams = list(dict.fromkeys(d['fam']))
    lookup = {name: (c, m) for name, c, m, _ in FAMILIES}
    lookup['other'] = ('#808080', 'X')
    fig.legend(
        handles=[Line2D([], [], marker=lookup[f][1], ls='none',
                        color=lookup[f][0], ms=6, label=f) for f in fams],
        loc='lower center', ncol=min(len(fams), 4), fontsize=8,
        frameon=False, bbox_to_anchor=(0.5, -0.015),
    )

    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path



def paired_stats(a, b, sd=None, mad=None):
    """Paired comparison statistics for one feature over a set of landscapes.

    sd and mad may be supplied to override the scales computed from `a`.  That
    is available for callers that need one yardstick across separate runs:
    within a restricted range a feature's spread shrinks, so shift and
    med_disp scaled by that range's own spread are inflated relative to the
    whole-ensemble values.  Left unset, both scales come from `a`.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return {'rho': np.nan, 'pearson': np.nan, 'shift': np.nan,
                'med_disp': np.nan, 'med_rel': np.nan, 'min_abs': np.nan,
                'n': len(a)}

    rho = stats.spearmanr(a, b).statistic
    r = stats.pearsonr(a, b).statistic if np.std(a) and np.std(b) else np.nan

    _sd = np.std(a, ddof=1) if sd is None else sd
    _mad = (np.median(np.abs(a - np.median(a))) * MAD_TO_SD
            if mad is None else mad)

    shift = (np.mean(b - a) / _sd) if _sd > 0 else np.nan
    med_disp = (np.median(np.abs(b - a)) / _mad) if _mad > 0 else np.nan
    min_abs = (np.min(np.abs(a)) / _mad) if _mad > 0 else np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(b - a) / np.abs(a)
    med_rel = (np.nanmedian(rel[np.isfinite(rel)])
               if np.isfinite(rel).any() else np.nan)

    return {'rho': rho, 'pearson': r, 'shift': shift, 'med_disp': med_disp,
            'med_rel': med_rel, 'min_abs': min_abs, 'n': len(a)}


def resolve_range(values, spec, as_percentile):
    """Turn a 'lo:hi' specification into a boolean mask over `values`.

    Either side may be left blank to mean open.  Bounds are read as values of
    the label unless as_percentile, in which case they are percentiles of it
    and are converted with np.quantile.  The upper bound is inclusive so that
    '90:100' keeps the maximum rather than silently dropping it.
    """
    if ':' not in spec:
        sys.exit(f"ERROR: range '{spec}' must be written 'lo:hi' "
                 f"(either side may be left blank).")
    lo_s, hi_s = spec.split(':', 1)
    try:
        lo = float(lo_s) if lo_s.strip() else None
        hi = float(hi_s) if hi_s.strip() else None
    except ValueError:
        sys.exit(f"ERROR: could not read bounds in '{spec}'.")

    if as_percentile:
        for v in (lo, hi):
            if v is not None and not 0 <= v <= 100:
                sys.exit(f"ERROR: percentile {v} outside 0-100 in '{spec}'.")
        lo_v = np.quantile(values, lo / 100) if lo is not None else -np.inf
        hi_v = np.quantile(values, hi / 100) if hi is not None else np.inf
    else:
        lo_v = lo if lo is not None else -np.inf
        hi_v = hi if hi is not None else np.inf

    if lo_v > hi_v:
        sys.exit(f"ERROR: lower bound exceeds upper bound in '{spec}'.")
    return (values >= lo_v) & (values <= hi_v), lo_v, hi_v


def range_tag(spec, as_percentile):
    """Filename-safe tag recording the restriction, so that a restricted run
    cannot silently overwrite an unrestricted one."""
    t = spec.replace(':', '-').replace('.', 'p')
    return ('pct' if as_percentile else 'val') + t



def plot_paired(merged, features, path, colour_by=None, dpi=200, log=False):
    """Per-landscape scatter of a feature's two values, one panel per feature.

    Each point is one landscape: its value at 10 m against its value at 1 m.
    The 1:1 line is where a feature would sit if the observational setting
    made no difference, so vertical offset from it is displacement of level
    and departure from a straight line is the loss of shape that a rank
    correlation cannot show.  Colouring by a parameter ratio reveals whether
    that departure is confined to part of the parameter space.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    features = [f for f in features if f + '_n10' in merged.columns]
    if not features:
        return None

    ncol = min(len(features), 3)
    nrow = int(np.ceil(len(features) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.9 * nrow),
                             squeeze=False)

    c = None
    if colour_by and colour_by in merged.columns:
        c = merged[colour_by].to_numpy(dtype=float)
        norm = LogNorm(vmin=np.nanmin(c[c > 0]), vmax=np.nanmax(c)) \
            if np.all(c[np.isfinite(c)] > 0) else None

    sc = None
    for k, f in enumerate(features):
        ax = axes[k // ncol][k % ncol]
        x = merged[f + '_n10'].to_numpy(dtype=float)
        y = merged[f + '_n01'].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)

        if c is not None:
            sc = ax.scatter(x[ok], y[ok], c=c[ok], norm=norm, s=7,
                            cmap='viridis', linewidths=0, alpha=0.75)
        else:
            ax.scatter(x[ok], y[ok], s=7, color='#4C72B0',
                       linewidths=0, alpha=0.6)

        lo = np.nanmin([x[ok].min(), y[ok].min()])
        hi = np.nanmax([x[ok].max(), y[ok].max()])
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color='0.35', lw=0.9, ls='--', zorder=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect('equal', adjustable='box')
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        st = paired_stats(x, y)
        ax.set_title(f, fontsize=10, loc='left')
        ax.text(0.04, 0.96,
                rf"$\rho$={st['rho']:.3f}" + "\n" + rf"$r$={st['pearson']:.3f}",
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                color='0.25')
        ax.set_xlabel(r'value at $\pm$10 m')
        ax.set_ylabel(r'value at $\pm$1 m')
        ax.grid(color='0.93', lw=0.6)
        ax.set_axisbelow(True)

    for k in range(len(features), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')

    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85,
                          pad=0.02)
        cb.set_label(colour_by)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


def subset_frame(df, models_pkl, subset):
    """Restrict a freshly loaded feature frame to the train or test split."""
    if subset == 'all':
        return df
    if not models_pkl:
        sys.exit("ERROR: --subset train/test requires --models-pkl.")
    with open(models_pkl, 'rb') as f:
        meta = pickle.load(f).get('_meta', {})
    idx = meta.get(f'{subset}_idx')
    if not idx:
        sys.exit(f"ERROR: {subset}_idx not found in {models_pkl} _meta.")
    missing = set(idx) - set(df.index)
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} split indices absent from the loaded frame.\n"
            f"       The split indices are positional labels of the concatenated\n"
            f"       table, so they are only valid if the same job_ids are loaded\n"
            f"       in the same order as when the models were trained."
        )
    return df.loc[idx]


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
                   help='Full-feature pkl supplying split indices for --subset.')
    p.add_argument('--subset', choices=['all', 'train', 'test'], default='all',
                   help="Landscapes to include. 'train' matches the subset on "
                        "which feature clustering is performed. Default: all.")
    p.add_argument('--features', nargs='+', default=None,
                   help='Restrict output to these features. Default: all.')
    p.add_argument('--out', required=True, help='Output CSV path.')
    p.add_argument('--plot', default=None,
                   help='Path for the summary figure (e.g. .png or .pdf). '
                        'Omitted by default.')
    p.add_argument('--shift-max', type=float, default=0.1,
                   help='Value-stable boundary for |shift|, in units of '
                        'sd(x10). Default 0.1.')
    p.add_argument('--plot-paired', default=None,
                   help='Path for a per-landscape scatter of each feature '
                        "in --paired-features: its value at 10 m against its "
                        'value at 1 m, with the 1:1 line. Shows displacement '
                        'and loss of shape directly, which the summary '
                        'statistics only summarise.')
    p.add_argument('--paired-features', nargs='+', default=('Rb0', 'Rl0'),
                   help='Features to draw in --plot-paired. Default Rb0 Rl0.')
    p.add_argument('--paired-colour-by', default=None,
                   help='Label column to colour the paired scatter by. '
                        'Defaults to --restrict-by when that is given.')
    p.add_argument('--paired-log', action='store_true',
                   help='Log axes on the paired scatter.')
    p.add_argument('--restrict-by', default=None,
                   help="Label column to restrict the comparison on, e.g. "
                        "'kh_ks' or 'u_ks'. Used with --range. Also the "
                        'default colouring for the paired scatter.')
    p.add_argument('--range', default=None, metavar='LO:HI',
                   help="Restrict the comparison to landscapes whose "
                        "--restrict-by label falls in this range, written "
                        "'lo:hi' with either side optionally blank (e.g. "
                        "'0.1:1', ':0.5', '1:'). Every output of the run -- "
                        'table, figures, printed summary -- then describes '
                        'only those landscapes. Bounds are values unless '
                        '--range-as-percentile is set.')
    p.add_argument('--range-as-percentile', action='store_true',
                   help='Read --range bounds as percentiles of the label '
                        'rather than as its values.')
    p.add_argument('--rel-max', type=float, default=0.1,
                   help='Value-stable boundary for med_disp, in the same units '
                        'as shift once the MAD consistency constant is '
                        'applied. Default 0.1.')
    args = p.parse_args()

    print("Loading 10 m ensemble...")
    df10 = load_features(args.n10_features_dir, args.job_ids,
                         args.n10_features_hash)
    print("Loading 1 m ensemble...")
    df01 = load_features(args.n01_features_dir, args.job_ids,
                         args.n01_features_hash)

    df10 = subset_frame(df10, args.models_pkl, args.subset)
    df01 = subset_frame(df01, args.models_pkl, args.subset)

    for name, df in (('n10', df10), ('n01', df01)):
        missing = [k for k in KEY if k not in df.columns]
        if missing:
            sys.exit(f"ERROR: {name} table lacks {missing}; cannot align pairs.")

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

    # The restricting label is a parameter of the landscape, identical in both
    # ensembles, so it is carried through from the 10 m table unsuffixed.
    if args.range and not args.restrict_by:
        sys.exit("ERROR: --range needs --restrict-by to say which label the "
                 "bounds apply to.")
    if args.range_as_percentile and not args.range:
        sys.exit("ERROR: --range-as-percentile has no effect without --range.")

    carry = []
    colour_by = args.paired_colour_by or args.restrict_by
    for col in (args.restrict_by, colour_by):
        if col and col not in carry:
            if col not in df10.columns:
                sys.exit(
                    f"ERROR: '{col}' not in the feature table. Available: "
                    f"{sorted(c for c in df10.columns if c in _NON_FEATURE_COLS)}")
            carry.append(col)

    merged = df10[KEY + carry + feat_cols].merge(
        df01[KEY + feat_cols], on=KEY, suffixes=('_n10', '_n01'),
        how='inner', validate='one_to_one',
    )
    print(f"\nMatched {len(merged):,} landscapes on {KEY} "
          f"(n10={len(df10):,}, n01={len(df01):,}).")
    if len(merged) < min(len(df10), len(df01)):
        print("  WARNING: some landscapes did not match; only pairs are used.")

    # The restriction is applied once, here, so that every statistic and every
    # figure downstream describes the same landscapes.  Note that narrowing
    # the range of the underlying signal attenuates correlation for every
    # feature, so a restricted run's rho is read against the other features in
    # that run -- the value-stable ones supply the baseline -- and not against
    # the whole-ensemble value.
    tag = ''
    if args.range:
        vals = merged[args.restrict_by].to_numpy(dtype=float)
        mask, lo_v, hi_v = resolve_range(vals, args.range,
                                         args.range_as_percentile)
        kept, total = int(mask.sum()), len(merged)
        print(f"\nRestricted to {args.restrict_by} in [{lo_v:.6g}, {hi_v:.6g}]"
              + (f" (percentiles {args.range})" if args.range_as_percentile
                 else "")
              + f": {kept:,} of {total:,} landscapes.")
        if kept < 3:
            sys.exit("ERROR: fewer than 3 landscapes in range; nothing to "
                     "compare.")
        if kept < 50:
            print("  WARNING: a narrow restriction attenuates every "
                  "correlation; read features against each other, not "
                  "against the unrestricted table.")
        merged = merged.loc[mask].reset_index(drop=True)
        tag = '-' + args.restrict_by + '-' + range_tag(
            args.range, args.range_as_percentile)

    rows = []
    for f in feat_cols:
        a = merged[f + '_n10'].to_numpy(dtype=float)
        b = merged[f + '_n01'].to_numpy(dtype=float)
        rows.append({'feature': f, **paired_stats(a, b)})

    out = pd.DataFrame(rows).sort_values('rho', ascending=False)

    def tagged(path):
        """Insert the restriction tag before the extension, so a restricted
        run cannot silently overwrite an unrestricted one."""
        if not tag:
            return path
        root, ext = os.path.splitext(path)
        return root + tag + ext

    out_path = tagged(args.out)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    out.to_csv(out_path, index=False)

    pd.set_option('display.width', 120)
    print(f"\nSubset: {args.subset}\n")
    print(out.to_string(index=False,
                        float_format=lambda v: f'{v: .4f}'))
    print(f"\nWrote {out_path}")

    if args.plot:
        plot_stability(out, tagged(args.plot), shift_max=args.shift_max,
                       rel_max=args.rel_max)
        print(f"Wrote {tagged(args.plot)}")

    if args.plot_paired:
        p = plot_paired(merged, list(args.paired_features),
                        tagged(args.plot_paired),
                        colour_by=colour_by, log=args.paired_log)
        if p:
            print(f"Wrote {p}")
        else:
            print(f"WARNING: none of {list(args.paired_features)} found; "
                  f"no paired scatter written.")


if __name__ == '__main__':
    main()
