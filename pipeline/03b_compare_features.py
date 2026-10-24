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
delta_net       Net movement across landscapes, median(x01 - x10) / scale.
            Signed.  A feature can have rho close to 1 and a nonzero delta_net:
            identical ordering at a displaced level.  That combination is
            what produces equivalent accuracy on retraining but failure on
            transfer.
delta_tot    Total movement per landscape, median(|x01 - x10|) / scale.
            Unsigned.  The median of the absolute differences, not the
            difference of medians: nothing cancels, so landscapes moving in
            opposite directions both count.  Because it is scaled by the
            ensemble spread rather than by the feature's own value, it is
            defined for every feature, including those centred on zero, and
            is the measure used to decide value stability.
scale       The common denominator of both, 1.4826 * MADM(x10), where MADM is
            the median absolute deviation from the median,
            median(|x10 - median(x10)|).  MADM is written out rather than
            abbreviated MAD, which is also used for the mean absolute
            deviation and which names neither of the two medians involved.
            One scale for both statistics is what makes a single threshold
            mean the same thing for each.  The 1.4826 is a fixed unit
            convention, not a distributional claim: it would make MADM a
            consistent estimator of the standard deviation if the feature
            were normal, which these features are not -- sd(x10) exceeds
            scale by a median factor of 1.29 across the feature set and by
            4.8 for htcrv_min.  Because the constant divides both numerators
            it cancels from every comparison between them and from every
            ranking, so no result here depends on it.  What matters is only
            that the two statistics share it.  See Hampel (1974), who
            introduced the statistic as the median deviation and used the
            equivalent 0.6745^-1 rescaling.  MADM is
            chosen for its breakdown point of 1/2, the highest attainable
            for a scale estimator, against 1/4 for the interquartile range;
            the cost is efficiency, which is immaterial for a unit computed
            once from more than a thousand landscapes.  Its known limitation
            is that it measures deviation symmetrically about the median
            (Hampel 1974; Rousseeuw and Croux 1993), which is why the choice
            of robust scale is checked against the interquartile range
            (Spearman rho 0.998 between the two denominations of delta_tot,
            so the choice does not reorder the feature set).
stable      (|delta_net| < delta_net_max) & (delta_tot < delta_tot_max).  Recorded here so
            that the classification drawn in the figure is in the table too,
            and downstream scripts do not each re-derive it.
n           Number of matched landscapes contributing.

delta_net and delta_tot are a matched pair over one denominator, differing in a
single place: when the absolute value is taken.  delta_net subtracts and then
aggregates, so landscapes moving opposite ways cancel; delta_tot takes the
magnitude first, so nothing cancels.  The mechanical analogy is displacement
against distance travelled, with the cancellation running across landscapes
rather than along one path: delta_tot says whether the values moved, delta_net
says whether they moved together, and delta_tot >= |delta_net| always.

Value stability is judged on the two together, never on either alone.  delta_net
alone is ambiguous at small values -- it reads near zero both when nothing
moved and when movement cancelled: htcrv_min shifts by 0.03 while individual
landscapes move by 0.80, because they move both ways.  delta_tot alone cannot
separate a uniform offset, which preserves ordering and could in principle be
recalibrated, from scatter, which cannot.

Only the magnitude of delta_net enters the stability criterion, since a feature
displaced by a given amount is equally unreliable in either direction.  The
sign is retained in the table because it belongs to the interpretation of
mechanism -- Rl0 is displaced downward and Rb0 upward -- not to the
assessment of robustness.

The numerator of delta_net is a median rather than a mean, for two reasons.  It
is the choice that makes delta_tot >= |delta_net| hold and so makes the gap
between the two readable as incoherence alone.  And it is robust, which matters
for features whose paired differences are themselves heavy-tailed: for
htcrv_min the mean difference is 0.43 and the median 0.03, so a minority of
landscapes with large one-way movement sets the mean.  That asymmetry is
real and is worth knowing, but it is a statement about the tail of the
difference distribution rather than about the typical landscape, and
reporting it through delta_net would misdescribe the ensemble.  Nothing is lost
operationally: htcrv_min fails the stability test on delta_tot either way.

Alignment is on (job_id, landscape_idx), not on row position: load_features
concatenates with ignore_index=True, so positional indices are an artifact of
file ordering rather than a landscape identifier.  The train/test split is
applied the same way, from the identities recorded in a results pkl _meta,
so --subset is unaffected by which jobs are loaded or in what order.

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
        --out-csv <path-to-output-csv> \
        --out-plot <path-to-output-figure>

The script produces two things: the table, and one figure.  The figure has
four panels.  A and B are the per-feature summary, one row per feature, A
giving the correlation pair and B the displacement pair.  C and D are the
per-landscape scatters of the two features named by --paired-features, one
value-stable and one not, where the four statistics are visible as landscapes
rather than as numbers.  --out-plot is omitted by default and the table is
then written alone.

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

--out-csv and --out-plot are given explicitly and carry no subset tag of their
own, so name the runs apart if both are kept: a second invocation otherwise
overwrites the first with outputs describing a different population.

References
----------
Hampel, F. R. (1974). The influence curve and its role in robust estimation.
    Journal of the American Statistical Association, 69(346), 383-393.
    https://doi.org/10.1080/01621459.1974.10482962

Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median absolute
    deviation. Journal of the American Statistical Association, 88(424),
    1273-1283. https://doi.org/10.1080/01621459.1993.10476408
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

# Fixed unit convention for the common denominator of delta_net and delta_tot:
# the conventional factor for reporting a rescaled MADM (Hampel 1974, as
# 0.6745^-1).  It would make MADM a consistent estimator of the
# standard deviation if the features were normal; they are not, and nothing here
# relies on that.  The constant divides both numerators and so cancels from
# every comparison between them -- it sets the unit, not the result.  Named
# for what it is rather than MAD_TO_SD, which asserted a conversion to a
# standard deviation that does not hold for these features.
MADM_SCALE_CONST = 1.4826


# Tried in order, first match wins. The network set is listed first because
# several of its members would otherwise be claimed by a prefix rule below:
# grd0_mean by gradient, rlf0_mean by elevation. Both are properties of the
# zero-order network rather than of the quantity their prefix names.
FAMILIES = [
    ('network',           '#937860', 'P', lambda f: f in {
        'Rb', 'Rl', 'Rb0', 'Rl0', 'n0', 'n_nodes',
        'L0', 'l0_mean', 'path_max', 'rlf0_mean', 'grd0_mean'}),
    ('elevation',         '#4C72B0', 'o', lambda f: (
        f.startswith('Z_') or f.startswith('rlf') or f == 'hyp_int')),
    ('gradient',          '#DD8452', 's', lambda f: f.startswith('grd')),
    ('curvature',         '#55A868', '^', lambda f: f.startswith('crv')),
    ('hilltop curvature', '#C44E52', 'v', lambda f: f.startswith('htcrv')),
]


def _family(feature):
    """Assign a feature to a morphometric family for plotting.

    FAMILIES above is the single place membership is defined, and it is
    where to edit it.  Each entry is (name, colour, marker, test), and the
    tests are tried in order, so the first match wins and a feature matching
    two entries takes the earlier one.  A test is any callable on the
    feature name: a prefix rule for families whose members share a stem, an
    explicit set where they do not.  Order therefore matters whenever an
    explicit member would also satisfy a prefix rule, which is why the
    network set is listed first.

    Every feature in the current set matches a family, so the grey 'other'
    fallback is unused and absent from the legend.  It is retained as a
    guard: a feature added upstream and not placed in FAMILIES would
    otherwise have no style at all.  It warns when it fires, since a silent
    grey marker is easy to miss in a figure with thirty-odd rows.
    """
    for name, colour, marker, test in FAMILIES:
        if test(feature):
            return name, colour, marker
    print(f"  WARNING: '{feature}' matches no entry in FAMILIES; plotted as "
          f"'other'. Add it to FAMILIES to give it a family.")
    return 'other', '#808080', 'X'


def panel_identity(ax, merged, feature, c=None, norm=None, log=False,
                   title=None, fontsize=8, text_y=0.96, name_in_text=False):
    """Draw one feature's per-landscape scatter, with the 1:1 line.

    Each point is one landscape, its value at 10 m against its value at 1 m.
    All four summary statistics are legible here: offset from the line is
    delta_net, spread about it is delta_tot, and curvature or fanning is the
    rho - pearson gap.  They are annotated as well as drawn, so the panel can
    be read against the corresponding row of panels A and B rather than only
    alongside them.

    Returns the scatter handle when `c` was supplied, so that a caller can
    attach one colourbar for several panels.
    """
    x = merged[feature + '_n10'].to_numpy(dtype=float)
    y = merged[feature + '_n01'].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)

    sc = None
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
    ax.set_title(title if title is not None else feature,
                 fontsize=10, loc='left')
    # With the title suppressed the panel would otherwise not say which
    # feature it shows, so the name leads the annotation instead.
    lead = (feature + "\n") if name_in_text else ""
    ax.text(0.04, text_y,
            lead
            + rf"$\rho$={st['rho']:.3f}" + "\n"
            + rf"$r$={st['pearson']:.3f}" + "\n"
            + rf"$\delta_{{\mathrm{{net}}}}$={st['delta_net']:.3f}" + "\n"
            + rf"$\delta_{{\mathrm{{tot}}}}$={st['delta_tot']:.3f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=fontsize,
            color='0.25')
    ax.set_xlabel(r'value at $\pm$10 m')
    ax.set_ylabel(r'value at $\pm$1 m')
    ax.grid(color='0.93', lw=0.6)
    ax.set_axisbelow(True)
    return sc


def _prepare_summary(df):
    """Sort the stability table for plotting and attach family styling.

    Features are ordered by Spearman rho so that each row of the summary
    panels is one feature and the panels can be read across.  Returned
    separately from the drawing functions because all three figures that
    use these panels need the same ordering.
    """
    d = df.dropna(subset=['rho']).copy()
    d = d.sort_values('rho').reset_index(drop=True)
    meta = [_family(f) for f in d['feature']]
    d['fam'] = [m[0] for m in meta]
    d['colour'] = [m[1] for m in meta]
    d['marker'] = [m[2] for m in meta]
    return d


def _family_handles(target, d, **kwargs):
    """Draw the morphometric-family key on a figure or an axis.

    `target` is whichever object should own the legend: the figure, for a
    key in its own band beneath the panels, or an axis, for a key tucked
    into empty space inside one.
    """
    from matplotlib.lines import Line2D
    fams = list(dict.fromkeys(d['fam']))
    lookup = {name: (c, m) for name, c, m, _ in FAMILIES}
    lookup['other'] = ('#808080', 'X')
    return target.legend(
        handles=[Line2D([], [], marker=lookup[f][1], ls='none',
                        color=lookup[f][0], ms=6, label=f) for f in fams],
        **kwargs,
    )


def panel_ordering(ax, d, label_axis=True, title='ordering and shape',
                   xlabel=r'coefficients, $\rho$ and $r$',
                   family_legend=False, label_side='left', legend_top=1.0):
    """Draw the correlation panel: Spearman rho and Pearson r per feature.

    The span between the two markers is rho - r; a wide span means the
    ordering of landscapes survives the change of setting but the shape of
    the relationship does not, which is the signature of a feature that
    cannot be recovered by rescaling.

    Feature labels are drawn only when label_axis, so that a figure placing
    this panel beside others can carry one set of labels.  Those passing
    both stability thresholds are set in bold, so the figure carries the
    classification rather than leaving it to be inferred by eye.

    Colour and marker encode the morphometric family, which is what makes
    the systematic behaviour visible: quantities determined by discrete
    routing decisions separate from averages taken over many cells.
    """
    import matplotlib.pyplot as plt  # noqa: F401  (backend side effects)
    from matplotlib.lines import Line2D

    n = len(d)
    y = np.arange(n)
    has_r = 'pearson' in d.columns
    if has_r:
        ax.hlines(y, d['rho'], d['pearson'].fillna(d['rho']),
                  color='0.78', lw=1.3, zorder=1)
    for i, row in d.iterrows():
        ax.plot(row['rho'], i, marker=row['marker'], ms=6,
                mfc='none', mec=row['colour'], mew=1.3, ls='none', zorder=3)
        if has_r and np.isfinite(row.get('pearson', np.nan)):
            ax.plot(row['pearson'], i, marker=row['marker'], ms=5,
                    color=row['colour'], ls='none', zorder=3)

    ax.axvline(0.0, color='0.45', lw=0.8, ls=':')
    ax.set_yticks(y)
    # With this panel beside a second one that shares the axis, the feature
    # names read better in the gutter between the two: one list serving both
    # panels, each name adjacent to its own row on either side, instead of
    # the eye travelling back across one panel to reach the other.
    if label_side == 'right':
        ax.yaxis.tick_right()
    if label_axis:
        ax.set_yticklabels(d['feature'], fontsize=8)
        flags = (d['stable'] if 'stable' in d.columns
                 else pd.Series(False, index=d.index))
        for tick, colour, ok in zip(ax.get_yticklabels(), d['colour'], flags):
            tick.set_color(colour)
            if bool(ok):
                tick.set_fontweight('bold')
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color='0.92', lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc='left', fontsize=10)
    # legend_top leaves room above the keys for a panel letter set inside
    # the axes; at 1.0 they sit flush with the top as usual.
    marker_key = ax.legend(handles=[
        Line2D([], [], marker='o', ls='none', mfc='none', mec='0.35',
               ms=6, label=r'Spearman $\rho$'),
        Line2D([], [], marker='o', ls='none', color='0.35',
               ms=5, label='Pearson $r$'),
    ], loc='upper left', fontsize=8, frameon=True,
        bbox_to_anchor=(0.0, legend_top))

    # The family key can live inside this panel, directly beneath the marker
    # key, where the upper left is empty for every feature with a high rho.
    # A figure-level key would otherwise claim a band of its own.
    if family_legend:
        ax.add_artist(marker_key)
        _family_handles(ax, d, loc='upper left', fontsize=8, frameon=True,
                        bbox_to_anchor=(0.0, legend_top - 0.075))
    return ax


def panel_displacement(ax, d, delta_net_max=0.1, delta_tot_max=0.1,
                       title='displacement of value',
                       xlabel=r'displacement, $|\delta_{\mathrm{net}}|$ and '
                              r'$\delta_{\mathrm{tot}}$'):
    """Draw the displacement panel: |delta_net| and delta_tot on a log axis.

    Both are divided by the same spread of the feature across landscapes, so
    horizontal position means the same thing for every feature, and the
    connector between the two markers has a reading of its own: its length
    is log10(delta_tot / |delta_net|), the incoherence of the displacement.  A
    short connector is a one-directional bias that preserves ordering; a
    long one is landscape-specific movement that nothing can recalibrate
    away.  delta_net_max and delta_tot_max are drawn as dashed and dotted lines and
    coincide when they are equal.  Features left of the dashed boundary are
    stable in value, not merely in ordering.

    The magnitude of delta_net is what the stability criterion uses, so nothing
    the criterion needs is lost by plotting |delta_net|; the log axis could not
    carry the sign in any case.  But the sign is sometimes the result --
    Rl0 and Rb0 are displaced downward and upward respectively -- so read it
    from the table, where paired_stats keeps it.
    """
    from matplotlib.lines import Line2D

    n = len(d)
    y = np.arange(n)
    floor = 1e-5
    delta_net = d['delta_net'].abs().clip(lower=floor) if 'delta_net' in d else None
    disp = d['delta_tot'].abs().clip(lower=floor) if 'delta_tot' in d else None

    ax.axvspan(floor, min(delta_net_max, delta_tot_max), color='0.93', zorder=0)
    # A single line when the two thresholds coincide, so the figure does not
    # imply a distinction the caption would then have to explain away.
    if np.isclose(delta_net_max, delta_tot_max):
        ax.axvline(delta_net_max, color='0.6', lw=0.8, ls='--', zorder=1)
    else:
        for bound, style in ((delta_net_max, '--'), (delta_tot_max, ':')):
            ax.axvline(bound, color='0.6', lw=0.8, ls=style, zorder=1)

    if delta_net is not None and disp is not None:
        ax.hlines(y, delta_net, disp, color='0.78', lw=1.3, zorder=2)
    for i, row in d.iterrows():
        if delta_net is not None:
            ax.plot(delta_net[i], i, marker=row['marker'], ms=6, mfc='none',
                    mec=row['colour'], mew=1.3, ls='none', zorder=3)
        if disp is not None:
            ax.plot(disp[i], i, marker=row['marker'], ms=5,
                    color=row['colour'], ls='none', zorder=3)

    ax.set_xscale('log')
    # The clip floor only exists to keep zeros on a log axis; letting it set
    # the lower limit would spend several decades on empty space.
    finite = np.concatenate([
        v.to_numpy() for v in (delta_net, disp) if v is not None])
    finite = finite[finite > floor]
    if finite.size:
        ax.set_xlim(finite.min() / 3.0, finite.max() * 3.0)

    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color='0.92', lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc='left', fontsize=10)
    ax.legend(handles=[
        Line2D([], [], marker='o', ls='none', mfc='none', mec='0.35',
               ms=6, label=r'$|\delta_{\mathrm{net}}|$'),
        Line2D([], [], marker='o', ls='none', color='0.35',
               ms=5, label=r'$\delta_{\mathrm{tot}}$'),
    ], loc='upper right', fontsize=8, frameon=True)
    return ax


def robust_scale(a):
    """The between-landscape spread of one feature, 1.4826 * MADM(a).

    MADM is the median absolute deviation from the median,
    median(|a - median(a)|) -- two medians, one to centre the deviations and
    one to aggregate them.

    Non-finite values are dropped first.  Returns NaN for an empty input and
    0.0 where at least half the landscapes share the median value, which the
    callers treat as "no usable scale" rather than dividing by it.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    madm = np.median(np.abs(a - np.median(a)))
    return float(madm * MADM_SCALE_CONST)


def paired_stats(a, b, scale=None):
    """Paired comparison statistics for one feature over a set of landscapes.

    delta_net and delta_tot share one denominator, so that a single threshold
    means the same thing for both and the gap between them carries only the
    incoherence of the displacement.  The denominator is the robust one:
    sd(a) and 1.4826 * MADM(a) differ by up to a factor of five across this
    feature set, and using one for each statistic made their difference part
    real and part an artifact of the reference distribution's tail.

    The two numerators differ only in when the absolute value is taken --
    net movement against total movement across landscapes.

    `scale` may be supplied to override the value computed from `a`.  That is
    for callers needing one yardstick across separate runs: within a
    restricted range a feature's spread shrinks, so statistics scaled by that
    range's own spread are inflated relative to the whole-ensemble values.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return {'rho': np.nan, 'pearson': np.nan, 'delta_net': np.nan,
                'delta_tot': np.nan, 'scale': np.nan, 'n': len(a)}

    rho = stats.spearmanr(a, b).statistic
    r = stats.pearsonr(a, b).statistic if np.std(a) and np.std(b) else np.nan

    _scale = robust_scale(a) if scale is None else scale
    usable = np.isfinite(_scale) and _scale > 0

    delta_net = (np.median(b - a) / _scale) if usable else np.nan
    # delta_tot >= |delta_net| always, since median(|d|) >= |median(d)| and
    # the two share a denominator. Their ratio is the coherence of the
    # displacement; it is not reported as a column, being a ratio of two
    # adjacent columns and already drawn as the connector length in
    # panel B.
    delta_tot = (np.median(np.abs(b - a)) / _scale) if usable else np.nan

    return {'rho': rho, 'pearson': r, 'delta_net': delta_net,
            'delta_tot': delta_tot, 'scale': _scale, 'n': len(a)}


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



def plot_combined(df, merged, features, path, delta_net_max=0.1, delta_tot_max=0.1,
                  colour_by=None, dpi=200, log=False):
    """One figure carrying the per-feature summary and its worked examples.

    Layout is three columns over two rows.  A and B are the summary panels,
    each spanning both rows because they carry one row per feature and are
    therefore tall; C and D are the per-landscape scatters for the two
    features named in `features`, stacked in the third column.

    The pairing is the point of the figure.  A and B assert which
    features are stable using four numbers each and showing no data; C and D
    show what those numbers look like as landscapes, for one feature that
    passes and one that does not.  Reading across from a feature's row in A
    and B to its panel in C or D is what makes the summary legible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    d = _prepare_summary(df)
    features = [f for f in features if f + '_n10' in merged.columns]
    if len(features) < 2:
        return None
    features = features[:2]

    height = max(6.0, 0.235 * len(d))
    fig = plt.figure(figsize=(14.0, height))
    # The third column is widened because its panels hold a 1:1 aspect and
    # are therefore sized by whichever of width or height is scarcer; too
    # narrow a column leaves them floating in vertical whitespace.
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.30, hspace=0.18,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2])

    panel_ordering(ax_a, d, title='', family_legend=True,
                   label_side='right', legend_top=0.955)
    panel_displacement(ax_b, d, delta_net_max, delta_tot_max, title='')
    # sharey keeps the two summary panels aligned; only A carries the labels.
    plt.setp(ax_b.get_yticklabels(), visible=False)

    c = None
    norm = None
    if colour_by and colour_by in merged.columns:
        c = merged[colour_by].to_numpy(dtype=float)
        norm = LogNorm(vmin=np.nanmin(c[c > 0]), vmax=np.nanmax(c)) \
            if np.all(c[np.isfinite(c)] > 0) else None

    sc = None
    for ax, f in ((ax_c, features[0]), (ax_d, features[1])):
        sc = panel_identity(ax, merged, f, c=c, norm=norm, log=log,
                            title='', text_y=0.90,
                            name_in_text=True) or sc
    if sc is not None:
        cb = fig.colorbar(sc, ax=[ax_c, ax_d], shrink=0.85, pad=0.02)
        cb.set_label(colour_by)

    # The letters are the panels' only titles, set inside the axes at the
    # upper left.  The keys in A and the annotations in C and D are pushed
    # down to clear them.
    for ax, letter in ((ax_a, 'A'), (ax_b, 'B'),
                       (ax_c, 'C'), (ax_d, 'D')):
        ax.text(0.025, 0.978, letter, transform=ax.transAxes,
                ha='left', va='top', fontsize=13, fontweight='bold')

    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


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
                        '--subset.')
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
    p.add_argument('--out-csv', required=True, dest='out_csv',
                   help='Output CSV path.')
    p.add_argument('--out-plot', default=None, dest='out_plot',
                   help='Path for the figure (e.g. .png or .pdf): panels A '
                        'and B, the per-feature summary, beside panels C '
                        'and D, the per-landscape scatters of the two '
                        '--paired-features. Omitted by default.')
    p.add_argument('--delta-net-max', type=float, default=0.1,
                   help='Value-stable boundary for |delta_net|, in units of '
                        'the common scale 1.4826*MADM(x10). Default 0.1.')
    p.add_argument('--paired-features', nargs='+',
                   default=('Z_mean', 'Rl0'),
                   help='The two features drawn in panels C and D. Default '
                        'Z_mean Rl0: one value-stable feature against one '
                        'that is not, so the contrast panels A and B assert '
                        'is visible directly. Rl0 is the instructive failure '
                        'because its ordering survives (rho near 1) while '
                        'its values are displaced; Rb0 is the harder case, '
                        'where the ordering itself inverts.')
    p.add_argument('--paired-colour-by', default=None,
                   help='Label column to colour panels C and D by. Off by '
                        'default, including under --restrict-by: the figure '
                        'is a statement about features, and a parameter '
                        'colour map adds a dimension the comparison does '
                        'not turn on.')
    p.add_argument('--paired-log', action='store_true',
                   help='Log axes on panels C and D.')
    p.add_argument('--restrict-by', default=None,
                   help="Label column to restrict the comparison on, e.g. "
                        "'kh_ks' or 'u_ks'. Used with --range.")
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
    p.add_argument('--delta-tot-max', type=float, default=0.1,
                   help='Value-stable boundary for delta_tot, in the same '
                        'units as delta_net: the two share a denominator, '
                        'so one threshold means the same thing for both. '
                        'Default 0.1.')
    args = p.parse_args()

    print("Loading 10 m ensemble...")
    df10 = load_features(args.n10_features_dir, args.job_ids,
                         args.n10_features_hash)
    print("Loading 1 m ensemble...")
    df01 = load_features(args.n01_features_dir, args.job_ids,
                         args.n01_features_hash)

    # The pkl supplies the split identities, which already name none but the
    # landscapes its models were trained on. No steady-state manifest is
    # read: applying one would change nothing for --subset train/test while
    # making the run depend on that path still resolving. The manifest
    # recorded in _meta is reported as provenance only.
    meta = read_models_meta(args.models_pkl)
    if meta.get('retained_csv') and args.subset != 'all':
        print(f"Split identities describe the steady-state subset of "
              f"{os.path.basename(meta['retained_csv'])}.")

    df10 = subset_frame(df10, meta, args.subset, args.models_pkl)
    df01 = subset_frame(df01, meta, args.subset, args.models_pkl)

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

    # Label columns carried through the merge: the one restricted on, and
    # the one panels C and D are coloured by. They are independent --
    # --restrict-by does not imply a colour map -- so both are listed.
    carry = []
    for col in (args.restrict_by, args.paired_colour_by):
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
    # Scales are taken from the unrestricted ensemble and held fixed through
    # the restriction, so that a restricted run's delta_net and delta_tot are
    # readable against the whole-ensemble table. Without this a narrow range
    # shrinks every feature's spread and inflates both statistics for a
    # reason that has nothing to do with the observational setting. rho is
    # not protected this way and remains comparable only within its own run.
    scales = None
    tag = ''
    if args.range:
        scales = {f: robust_scale(merged[f + '_n10'].to_numpy(dtype=float))
                  for f in feat_cols}
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
        rows.append({'feature': f,
                     **paired_stats(a, b,
                                    scale=scales[f] if scales else None)})

    out = pd.DataFrame(rows).sort_values('rho', ascending=False)
    # Recorded in the table as well as drawn in the figure, so that the
    # classification has one definition and 04b does not re-derive it.
    # A NaN on either statistic compares False and so is not called stable.
    out['stable'] = ((out['delta_net'].abs() < args.delta_net_max)
                     & (out['delta_tot'] < args.delta_tot_max))

    def tagged(path):
        """Insert the restriction tag before the extension, so a restricted
        run cannot silently overwrite an unrestricted one."""
        if not tag:
            return path
        root, ext = os.path.splitext(path)
        return root + tag + ext

    out_path = tagged(args.out_csv)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    out.to_csv(out_path, index=False)

    pd.set_option('display.width', 120)
    print(f"\nSubset: {args.subset}\n")
    print(out.to_string(index=False,
                        float_format=lambda v: f'{v: .4f}'))
    print(f"\nWrote {out_path}")

    if args.out_plot:
        p = plot_combined(out, merged, list(args.paired_features),
                          tagged(args.out_plot),
                          delta_net_max=args.delta_net_max,
                          delta_tot_max=args.delta_tot_max,
                          colour_by=args.paired_colour_by,
                          log=args.paired_log)
        if p:
            print(f"Wrote {p}")
        else:
            print(f"WARNING: fewer than two of "
                  f"{list(args.paired_features)} found in the feature "
                  f"table; no figure written.")


if __name__ == '__main__':
    main()
