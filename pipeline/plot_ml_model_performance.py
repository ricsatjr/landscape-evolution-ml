#!/usr/bin/env python3
"""Generate the ML performance figure for the JGR:ES manuscript.

Layout
------
One row per results pickle, one panel per target label within a row.  Score
($R^2$) runs along x; algorithms run down y in a fixed order.

For every algorithm and target the panel shows

  * the distribution of outer-fold $R^2$ scores plus their mean — the
    generalized performance estimate, drawn as the dominant mark; and
  * the held-out test-set $R^2$ — a single confirmatory evaluation, drawn as a
    secondary mark in a contrasting hue.

Both measures appear for ALL algorithms, so no selection-by-performance is
implied and no test-set filtering occurs.  Algorithms are drawn in a fixed
order (roughly linear -> kernel -> tree -> neural), never sorted by score, so
the panel cannot be read as a leaderboard and the axis encodes the diversity
of inductive biases that makes cross-algorithm agreement meaningful.

``--scale-mode`` controls panel widths: ``equal-width`` (default) draws every
panel at the same physical width regardless of its range, so a two-panel row
is narrower than a three-panel row and, when centred, straddles the joins of
the row below; ``equal-units`` instead fixes the data units per unit width, so
range determines width; ``full-width`` lets each row fill the figure.

Examples
--------
Ratios and individual parameters, full page width::

    python plot_cv_performance.py \\
        --pkl nested-cv-results-full-u_ks-kh_ks-<hash>.pkl \\
        --pkl nested-cv-results-full-u-kh-ks-<hash>.pkl \\
        --row-xlim 0.90 1.00 --row-xlim -0.10 0.20 \\
        --width-cm 19 --height-cm 11 \\
        --out fig-nested-cv-results.pdf
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

# ---------------------------------------------------------------------------
# Defaults — all overridable from the command line
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ORDER = ['lin', 'lso', 'knn', 'svm', 'dtr', 'rfo', 'gbs', 'mlp']

DEFAULT_TARGET_LABELS = {
    'u_ks':  r'$\log U/K_s$',
    'kh_ks': r'$\log K_h/K_s$',
    'u':     r'$\log U$',
    'kh':    r'$\log K_h$',
    'ks':    r'$\log K_s$',
}

DEFAULT_MODEL_LABELS = {m: m for m in DEFAULT_MODEL_ORDER}

CM_PER_INCH = 2.54


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_results(paths):
    """Load one or more results pickles from ``03_train_models.py``."""
    runs = []
    for p in paths:
        with warnings.catch_warnings():
            # Unpickling fitted estimators can warn about sklearn version skew.
            # Only arrays and floats are read here; predict() is never called.
            warnings.simplefilter('ignore')
            with open(p, 'rb') as fh:
                d = pickle.load(fh)
        if '_meta' not in d:
            raise ValueError(f"{p}: no '_meta' key — not a 03_ results pickle?")
        runs.append({'path': Path(p),
                     'meta': d['_meta'],
                     'models': {k: v for k, v in d.items() if k != '_meta'}})
    return runs


def collect_rows(runs, model_order, targets_filter=None):
    """Flatten runs into rows of panel specifications (one row per pickle)."""
    rows = []
    for run in runs:
        labels = list(run['meta']['label_names'])
        panels = []
        for t_idx, target in enumerate(labels):
            if targets_filter and target not in targets_filter:
                continue
            cv, test = {}, {}
            for m in model_order:
                if m not in run['models']:
                    continue
                res = run['models'][m]
                pt = res.get('per_target_r2')
                if pt is None:
                    raise KeyError(f"{run['path'].name}/{m}: 'per_target_r2' missing")
                cv[m] = np.asarray([np.asarray(f)[t_idx] for f in pt], dtype=float)

                fm = res.get('final_model') or {}
                ts = fm.get('test_set_r2')          # [t0, t1, ..., overall]
                test[m] = (float(ts[t_idx])
                           if ts is not None and t_idx < len(ts) - 1 else None)
            panels.append({'target': target, 'cv': cv, 'test': test,
                           'source': run['path'].name})
        if panels:
            rows.append(panels)
    return rows


def collect_prediction_row(runs, model, data_dir, features_hash, job_ids):
    """Build one true-vs-predicted panel per target for a single algorithm.

    The results pickle stores the fitted pipeline but not the data, so the
    feature table is reloaded and the stored ``test_idx`` re-selected.  Targets
    are log10-transformed exactly as ``split_features_labels`` does, so the
    axes match the units the model was trained on.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline_utils import load_features, split_features_labels

    groups = []
    for run in runs:
        panels = []
        meta = run['meta']
        if model not in run['models']:
            raise KeyError(f"{run['path'].name}: no '{model}' entry")
        reg = (run['models'][model].get('final_model') or {}).get('regressor')
        if reg is None:
            raise KeyError(f"{run['path'].name}/{model}: no fitted regressor")

        # NOTE: _meta['job_ids'] holds job_id *column values*, not file
        # identifiers, so it must not be forwarded here. 03_ loaded with
        # job_ids=None; matching that keeps test_idx aligned.
        df = load_features(data_dir, job_ids=job_ids,
                           features_hash=features_hash
                           or meta.get('features_hash'))
        X, y = split_features_labels(df, list(meta['label_names']))
        test_idx = meta['test_idx']
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        y_pred = np.asarray(reg.predict(X_test))

        for t_idx, target in enumerate(meta['label_names']):
            panels.append({'kind': 'pred', 'target': target,
                           'true': np.asarray(y_test.iloc[:, t_idx]),
                           'pred': y_pred[:, t_idx],
                           'model': model, 'source': run['path'].name})
        groups.append(panels)
    return groups


def draw_pred_panel(ax, panel, args, show_ylabel, show_xlabel):
    """Scatter of predicted against true label values for one target."""
    from matplotlib.ticker import MaxNLocator
    t, p = panel['true'], panel['pred']
    lo = min(t.min(), p.min())
    hi = max(t.max(), p.max())
    pad = (hi - lo) * 0.06 or 0.1
    lo, hi = lo - pad, hi + pad

    ax.plot([lo, hi], [lo, hi], ls='--', lw=0.7,
            color=args.pred_line_color, zorder=1)
    ax.plot(t, p, linestyle='none', marker='o', ms=args.pred_marker_size,
            mfc=args.pred_color, mec='none', alpha=args.pred_alpha, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(DEFAULT_TARGET_LABELS.get(panel['target'], panel['target']),
                 fontsize=args.title_size, fontweight=args.title_weight,
                 pad=args.title_pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=args.pred_max_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=args.pred_max_ticks))
    if args.pred_r2:
        # Computed from the plotted arrays, so the annotation always matches
        # the scatter above it.  Identical to the stored held-out test R^2.
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        ax.text(args.pred_r2_x, args.pred_r2_y,
                args.pred_r2_format.format(r2=r2),
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=args.pred_r2_size, color=args.pred_r2_color)

    if show_ylabel:
        ax.set_ylabel(args.pred_ylabel, fontsize=args.label_size)
    if show_xlabel:
        ax.set_xlabel(args.pred_xlabel, fontsize=args.label_size)
    if args.grid:
        ax.grid(color='0.90', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=args.tick_size, length=2.5, width=0.6)


def resolve_row_limits(rows, row_xlim, pad_frac):
    """Return one (lo, hi) tuple per row, from CLI or autoscaled with padding."""
    lims = []
    for r, row in enumerate(rows):
        perf = [p for p in row if p.get('kind') != 'pred']
        if not perf:
            lims.append(None)
            continue
        if row_xlim and r < len(row_xlim) and row_xlim[r] is not None:
            lims.append(tuple(row_xlim[r]))
            continue
        vals = []
        for panel in perf:
            for s in panel['cv'].values():
                vals.extend(s.tolist())
            vals.extend([t for t in panel['test'].values() if t is not None])
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * pad_frac or 0.01
        lims.append((lo - pad, hi + pad))
    return lims


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_panel(ax, panel, model_order, args, xlim, show_yticklabels, nbins,
               show_ylabel=False, show_xlabel=False):
    """Draw one target panel: score on x, algorithms on y (top to bottom)."""
    from matplotlib.ticker import MaxNLocator

    rng = np.random.default_rng(args.jitter_seed)
    models = [m for m in model_order if m in panel['cv']]
    n = len(models)

    for i, m in enumerate(models):
        scores = panel['cv'][m]
        y = n - 1 - i                       # first model at the top
        jit = (rng.uniform(-args.jitter, args.jitter, scores.size)
               if args.jitter > 0 else np.zeros(scores.size))

        ax.plot(scores, np.full(scores.size, y) + jit,
                linestyle='none', marker='o', ms=args.cv_marker_size,
                mfc=args.cv_color, mec='none', alpha=args.cv_alpha, zorder=2)

        mu, hw = scores.mean(), args.mean_bar_halfwidth
        ax.plot([mu, mu], [y - hw, y + hw], color=args.mean_color,
                lw=args.mean_bar_lw, solid_capstyle='butt', zorder=4)

        t = panel['test'].get(m)
        if t is not None and not args.no_test:
            ax.plot(t, y, marker=args.test_marker, ms=args.test_marker_size,
                    mfc=args.test_color, mec=args.test_edge_color,
                    mew=args.test_marker_lw, linestyle='none', zorder=3)

    names = [DEFAULT_MODEL_LABELS.get(m, m) for m in models][::-1]
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(names if show_yticklabels else [])
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=nbins, prune=args.tick_prune))

    ax.set_title(DEFAULT_TARGET_LABELS.get(panel['target'], panel['target']),
                 fontsize=args.title_size, fontweight=args.title_weight,
                 pad=args.title_pad)
    if show_ylabel:
        ax.set_ylabel(args.xlabel, fontsize=args.label_size)
    if show_xlabel:
        ax.set_xlabel(args.ylabel, fontsize=args.label_size)
    if args.grid:
        ax.grid(axis='x', color='0.90', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=args.tick_size, length=2.5, width=0.6)


def legend_artists(args):
    """Proxy handles so the CV-mean key is a vertical bar, as in the panels."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], linestyle='none', marker='o',
               ms=args.legend_cv_size, mfc=args.cv_color, mec='none',
               alpha=args.cv_alpha),
        # '|' renders a vertical tick, matching the mean bar in the panels
        Line2D([], [], linestyle='none', marker='|',
               ms=args.legend_mean_size, color=args.mean_color,
               mew=args.mean_bar_lw),
        Line2D([], [], linestyle='none', marker=args.test_marker,
               ms=args.legend_test_size, mfc=args.test_color,
               mec=args.test_edge_color, mew=args.test_marker_lw),
    ]
    labels = ['outer-fold CV score', 'CV mean (generalized)',
              'held-out test (confirmatory)']
    return handles, labels


def build_figure(rows, args):
    """One GridSpec per row, so rows can differ in panel count and width."""
    import matplotlib.pyplot as plt

    figsize = (args.width_cm / CM_PER_INCH, args.height_cm / CM_PER_INCH)
    fig = plt.figure(figsize=figsize)

    xlims = resolve_row_limits(rows, args.row_xlim, args.autoscale_pad)
    spans = [(hi - lo) if lim else 1.0 for lim in xlims
             for lo, hi in [lim or (0, 1)]]
    avail = args.right - args.left
    counts = [len(r) for r in rows]

    if args.scale_mode == 'equal-units':
        # A given delta R^2 spans the same physical length in every panel;
        # rows covering less range are drawn narrower.
        totals = [spans[r] * counts[r] for r in range(len(rows))]
        row_width = [avail * t / max(totals) for t in totals]
    elif args.scale_mode == 'equal-width':
        # Every panel is the same physical width whatever its range; a row
        # with fewer panels is narrower than one with more.
        n_max = max(counts)
        g = args.wspace
        pw = avail / (n_max + (n_max - 1) * g)
        row_width = [n * pw + (n - 1) * pw * g for n in counts]
    else:                                   # 'full-width'
        row_width = [avail] * len(rows)

    # Vertical budget: total height minus the inter-row gaps, divided
    # between rows in proportion to --row-height-ratios.  Square prediction
    # panels grow with row height until they hit their slot width, so give
    # those rows a larger share.
    n_rows = len(rows)
    total_h = args.top - args.bottom
    gap = args.row_gap * total_h / n_rows if n_rows > 1 else 0.0
    # Optional extra separation after specific rows, e.g. between the
    # performance block and the prediction block.
    extra = [0.0] * max(n_rows - 1, 0)
    for idx in (args.extra_gap_after or []):
        if 1 <= idx <= n_rows - 1:
            extra[idx - 1] += args.extra_gap * total_h / n_rows
    usable = total_h - gap * (n_rows - 1) - sum(extra)
    if args.row_height_ratios:
        rr = list(args.row_height_ratios)
        if len(rr) != n_rows:
            raise SystemExit(f'--row-height-ratios needs {n_rows} values, '
                             f'got {len(rr)}')
    else:
        rr = [1.0] * n_rows
    row_heights = [usable * v / sum(rr) for v in rr]
    row_tops = []
    y = args.top
    for i, h in enumerate(row_heights):
        row_tops.append(y)
        y -= h + gap + (extra[i] if i < len(extra) else 0.0)
    axes_grid = []

    # Which row, if any, hosts the legend in its spare grid slot?
    n_max = max(counts)
    host = None
    if args.legend and args.legend_slot == 'auto':
        for r, n in enumerate(counts):
            if n < n_max:
                host = r
                break

    kinds = [('pred' if row[0].get('kind') == 'pred' else 'perf') for row in rows]
    last_of = {k: max(i for i, kk in enumerate(kinds) if kk == k)
               for k in set(kinds)}

    for r, row in enumerate(rows):
        n_cols = n_max if r == host else len(row)
        w = avail if r == host else row_width[r]
        left = args.left if args.row_align == 'left' else args.left + (avail - w) / 2
        top = row_tops[r]
        gs = fig.add_gridspec(
            1, n_cols,
            left=left, right=left + w,
            top=top,
            bottom=top - row_heights[r],
            wspace=args.wspace,
        )
        if r == host:
            lax = fig.add_subplot(gs[0, len(row):])
            lax.axis('off')
            h, l = legend_artists(args)
            lax.legend(h, l, loc=args.legend_slot_loc,
                       ncol=args.legend_slot_ncol, fontsize=args.legend_size,
                       frameon=False, handletextpad=args.legend_handletextpad,
                       columnspacing=args.legend_columnspacing,
                       labelspacing=args.legend_labelspacing,
                       borderaxespad=0.0)
        row_axes = []
        for c, panel in enumerate(row):
            ax = fig.add_subplot(gs[0, c])
            if panel.get('kind') == 'pred':
                draw_pred_panel(ax, panel, args,
                                show_ylabel=(c == 0 and args.axis_labels),
                                show_xlabel=(r == last_of['pred']
                                             and args.axis_labels))
            else:
                nb = (args.row_max_xticks[r]
                      if args.row_max_xticks and r < len(args.row_max_xticks)
                      else args.max_xticks)
                draw_panel(ax, panel, args.models, args, xlims[r],
                           show_yticklabels=(c == 0), nbins=nb,
                           show_ylabel=(c == 0 and args.axis_labels),
                           show_xlabel=(r == last_of['perf']
                                        and args.axis_labels))
            row_axes.append(ax)
        axes_grid.append(row_axes)

    if args.panel_letters:
        letters = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        for row_axes in axes_grid:
            for ax in row_axes:
                ax.text(args.letter_x, args.letter_y, next(letters),
                        transform=ax.transAxes, fontsize=args.letter_size,
                        fontweight='bold', va='bottom', ha='left')

    # Per-row labels and figure-level super-labels would duplicate each other,
    # so --axis-labels suppresses the latter.
    if args.supxlabel and not args.axis_labels:
        fig.supxlabel(args.supxlabel, fontsize=args.label_size, y=args.supxlabel_y)
    if args.supylabel and not args.axis_labels:
        fig.supylabel(args.supylabel, fontsize=args.label_size, x=args.supylabel_x)

    if args.legend and host is None:
        h, l = legend_artists(args)
        fig.legend(h, l, loc=args.legend_loc, ncol=args.legend_ncol,
                   fontsize=args.legend_size, frameon=False,
                   handletextpad=args.legend_handletextpad,
                   columnspacing=args.legend_columnspacing,
                   labelspacing=args.legend_labelspacing,
                   bbox_to_anchor=tuple(args.legend_bbox))
    return fig


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(rows, model_order):
    for row in rows:
        for p in row:
            if p.get('kind') == 'pred':
                continue
            print(f"\n-- target: {p['target']}   ({p['source']})")
            print(f"  {'model':6s} {'CV mean':>9s} {'CV sd':>8s} {'CV min':>8s} "
                  f"{'CV max':>8s} {'test':>9s}")
            for m in model_order:
                if m not in p['cv']:
                    continue
                s, t = p['cv'][m], p['test'].get(m)
                tstr = f'{t:9.4f}' if t is not None else f'{"n/a":>9s}'
                print(f"  {m:6s} {s.mean():9.4f} {s.std():8.4f} {s.min():8.4f} "
                      f"{s.max():8.4f} {tstr}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    io = p.add_argument_group('input / output')
    io.add_argument('--pkl', action='append', required=True, metavar='PATH',
                    help='Results pickle from 03_train_models.py. Repeatable; '
                         'each becomes one row, in the order given.')
    io.add_argument('--out', default='fig-nested-cv-results.pdf',
                    help='Output path; extension sets the format. [%(default)s]')
    io.add_argument('--dpi', type=int, default=600,
                    help='Raster resolution for png/tif. [%(default)s]')
    io.add_argument('--targets', nargs='*', default=None, metavar='NAME',
                    help='Restrict to these target labels.')
    io.add_argument('--pred-model', default=None, metavar='NAME',
                    help='Append a true-vs-predicted row for this algorithm '
                         "(e.g. 'mlp'), one panel per target. Requires "
                         '--data-dir, since the pickle stores the fitted '
                         'pipeline but not the data.')
    io.add_argument('--data-dir', default='data/features',
                    help='Directory of features-*.pkl, as passed to 03_. '
                         '[%(default)s]')
    io.add_argument('--features-hash', default=None,
                    help='Git hash of the feature files; defaults to the value '
                         'recorded in the pickle.')
    io.add_argument('--job-ids', nargs='+', default=None,
                    help='Restrict feature loading to these job ids; defaults '
                         'to those recorded in the pickle.')
    io.add_argument('--summary', action='store_true',
                    help='Print the numeric summary table to stdout.')

    lay = p.add_argument_group('layout')
    lay.add_argument('--width-cm', type=float, default=19.0,
                     help='Figure width in cm. AGU: 8.4 single, 19 full. [%(default)s]')
    lay.add_argument('--height-cm', type=float, default=11.0,
                     help='Figure height in cm. [%(default)s]')
    lay.add_argument('--row-xlim', action='append', nargs=2, type=float,
                     default=None, metavar=('LO', 'HI'),
                     help='Score-axis limits for one row; repeat once per '
                          'pickle, in order. Omit to autoscale.')
    lay.add_argument('--autoscale-pad', type=float, default=0.05,
                     help='Fractional padding when a row autoscales. [%(default)s]')
    lay.add_argument('--scale-mode',
                     choices=['equal-width', 'equal-units', 'full-width'],
                     default='equal-width',
                     help="'equal-width': every panel the same physical width "
                          'whatever its range, so a 2-panel row is narrower '
                          "than a 3-panel row. 'equal-units': same data units "
                          'per unit width everywhere, so range sets the width. '
                          "'full-width': each row fills the figure. [%(default)s]")
    lay.add_argument('--row-align', choices=['left', 'center'], default='left',
                     help='Placement of a row narrower than the figure. '
                          'With equal-width + center, a 2-panel row straddles '
                          'the joins of the 3-panel row below. [%(default)s]')
    lay.add_argument('--wspace', type=float, default=0.12,
                     help='Horizontal gap between panels within a row. [%(default)s]')
    lay.add_argument('--extra-gap-after', nargs='*', type=int, default=None,
                     metavar='ROW',
                     help='Insert extra vertical separation after these '
                          '1-based row numbers, e.g. 2 to separate the '
                          'performance rows from the prediction rows.')
    lay.add_argument('--extra-gap', type=float, default=0.55,
                     help="Size of that extra separation, as a fraction of a "
                          "row's height. [%(default)s]")
    lay.add_argument('--row-height-ratios', nargs='*', type=float, default=None,
                     metavar='R',
                     help='Relative height of each row, one value per row '
                          "(e.g. '1 1 1.7 1.7' to enlarge prediction rows). "
                          'Default: all equal.')
    lay.add_argument('--row-gap', type=float, default=0.42,
                     help='Vertical gap between rows, as a fraction of one '
                          "row's height. Raise it to give the lower row's "
                          'panel titles more clearance. [%(default)s]')
    lay.add_argument('--left', type=float, default=0.075)
    lay.add_argument('--right', type=float, default=0.985)
    lay.add_argument('--top', type=float, default=0.945)
    lay.add_argument('--bottom', type=float, default=0.165)

    st = p.add_argument_group('style - outer-fold CV points')
    st.add_argument('--cv-color', default='#A8C0D6')
    st.add_argument('--cv-marker-size', type=float, default=3.0)
    st.add_argument('--cv-alpha', type=float, default=0.85)
    st.add_argument('--jitter', type=float, default=0.10,
                    help='Vertical jitter half-width; 0 disables. [%(default)s]')
    st.add_argument('--jitter-seed', type=int, default=42,
                    help='Seed for jitter, so the figure is reproducible. [%(default)s]')

    sm = p.add_argument_group('style - CV mean bar')
    sm.add_argument('--mean-color', default='#1F4E79')
    sm.add_argument('--mean-bar-lw', type=float, default=2.0)
    sm.add_argument('--mean-bar-halfwidth', type=float, default=0.26)

    tt = p.add_argument_group('style - held-out test marker')
    tt.add_argument('--test-marker', default='D')
    tt.add_argument('--test-marker-size', type=float, default=3.6)
    tt.add_argument('--test-color', default='none')
    tt.add_argument('--test-edge-color', default='#C1663B')
    tt.add_argument('--test-marker-lw', type=float, default=1.1)
    tt.add_argument('--no-test', action='store_true')

    pr = p.add_argument_group('style - true-vs-predicted row')
    pr.add_argument('--pred-color', default='#8A8A8A',
                    help='Grayscale by default, so the prediction panels read '
                         'as a different kind of plot. [%(default)s]')
    pr.add_argument('--pred-marker-size', type=float, default=2.0)
    pr.add_argument('--pred-alpha', type=float, default=0.30)
    pr.add_argument('--pred-line-color', default='#3A3A3A')
    pr.add_argument('--pred-r2', dest='pred_r2', action='store_true',
                    default=True,
                    help='Annotate each prediction panel with its held-out '
                         'test R^2. [on]')
    pr.add_argument('--no-pred-r2', dest='pred_r2', action='store_false')
    pr.add_argument('--pred-r2-format', default=r'$R^2 = {r2:.3f}$',
                    help='Format string; {r2} is substituted. [%(default)s]')
    pr.add_argument('--pred-r2-x', type=float, default=0.96,
                    help='Annotation x in axes fraction, right-aligned. [%(default)s]')
    pr.add_argument('--pred-r2-y', type=float, default=0.05,
                    help='Annotation y in axes fraction, bottom-aligned. [%(default)s]')
    pr.add_argument('--pred-r2-size', type=float, default=7.5)
    pr.add_argument('--pred-r2-color', default='#1F1F1F')
    pr.add_argument('--pred-xlabel', default='true')
    pr.add_argument('--pred-ylabel', default='predicted')
    pr.add_argument('--pred-max-ticks', type=int, default=4)
    pr.add_argument('--pred-layout', choices=['inline', 'newrow'],
                    default='newrow',
                    help="'inline': prediction panels follow the performance "
                         'panels in the same row, so the shorter row leaves '
                         "spare slots for the legend. 'newrow': a separate "
                         'trailing row. [%(default)s]')

    tx = p.add_argument_group('style - text')
    tx.add_argument('--font-family', default='sans-serif')
    tx.add_argument('--font-size', type=float, default=8.0,
                    help='Base font size in pt; AGU minimum is 8. [%(default)s]')
    tx.add_argument('--tick-size', type=float, default=7.0)
    tx.add_argument('--label-size', type=float, default=9.0)
    tx.add_argument('--title-size', type=float, default=10.0)
    tx.add_argument('--title-weight', default='bold')
    tx.add_argument('--title-pad', type=float, default=5.0)
    tx.add_argument('--max-xticks', type=int, default=5,
                    help='Maximum score-axis ticks per panel. [%(default)s]')
    tx.add_argument('--row-max-xticks', action='append', type=int, default=None,
                    metavar='N',
                    help='Per-row tick cap; repeat once per pickle, in order. '
                         'Narrow rows need fewer. Falls back to --max-xticks.')
    tx.add_argument('--tick-prune', default=None,
                    choices=[None, 'lower', 'upper', 'both'],
                    help='Drop an end tick so neighbouring panels do not collide.')
    tx.add_argument('--xlabel', default='algorithm',
                    help='Axis name for the algorithm axis, used with '
                         '--axis-labels. [%(default)s]')
    tx.add_argument('--ylabel', default=r'$R^2$',
                    help='Axis name for the score axis, used with '
                         '--axis-labels. [%(default)s]')
    tx.add_argument('--supxlabel', default=r'$R^2$',
                    help="Figure-level x label; '' to omit. Ignored when "
                         '--axis-labels is set. [%(default)s]')
    tx.add_argument('--supylabel', default='',
                    help="Figure-level y label; '' to omit. Ignored when "
                         '--axis-labels is set. [%(default)s]')
    tx.add_argument('--axis-labels', dest='axis_labels', action='store_true',
                    default=False,
                    help='Label each row individually instead of using figure-'
                         'level super-labels. Needed when performance and '
                         'prediction rows are mixed, since their axes differ.')
    tx.add_argument('--supxlabel-y', type=float, default=0.085)
    tx.add_argument('--supylabel-x', type=float, default=0.012)

    lg = p.add_argument_group('style - legend, letters, grid')
    lg.add_argument('--legend', dest='legend', action='store_true', default=True)
    lg.add_argument('--no-legend', dest='legend', action='store_false')
    lg.add_argument('--legend-loc', default='lower center')
    lg.add_argument('--legend-ncol', type=int, default=3)
    lg.add_argument('--legend-size', type=float, default=7.5)
    lg.add_argument('--legend-bbox', nargs=2, type=float, default=[0.5, 0.0],
                    metavar=('X', 'Y'))
    lg.add_argument('--legend-cv-size', type=float, default=3.6,
                    help='Marker size of the CV-point key. [%(default)s]')
    lg.add_argument('--legend-mean-size', type=float, default=8.0,
                    help='Height of the vertical CV-mean bar key. [%(default)s]')
    lg.add_argument('--legend-test-size', type=float, default=4.2,
                    help='Marker size of the held-out test key. [%(default)s]')
    lg.add_argument('--legend-slot', choices=['auto', 'figure', 'none'],
                    default='auto',
                    help="'auto': place the legend in the spare grid slot of "
                         'the first row that has fewer panels than the widest '
                         "row, falling back to a figure legend. 'figure': "
                         'always below the figure. [%(default)s]')
    lg.add_argument('--legend-slot-loc', default='center left',
                    help='Legend anchor within its grid slot. [%(default)s]')
    lg.add_argument('--legend-slot-ncol', type=int, default=1,
                    help='Columns when the legend sits in a grid slot. [%(default)s]')
    lg.add_argument('--legend-labelspacing', type=float, default=0.9)
    lg.add_argument('--legend-handletextpad', type=float, default=0.6)
    lg.add_argument('--legend-columnspacing', type=float, default=2.2)
    lg.add_argument('--panel-letters', dest='panel_letters', action='store_true',
                    default=True)
    lg.add_argument('--no-panel-letters', dest='panel_letters', action='store_false')
    lg.add_argument('--letter-size', type=float, default=9.5)
    lg.add_argument('--letter-x', type=float, default=-0.015)
    lg.add_argument('--letter-y', type=float, default=1.04)
    lg.add_argument('--grid', dest='grid', action='store_true', default=True)
    lg.add_argument('--no-grid', dest='grid', action='store_false')

    md = p.add_argument_group('models')
    md.add_argument('--models', nargs='*', default=DEFAULT_MODEL_ORDER,
                    help='Algorithms and their top-to-bottom order. The default '
                         'runs linear -> kernel -> tree -> neural and is '
                         'deliberately NOT sorted by score. [%(default)s]')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': args.font_family,
        'font.size': args.font_size,
        'axes.linewidth': 0.7,
        'pdf.fonttype': 42,     # embed TrueType; required by AGU
        'ps.fonttype': 42,
    })

    runs = load_results(args.pkl)
    rows = collect_rows(runs, args.models, args.targets)
    if args.pred_model:
        groups = collect_prediction_row(
            runs, args.pred_model, args.data_dir,
            args.features_hash, args.job_ids)
        if args.pred_layout == 'inline':
            for i, g in enumerate(groups):
                if i < len(rows):
                    rows[i] = rows[i] + g
                else:
                    rows.append(g)
        else:
            for g in groups:
                rows.append(g)
    if not rows:
        sys.exit('No panels to draw - check --pkl and --targets.')

    if args.summary:
        print_summary(rows, args.models)

    fig = build_figure(rows, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f'\nWrote {out}  ({args.width_cm:g}x{args.height_cm:g} cm, '
          f'{len(rows)} row(s), {sum(len(r) for r in rows)} panel(s))')
    return 0


if __name__ == '__main__':
    sys.exit(main())
