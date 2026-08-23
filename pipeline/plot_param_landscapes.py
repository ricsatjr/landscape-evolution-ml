#!/usr/bin/env python3
"""Combined parameter-space and sample-landscape figure.

Row 1  A, B   the constrained parameter space: log U against log Ks coloured
              by log U/Ks, and log Kh against log Ks coloured by log Kh/Ks.
              The landscapes drawn below are marked here, so each map can be
              located in the sampled space.
Rows 2+ C...  elevation maps of the selected landscapes, arranged across the
              (log U/Ks, log Kh/Ks) plane.

Landscape selection is by nearest neighbour in log-ratio space, with targets
given as quantiles so the figure adapts if the parameter space changes.

Examples
--------
Default 2 x 2 corners of ratio space::

    python plot_param_landscapes.py \\
        --data-dir outputs/features --rasnet-dir outputs/features \\
        --out fig-param-space-landscapes.pdf --summary

Matched-ratio pairs instead of corners::

    python plot_param_landscapes.py --mode matched --n-pairs 2 \\
        --data-dir outputs/features --rasnet-dir outputs/features \\
        --no-axis-headers --out fig-matched-pairs.pdf
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

CM_PER_INCH = 2.54

AXIS_LABELS = {
    'u':     r'$\log\ U\ (\mathrm{m\ y^{-1}})$',
    'ks':    r'$\log\ K_s\ (\mathrm{y^{-1}})$',
    'kh':    r'$\log\ K_h\ (\mathrm{m^2\ y^{-1}})$',
}
CBAR_LABELS = {
    'u_ks':  r'$\log\ U/K_s$',
    'kh_ks': r'$\log\ K_h/K_s$',
    'u_kh':  r'$\log\ U/K_h$',
}
# Inline form for headers; the fraction form is too tall for a one-line title
# and for a rotated y label.
RATIO_TEX = {'u_ks': r'U/K_s', 'kh_ks': r'K_h/K_s'}
RATIO_TEX_FRAC = {'u_ks': r'\frac{U}{K_s}', 'kh_ks': r'\frac{K_h}{K_s}'}

DEFAULT_PARAM_PANELS = [('ks', 'u', 'u_ks'), ('ks', 'kh', 'kh_ks')]
# A on the left, B on the right, so the two colourbars sit away from the cloud
DEFAULT_PCBAR_RECTS = [[0.09, 0.40, 0.035, 0.30], [0.87, 0.40, 0.035, 0.30]]
DEFAULT_NOTES = [
    (0, 0.28, 0.95, 'relief is\ntoo high'),
    (0, 0.60, 0.30, 'relief is\ntoo low'),
    (1, 0.03, 0.97, 'mostly\nhillslopes'),
    (1, 0.62, 0.24, 'mostly\nstreams'),
]
MARKER_CYCLE = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_table(data_dir, features_hash, job_ids):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline_utils import load_features
    try:
        return load_features(data_dir, job_ids=job_ids,
                             features_hash=features_hash)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(
            f"Could not load feature files from '{data_dir}': {exc}\n"
            'Check --data-dir, and pass --features-hash if several feature '
            'file versions are present.') from exc


def rasnet_path(rasnet_dir, row, elev_err):
    """Reproduce 02_extract_features.py's filename convention."""
    err = int(elev_err if elev_err is not None else row.get('elev_err', 10.0))
    return (Path(rasnet_dir) /
            f"rasnet-n{err}-{row['job_id']}-{int(row['landscape_idx'])}"
            f"-{int(row['ts_index'])}.pkl")


def load_elevation(path, mask_invalid):
    """Return (elevation 2-D array, dx) from a cached rasnet pickle."""
    if not path.exists():
        raise SystemExit(f'rasnet file not found: {path}\n'
                         'Check --rasnet-dir and --elev-err.')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with open(path, 'rb') as fh:
            obj = pickle.load(fh)
    mg, mask = obj[1], obj[2]
    z = np.asarray(mg.at_node['topographic__elevation']).reshape(mg.shape)
    if mask_invalid and mask is not None:
        m = np.asarray(mask)
        if m.shape == z.shape:
            z = np.where(m, z, np.nan)
    return z, float(mg.dx)


def nearest_row(lx, ly, tx, ty, exclude=()):
    sx = lx.max() - lx.min() or 1.0
    sy = ly.max() - ly.min() or 1.0
    d = ((lx - tx) / sx) ** 2 + ((ly - ty) / sy) ** 2
    if exclude:
        d = d.copy()
        d[list(exclude)] = np.inf
    return int(np.argmin(d))


def select_grid(df, args):
    lx = np.log10(df[args.xratio].to_numpy(float))
    ly = np.log10(df[args.yratio].to_numpy(float))
    qx = np.linspace(args.q_low, args.q_high, args.ncols)
    qy = np.linspace(args.q_high, args.q_low, args.nrows)   # high row first
    cells, used = [], set()
    for iy, qyy in enumerate(qy):
        for ix, qxx in enumerate(qx):
            k = nearest_row(lx, ly, np.quantile(lx, qxx),
                            np.quantile(ly, qyy), used)
            used.add(k)
            cells.append({'k': k, 'row': df.iloc[k], 'lx': lx[k], 'ly': ly[k],
                          'ix': ix, 'iy': iy})
    return cells, args.nrows, args.ncols


def select_matched(df, args):
    lx = np.log10(df[args.xratio].to_numpy(float))
    ly = np.log10(df[args.yratio].to_numpy(float))
    lks = np.log10(df['ks'].to_numpy(float))
    sx = lx.max() - lx.min() or 1.0
    sy = ly.max() - ly.min() or 1.0
    used, cells = set(), []
    for j in range(args.n_pairs):
        tq = (j + 0.5) / args.n_pairs
        tx, ty = np.quantile(lx, tq), np.quantile(ly, tq)
        d = ((lx - tx) / sx) ** 2 + ((ly - ty) / sy) ** 2
        near = [k for k in np.argsort(d)[:args.match_pool] if k not in used]
        if len(near) < 2:
            raise SystemExit('Not enough candidates for a matched pair; '
                             'raise --match-pool or lower --n-pairs.')
        a = min(near, key=lambda k: lks[k])
        b = max(near, key=lambda k: lks[k])
        used.update({a, b})
        for i, k in enumerate((a, b)):
            cells.append({'k': k, 'row': df.iloc[k], 'lx': lx[k], 'ly': ly[k],
                          'ix': i, 'iy': j})
    return cells, args.n_pairs, 2


def apply_filters(df, filters):
    """Restrict the table with repeated ``--filter COL LO HI`` ranges."""
    if not filters:
        return df, []
    out, notes = df, []
    for col, lo, hi in filters:
        if col not in out.columns:
            raise SystemExit(f"--filter column '{col}' not in the feature table.")
        lo, hi = float(lo), float(hi)
        out = out[(out[col] >= lo) & (out[col] <= hi)]
        notes.append(f'{col} in [{lo:g}, {hi:g}]')
    if out.empty:
        raise SystemExit('No landscapes remain after --filter.')
    return out.reset_index(drop=True), notes


def select_corners(df, args):
    """Four corners of ratio space, spread in the absolute parameters.

    Corners are quantile targets rather than extremes: taking the outermost
    landscape in each direction makes the figure depend on one or two unusual
    members of the ensemble.  The search returns the combination closest to
    the four targets, subject to the selected landscapes spanning at least
    ``--min-individual-dex`` decades in each of U, Kh and Ks.  Without that
    constraint the corner landscapes can share similar absolute parameters,
    leaving the figure unable to show that form follows the ratios.
    """
    import itertools

    lx = np.log10(df[args.xratio].to_numpy(float))
    ly = np.log10(df[args.yratio].to_numpy(float))
    absl = {c: np.log10(df[c].to_numpy(float))
            for c in args.individual if c in df.columns}

    qx_lo = args.qx_low if args.qx_low is not None else args.q_low
    qx_hi = args.qx_high if args.qx_high is not None else args.q_high
    qy_lo = args.qy_low if args.qy_low is not None else args.q_low
    qy_hi = args.qy_high if args.qy_high is not None else args.q_high

    tx = {0: np.quantile(lx, qx_lo), 1: np.quantile(lx, qx_hi)}
    ty = {0: np.quantile(ly, qy_hi), 1: np.quantile(ly, qy_lo)}  # row 0 = high

    sx = lx.std() or 1.0
    sy = ly.std() or 1.0
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]

    quad, dists = {}, {}
    for iy, ix in order:
        d = ((lx - tx[ix]) / sx) ** 2 + ((ly - ty[iy]) / sy) ** 2
        idx = np.argsort(d)[:args.corner_pool]
        quad[(iy, ix)] = idx
        dists[(iy, ix)] = d

    best = None
    for combo in itertools.product(*[quad[q] for q in order]):
        if len(set(combo)) < 4:
            continue
        i = list(combo)
        ind = (min(v[i].max() - v[i].min() for v in absl.values())
               if absl else np.inf)
        if ind < args.min_individual_dex:
            continue
        cost = sum(dists[q][k] for q, k in zip(order, combo))
        if best is None or cost < best[0]:
            best = (cost, dict(zip(order, combo)), ind)

    if best is None:
        raise SystemExit(
            f'No corner set satisfies --min-individual-dex '
            f'{args.min_individual_dex:g}. Lower it, widen --filter, or raise '
            '--corner-pool.')

    _, c, ind = best
    cells = [{'k': int(c[q]), 'row': df.iloc[int(c[q])],
              'lx': float(lx[c[q]]), 'ly': float(ly[c[q]]),
              'iy': q[0], 'ix': q[1]} for q in order]
    cells.sort(key=lambda d: (d['iy'], d['ix']))

    sep_x = min(lx[c[(0, 1)]] - lx[c[(0, 0)]], lx[c[(1, 1)]] - lx[c[(1, 0)]])
    sep_y = min(ly[c[(0, 0)]] - ly[c[(1, 0)]], ly[c[(0, 1)]] - ly[c[(1, 1)]])
    diag = {'sep_x': sep_x, 'sep_y': sep_y, 'ind': ind,
            'max_x': lx.max() - lx.min(), 'max_y': ly.max() - ly.min(),
            'q': (qx_lo, qx_hi, qy_lo, qy_hi),
            'targets': (tx[0], tx[1], ty[1], ty[0])}
    return cells, 2, 2, diag


def parse_rasnet_name(path):
    """Recover (job_id, landscape_idx, ts_index) from a rasnet filename.

    Convention from 02_extract_features.py:
    ``rasnet-n{elev_err}-{job_id}-{landscape_idx}-{ts_index}.pkl``
    """
    parts = Path(path).stem.split('-')
    if len(parts) < 5 or parts[0] != 'rasnet':
        raise SystemExit(
            f"Cannot parse '{Path(path).name}'. Expected "
            'rasnet-n{elev_err}-{job_id}-{landscape_idx}-{ts_index}.pkl')
    return parts[2], int(parts[3]), int(parts[4])


def select_from_paths(df, args):
    """Use rasnet files named on the command line, in reading order."""
    lx = np.log10(df[args.xratio].to_numpy(float))
    ly = np.log10(df[args.yratio].to_numpy(float))
    ncols = args.ncols
    cells = []
    for i, path in enumerate(args.rasnet):
        jid, idx, ts = parse_rasnet_name(path)
        m = ((df['job_id'].astype(str) == str(jid)) &
             (df['landscape_idx'].astype(int) == idx) &
             (df['ts_index'].astype(int) == ts))
        if not m.any():
            raise SystemExit(
                f'{Path(path).name} names job {jid}, index {idx}, ts {ts}, '
                'which is not in the feature table.')
        k = int(np.flatnonzero(m.to_numpy())[0])
        cells.append({'k': k, 'row': df.iloc[k], 'lx': float(lx[k]),
                      'ly': float(ly[k]), 'path': Path(path),
                      'ix': i % ncols, 'iy': i // ncols})
    return cells, int(np.ceil(len(cells) / ncols)), ncols


def select_explicit(df, args):
    lx = np.log10(df[args.xratio].to_numpy(float))
    ly = np.log10(df[args.yratio].to_numpy(float))
    cells = []
    for i, (jid, idx, ts) in enumerate(args.landscape):
        m = ((df['job_id'].astype(str) == str(jid)) &
             (df['landscape_idx'].astype(int) == int(idx)) &
             (df['ts_index'].astype(int) == int(ts)))
        if not m.any():
            raise SystemExit(f'No feature row for job {jid}, index {idx}, ts {ts}')
        k = int(np.flatnonzero(m.to_numpy())[0])
        cells.append({'k': k, 'row': df.iloc[k], 'lx': lx[k], 'ly': ly[k],
                      'ix': i % args.ncols, 'iy': i // args.ncols})
    return cells, int(np.ceil(len(cells) / args.ncols)), args.ncols


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_param_panel(ax, df, spec, cells, letters, args, letter, full_df=None):
    import matplotlib.pyplot as plt
    xcol, ycol, ccol = spec
    spec_cols = tuple(spec)

    # Draw the whole ensemble faintly behind the filtered subset, so the
    # figure still shows the parameter space the study actually sampled.
    if full_df is not None and args.show_full:
        ax.scatter(np.log10(full_df[xcol].to_numpy(float)),
                   np.log10(full_df[ycol].to_numpy(float)),
                   s=args.full_marker_size, c=args.full_color,
                   alpha=args.full_alpha, linewidths=0, edgecolors='none',
                   rasterized=args.rasterize, zorder=1)

    x = np.log10(df[xcol].to_numpy(float))
    y = np.log10(df[ycol].to_numpy(float))
    c = np.log10(df[ccol].to_numpy(float))

    pidx_cb = args.panel.index(list(spec_cols))
    sc = ax.scatter(x, y, c=c, cmap=args.cmap, s=args.marker_size,
                    alpha=args.alpha, linewidths=0, edgecolors='none',
                    rasterized=args.rasterize, zorder=2)

    # Mark the landscapes drawn below.  A white-filled, black-outlined
    # circle reads clearly against viridis at any value; the letter sits at a
    # fixed offset from it.
    overrides = {}
    for spec in (args.mark_label_pos or []):
        lab_, pidx, dx_, dy_ = spec
        overrides[(lab_.upper(), int(pidx))] = (float(dx_), float(dy_))

    pidx = args.panel.index(list(spec_cols))
    for cell, lab in zip(cells, letters):
        k = cell['k']
        ax.plot(x[k], y[k], marker=args.mark_marker, ms=args.mark_size,
                mfc=args.mark_face, mec=args.mark_edge, mew=args.mark_lw,
                linestyle='none', zorder=5)
        off = overrides.get((lab, pidx), tuple(args.mark_label_offset))
        ax.annotate(lab, xy=(x[k], y[k]), xycoords='data',
                    xytext=off, textcoords='offset points',
                    fontsize=args.mark_label_size, fontweight='bold',
                    color=args.mark_label_color,
                    ha=args.mark_label_ha, va=args.mark_label_va, zorder=6,
                    bbox=dict(boxstyle=f'square,pad={args.mark_label_pad}',
                              fc=args.mark_label_bg,
                              ec=args.mark_label_bg_edge,
                              lw=args.mark_label_bg_lw,
                              alpha=args.mark_label_bg_alpha))

    ax.set_xlabel(AXIS_LABELS.get(xcol, xcol), fontsize=args.label_size)
    ax.set_ylabel(AXIS_LABELS.get(ycol, ycol), fontsize=args.label_size)
    ax.tick_params(labelsize=args.tick_size, length=2.5, width=0.6)

    rects = args.pcbar_rect or DEFAULT_PCBAR_RECTS
    cax = ax.inset_axes(rects[pidx_cb] if pidx_cb < len(rects) else rects[-1])
    cb = plt.colorbar(sc, cax=cax, orientation='vertical')
    cb.ax.set_title(CBAR_LABELS.get(ccol, ccol),
                    fontsize=args.cbar_label_size, pad=3)
    if args.cbar_ticks_left and (rects[pidx_cb][0] if pidx_cb < len(rects)
                                 else rects[-1][0]) > 0.5:
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(labelsize=args.cbar_tick_size, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)

    ax.text(args.letter_x, args.letter_y, letter, transform=ax.transAxes,
            fontsize=args.letter_size, fontweight=args.letter_weight,
            va='bottom', ha='left')


def draw_map(ax, cell, args, letter, show_x, show_y, window):
    import matplotlib.pyplot as plt

    z, dx = cell['z'], cell['dx']
    ny, nx = z.shape
    keep_cols, keep_rows = window          # in array cells, not distance

    def centre_slice(n, k):
        if not k or k >= n:
            return 0, n
        lo = (n - k) // 2
        return lo, lo + k

    cx0, cx1 = centre_slice(nx, keep_cols)
    cy0, cy1 = centre_slice(ny, keep_rows)
    z = z[cy0:cy1, cx0:cx1]
    ny, nx = z.shape

    if args.crop_origin == 'true':
        ext = [cx0 * dx / args.length_scale, cx1 * dx / args.length_scale,
               cy1 * dx / args.length_scale, cy0 * dx / args.length_scale]
    else:
        ext = [0, nx * dx / args.length_scale, ny * dx / args.length_scale, 0]

    im = ax.imshow(z, cmap=args.map_cmap, extent=ext, origin='upper',
                   interpolation=args.interpolation, rasterized=args.rasterize)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=args.tick_size, length=2.5, width=0.6)
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])

    cax = ax.inset_axes(args.mcbar_rect)
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label(args.cbar_label, fontsize=args.cbar_label_size,
                 labelpad=args.cbar_labelpad)
    cb.ax.tick_params(labelsize=args.cbar_tick_size, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)

    ax.text(args.letter_x, args.letter_y, letter, transform=ax.transAxes,
            fontsize=args.letter_size, fontweight=args.letter_weight,
            va='bottom', ha='left')
    if args.annotate_ratios:
        ax.text(0.02, 0.96,
                f"$\\log\\,{RATIO_TEX.get(args.xratio, args.xratio)}"
                f"={cell['lx']:.2f}$\n"
                f"$\\log\\,{RATIO_TEX.get(args.yratio, args.yratio)}"
                f"={cell['ly']:.2f}$",
                transform=ax.transAxes, fontsize=args.annot_size,
                va='top', ha='left', linespacing=1.5,
                bbox=dict(boxstyle='square,pad=0.22', fc='white', ec='none',
                          alpha=0.7))


def build_figure(df, cells, mrows, mcols, args, full_df=None):
    import matplotlib.pyplot as plt

    n_param = len(args.panel)
    n_rows = 1 + mrows
    figsize = (args.width_cm / CM_PER_INCH, args.height_cm / CM_PER_INCH)
    fig = plt.figure(figsize=figsize)

    ratios = args.row_height_ratios
    if ratios and len(ratios) != n_rows:
        raise SystemExit(f'--row-height-ratios needs {n_rows} values')
    if not ratios:
        ratios = [args.param_row_height] + [1.0] * mrows

    total_h = args.top - args.bottom
    gap = args.row_gap * total_h / n_rows
    extra = args.extra_gap * total_h / n_rows if args.extra_gap else 0.0
    usable = total_h - gap * (n_rows - 1) - extra
    heights = [usable * v / sum(ratios) for v in ratios]

    tops, y = [], args.top
    for i, h in enumerate(heights):
        tops.append(y)
        y -= h + gap + (extra if i == 0 else 0.0)

    letters = list('ABCDEFGHIJKLMNOP')
    map_letters = letters[n_param:n_param + len(cells)]

    # Row 1: parameter space
    gs0 = fig.add_gridspec(1, n_param, left=args.left, right=args.right,
                           top=tops[0], bottom=tops[0] - heights[0],
                           wspace=args.param_wspace)
    for i, spec in enumerate(args.panel):
        ax = fig.add_subplot(gs0[0, i])
        draw_param_panel(ax, df, spec, cells, map_letters, args, letters[i],
                         full_df=full_df)

    for spec in (args.note or DEFAULT_NOTES):
        idx, fx, fy, text = int(spec[0]), float(spec[1]), float(spec[2]), spec[3]
        if idx < n_param:
            fig.axes[idx].text(fx, fy, text.replace('\\n', '\n'),
                               transform=fig.axes[idx].transAxes,
                               fontsize=args.note_size, color=args.note_color,
                               va='top', ha='left', linespacing=1.25)

    # Preload the grids once, then pick a window every map can supply.
    for cell in cells:
        p = cell.get('path') or rasnet_path(args.rasnet_dir, cell['row'],
                                            args.elev_err)
        cell['z'], cell['dx'] = load_elevation(p, args.mask)
        if args.transpose:
            cell['z'] = cell['z'].T

    # Work in array cells: node spacing is isotropic, so an aspect ratio in
    # cells is the same aspect ratio in distance, and no unit conversion can
    # go wrong.
    n_rows_avail = min(c['z'].shape[0] for c in cells)
    n_cols_avail = min(c['z'].shape[1] for c in cells)
    keep_rows = min(args.crop_rows or n_rows_avail, n_rows_avail)
    keep_cols = args.crop_cols or int(round(args.map_aspect * keep_rows))
    keep_cols = min(keep_cols, n_cols_avail)
    window = (keep_cols, keep_rows)

    dx0 = cells[0]['dx']
    if args.summary or keep_cols == n_cols_avail:
        msg = (f'\nMap window: {keep_cols} x {keep_rows} cells '
               f'= {keep_cols * dx0 / args.length_scale:.2f} x '
               f'{keep_rows * dx0 / args.length_scale:.2f} '
               f'(aspect {keep_cols / keep_rows:.2f}:1); '
               f'grids are {n_cols_avail} x {n_rows_avail} cells.')
        if keep_cols == n_cols_avail:
            msg += ('\n  No horizontal cropping: the requested window is at '
                    'least as wide as the grid. If the maps look too long, '
                    'the array may be transposed - try --transpose, or run '
                    'diagnose_rasnet.py.')
        print(msg)

    # Rows 2+: landscape maps
    li = iter(map_letters)
    map_axes = {}
    for r in range(mrows):
        gs = fig.add_gridspec(1, mcols, left=args.left, right=args.map_right,
                              top=tops[1 + r], bottom=tops[1 + r] - heights[1 + r],
                              wspace=args.map_wspace)
        for c in range(mcols):
            cell = next((x for x in cells if x['iy'] == r and x['ix'] == c), None)
            ax = fig.add_subplot(gs[0, c])
            if cell is None:
                ax.set_visible(False)
                continue
            draw_map(ax, cell, args, next(li),
                     show_x=(r == mrows - 1), show_y=(c == 0),
                     window=window)
            map_axes[(r, c)] = ax

    # Headers describe the ratio axes of the map grid, so they apply whenever
    # the maps are laid out as one, including when plotted from --rasnet paths.
    if args.axis_headers and mrows > 1 and mcols > 1:
        xl = RATIO_TEX.get(args.xratio, args.xratio)
        yl = RATIO_TEX.get(args.yratio, args.yratio)
        for c, lab in zip(range(mcols), args.col_labels or []):
            if (0, c) in map_axes:
                map_axes[(0, c)].set_title(f'{lab} ${xl}$',
                                           fontsize=args.header_size,
                                           pad=args.header_pad)
        for r, lab in zip(range(mrows), args.row_labels or []):
            if (r, 0) in map_axes:
                map_axes[(r, 0)].set_ylabel(f'{lab} ${yl}$',
                                            fontsize=args.header_size,
                                            labelpad=args.header_pad,
                                            rotation=args.header_rotation,
                                            va='center')

    if args.map_xlabel:
        fig.text(args.map_xlabel_x, args.map_xlabel_y, args.map_xlabel,
                 ha='center', fontsize=args.label_size)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    io = p.add_argument_group('input / output')
    io.add_argument('--data-dir', default='outputs/features')
    io.add_argument('--rasnet-dir', default=None,
                    help='Defaults to --data-dir.')
    io.add_argument('--features-hash', default=None)
    io.add_argument('--job-ids', nargs='+', default=None)
    io.add_argument('--elev-err', type=float, default=None)
    io.add_argument('--out', default='fig-param-space-landscapes.pdf')
    io.add_argument('--dpi', type=int, default=600)
    io.add_argument('--select-only', action='store_true',
                    help='Run the selection, report the chosen landscapes and '
                         'the command to plot them, then stop. Reads only the '
                         'feature table; no rasnet file is opened.')
    io.add_argument('--rasnet', action='append', default=None, metavar='PATH',
                    help='Plot these rasnet files, in reading order, instead '
                         'of selecting. Each filename is parsed for its job '
                         'id, landscape index and timestep, which are looked '
                         'up in the feature table for the markers on A and B. '
                         'Repeatable.')
    io.add_argument('--summary', action='store_true')

    pn = p.add_argument_group('parameter-space panels')
    pn.add_argument('--panel', action='append', nargs=3,
                    metavar=('XCOL', 'YCOL', 'COLOURCOL'), default=None,
                    help="Default: 'ks u u_ks' and 'ks kh kh_ks'.")
    pn.add_argument('--note', action='append', nargs=4,
                    metavar=('PANEL', 'X', 'Y', 'TEXT'), default=None)

    se = p.add_argument_group('landscape selection')
    se.add_argument('--mode',
                    choices=['grid', 'corners', 'matched', 'explicit'],
                    default='corners',
                    help="'corners': maximise ratio separation subject to a "
                         'minimum spread in the absolute parameters. '
                         "'grid': plain quantile picks. [%(default)s]")
    se.add_argument('--filter', action='append', nargs=3,
                    metavar=('COL', 'LO', 'HI'), default=None,
                    help='Restrict to LO <= COL <= HI before selecting. '
                         "Repeatable, e.g. '--filter Ly 14500 15500'.")
    se.add_argument('--min-individual-dex', type=float, default=2.0,
                    help='Required span, in decades, of each absolute '
                         'parameter across the selected landscapes. '
                         '[%(default)s]')
    se.add_argument('--individual', nargs='*', default=['u', 'kh', 'ks'],
                    help='Absolute parameters the spread applies to. '
                         '[%(default)s]')
    se.add_argument('--corner-pool', type=int, default=12,
                    help='Candidates considered per quadrant; the search is '
                         'over all combinations, so cost grows as the fourth '
                         'power. [%(default)s]')
    se.add_argument('--xratio', default='u_ks')
    se.add_argument('--yratio', default='kh_ks')
    se.add_argument('--nrows', type=int, default=2)
    se.add_argument('--ncols', type=int, default=2)
    se.add_argument('--q-low', type=float, default=0.10,
                    help='Quantile for the low end of each ratio; corners are '
                         'placed at quantiles rather than extremes so the '
                         'figure does not hinge on outliers. [%(default)s]')
    se.add_argument('--q-high', type=float, default=0.90,
                    help='Quantile for the high end. [%(default)s]')
    se.add_argument('--qx-low', type=float, default=None,
                    help='Override --q-low for the x ratio only.')
    se.add_argument('--qx-high', type=float, default=None,
                    help='Override --q-high for the x ratio only.')
    se.add_argument('--qy-low', type=float, default=None,
                    help='Override --q-low for the y ratio only.')
    se.add_argument('--qy-high', type=float, default=None,
                    help='Override --q-high for the y ratio only.')
    se.add_argument('--n-pairs', type=int, default=2)
    se.add_argument('--match-pool', type=int, default=25)
    se.add_argument('--landscape', action='append', nargs=3,
                    metavar=('JOB_ID', 'IDX', 'TS'), default=None)

    lay = p.add_argument_group('layout')
    lay.add_argument('--width-cm', type=float, default=19.0)
    lay.add_argument('--height-cm', type=float, default=16.0)
    lay.add_argument('--left', type=float, default=0.105)
    lay.add_argument('--right', type=float, default=0.985)
    lay.add_argument('--map-right', type=float, default=0.925,
                     help='Right margin for the map rows, leaving room for '
                          'their colourbars. [%(default)s]')
    lay.add_argument('--top', type=float, default=0.945)
    lay.add_argument('--bottom', type=float, default=0.065)
    lay.add_argument('--param-wspace', type=float, default=0.30)
    lay.add_argument('--map-wspace', type=float, default=0.45)
    lay.add_argument('--row-gap', type=float, default=0.22)
    lay.add_argument('--extra-gap', type=float, default=0.30,
                     help='Additional separation after the parameter-space '
                          'row. [%(default)s]')
    lay.add_argument('--param-row-height', type=float, default=2.05,
                     help='Height of the parameter-space row relative to one '
                          'map row. [%(default)s]')
    lay.add_argument('--row-height-ratios', nargs='*', type=float, default=None,
                     help='Explicit per-row heights, overriding '
                          '--param-row-height.')
    lay.add_argument('--length-scale', type=float, default=1000.0)
    lay.add_argument('--map-aspect', type=float, default=2.0,
                     help='Length-to-height ratio of the map window. The '
                          'height is the smallest available across the '
                          'selected landscapes, so every map covers the same '
                          'area and shares one scale. Ignored when '
                          '--crop-cols is given. [%(default)s]')
    lay.add_argument('--crop-cols', type=int, default=0,
                     help='Keep this many central columns of the elevation '
                          'array. 0 derives the count from --map-aspect. '
                          '[%(default)s]')
    lay.add_argument('--crop-rows', type=int, default=0,
                     help='Keep this many central rows. 0 uses the smallest '
                          'row count among the selected landscapes. '
                          '[%(default)s]')
    lay.add_argument('--transpose', action='store_true',
                     help='Swap the array axes after loading, for grids '
                          'stored with rows along x.')
    lay.add_argument('--crop-origin', choices=['zero', 'true'], default='zero',
                     help="'zero' labels the crop from 0; 'true' keeps the "
                          'coordinates it occupies in the full domain. '
                          '[%(default)s]')

    st = p.add_argument_group('style - scatter and maps')
    st.add_argument('--cmap', default='viridis')
    st.add_argument('--map-cmap', default='viridis')
    st.add_argument('--marker-size', type=float, default=14.0)
    st.add_argument('--show-full', dest='show_full', action='store_true',
                    default=True,
                    help='Draw the unfiltered ensemble faintly behind the '
                         'filtered subset. [on]')
    st.add_argument('--no-show-full', dest='show_full', action='store_false')
    st.add_argument('--full-color', default='#C9CDD2')
    st.add_argument('--full-marker-size', type=float, default=7.0)
    st.add_argument('--full-alpha', type=float, default=0.55)
    st.add_argument('--alpha', type=float, default=0.5)
    st.add_argument('--interpolation', default='nearest')
    st.add_argument('--mask', dest='mask', action='store_true', default=False)
    st.add_argument('--rasterize', dest='rasterize', action='store_true',
                    default=True)
    st.add_argument('--no-rasterize', dest='rasterize', action='store_false')

    mk = p.add_argument_group('style - landscape markers on A and B')
    mk.add_argument('--mark-marker', default='o')
    mk.add_argument('--mark-size', type=float, default=6.5)
    mk.add_argument('--mark-face', default='white')
    mk.add_argument('--mark-edge', default='black')
    mk.add_argument('--mark-lw', type=float, default=1.2)
    mk.add_argument('--mark-label-size', type=float, default=10.0)
    mk.add_argument('--mark-label-color', default='black')
    mk.add_argument('--mark-label-offset', nargs=2, type=float,
                    default=[7.0, 5.0], metavar=('DX', 'DY'),
                    help='Label offset from its marker, in points. '
                         '[%(default)s]')
    mk.add_argument('--mark-label-ha', default='left',
                    choices=['left', 'center', 'right'])
    mk.add_argument('--mark-label-va', default='bottom',
                    choices=['bottom', 'center', 'top'])
    mk.add_argument('--mark-label-bg', default='white',
                    help="Label backing colour; 'none' disables it. "
                         '[%(default)s]')
    mk.add_argument('--mark-label-bg-alpha', type=float, default=0.7,
                    help='Opacity of the label backing. [%(default)s]')
    mk.add_argument('--mark-label-bg-edge', default='none')
    mk.add_argument('--mark-label-bg-lw', type=float, default=0.0)
    mk.add_argument('--mark-label-pad', type=float, default=0.18,
                    help='Padding inside the label backing, in font-size '
                         'units. [%(default)s]')
    mk.add_argument('--mark-label-pos', action='append', nargs=4,
                    metavar=('LETTER', 'PANEL', 'DX', 'DY'), default=None,
                    help='Override one label offset: its letter, 0-based '
                         'panel index, and dx/dy in points. Repeatable; use '
                         'when two markers sit close together.')

    cb = p.add_argument_group('style - colourbars')
    cb.add_argument('--pcbar-rect', action='append', nargs=4, type=float,
                    default=None, metavar=('X', 'Y', 'W', 'H'),
                    help='Inset colourbar rectangle for one parameter-space '
                         'panel, in axes fraction. Repeat once per panel; the '
                         'last given is reused for any remaining panels. '
                         'Default places A on the left and B on the right.')
    cb.add_argument('--cbar-ticks-left', dest='cbar_ticks_left',
                    action='store_true', default=True,
                    help='Put tick labels on the inner side when a colourbar '
                         'sits on the right. [on]')
    cb.add_argument('--no-cbar-ticks-left', dest='cbar_ticks_left',
                    action='store_false')
    cb.add_argument('--mcbar-rect', nargs=4, type=float,
                    default=[1.03, 0.05, 0.035, 0.90], metavar=('X', 'Y', 'W', 'H'))
    cb.add_argument('--cbar-label', default='elev, m')
    cb.add_argument('--cbar-label-size', type=float, default=8.5)
    cb.add_argument('--cbar-tick-size', type=float, default=7.0)
    cb.add_argument('--cbar-labelpad', type=float, default=2.0)

    tx = p.add_argument_group('style - text')
    tx.add_argument('--font-family', default='sans-serif')
    tx.add_argument('--font-size', type=float, default=8.0)
    tx.add_argument('--tick-size', type=float, default=8.0)
    tx.add_argument('--label-size', type=float, default=10.0)
    tx.add_argument('--note-size', type=float, default=9.0)
    tx.add_argument('--note-color', default='#1F1F1F')
    tx.add_argument('--header-size', type=float, default=12.0)
    tx.add_argument('--header-pad', type=float, default=8.0)
    tx.add_argument('--header-rotation', type=float, default=90.0,
                    help='Rotation of the row headers on the left-hand maps. '
                         '90 reads bottom-to-top; 0 reads horizontally. '
                         '[%(default)s]')
    tx.add_argument('--col-labels', nargs='*', default=['low', 'high'])
    tx.add_argument('--row-labels', nargs='*', default=['high', 'low'])
    tx.add_argument('--axis-headers', dest='axis_headers', action='store_true',
                    default=True)
    tx.add_argument('--no-axis-headers', dest='axis_headers',
                    action='store_false')
    tx.add_argument('--annotate-ratios', dest='annotate_ratios',
                    action='store_true', default=False)
    tx.add_argument('--annot-size', type=float, default=7.0)
    tx.add_argument('--map-xlabel', default='distance, km')
    tx.add_argument('--map-xlabel-x', type=float, default=0.5)
    tx.add_argument('--map-xlabel-y', type=float, default=0.012)
    tx.add_argument('--letter-size', type=float, default=12.0)
    tx.add_argument('--letter-weight', default='bold')
    tx.add_argument('--letter-x', type=float, default=-0.015)
    tx.add_argument('--letter-y', type=float, default=1.02)
    tx.add_argument('--letter-bbox-alpha', type=float, default=0.65)

    a = p.parse_args(argv)
    if a.panel is None:
        a.panel = [list(t) for t in DEFAULT_PARAM_PANELS]
    if a.rasnet_dir is None:
        a.rasnet_dir = a.data_dir
    if a.mode == 'explicit' and not a.landscape:
        p.error('--mode explicit requires at least one --landscape')
    return a


def main(argv=None):
    args = parse_args(argv)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': args.font_family,
        'font.size': args.font_size,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    full_df = load_table(args.data_dir, args.features_hash, args.job_ids)
    df, filt_notes = apply_filters(full_df, args.filter)
    if args.rasnet:
        res = select_from_paths(df, args)
    else:
        select = {'grid': select_grid, 'matched': select_matched,
                  'corners': select_corners,
                  'explicit': select_explicit}[args.mode]
        res = select(df, args)
    cells, mrows, mcols = res[0], res[1], res[2]
    diag = res[3] if len(res) > 3 else None

    letters = list('ABCDEFGHIJKLMNOP')[len(args.panel):]
    if args.summary or args.select_only:
        if filt_notes:
            print('\nFilters: ' + '; '.join(filt_notes))
        if diag and 'q' in diag:
            qxl, qxh, qyl, qyh = diag['q']
            txl, txh, tyl, tyh = diag['targets']
            print(f'\nCorner targets (quantiles of the filtered set):'
                  f'\n  {args.xratio:6s} q{qxl:.2f}={txl:.2f}  '
                  f'q{qxh:.2f}={txh:.2f}'
                  f'\n  {args.yratio:6s} q{qyl:.2f}={tyl:.2f}  '
                  f'q{qyh:.2f}={tyh:.2f}')
        if diag:
            print(f"\nRatio separation achieved (log10 decades):"
                  f"\n  {args.xratio:6s} {diag['sep_x']:.2f}  "
                  f"(most available after filtering: {diag['max_x']:.2f})"
                  f"\n  {args.yratio:6s} {diag['sep_y']:.2f}  "
                  f"(most available after filtering: {diag['max_y']:.2f})"
                  f"\n  smallest span across U, Kh, Ks: {diag['ind']:.2f}")
        print(f'\n{len(df):,} landscapes; {len(cells)} shown:')
        for lab, c in zip(letters, cells):
            r = c['row']
            print(f"  {lab}  job {r['job_id']} idx {int(r['landscape_idx'])} "
                  f"ts {int(r['ts_index'])}   "
                  f"log {args.xratio}={c['lx']:6.3f}  "
                  f"log {args.yratio}={c['ly']:6.3f}   "
                  f"log ks={np.log10(r['ks']):6.3f}")

    if args.select_only:
        rd = Path(args.rasnet_dir)
        paths = [rd / rasnet_path(rd, c['row'], args.elev_err).name
                 for c in cells]
        missing = [p for p in paths if not p.exists()]
        print('\nrasnet files:')
        for lab, p in zip(letters, paths):
            print(f"  {lab}  {p}{'   [MISSING]' if not p.exists() else ''}")
        print('\nTo plot these exact landscapes:\n')
        lines = [f'  python {Path(__file__).name}',
                 f'      --data-dir {args.data_dir}']
        lines += [f'      --rasnet {p}' for p in paths]
        lines += ['      --out fig-param-space-landscapes.pdf']
        print(' \\\n'.join(lines))
        if missing:
            print(f'\nNote: {len(missing)} file(s) not found in '
                  f"'{args.rasnet_dir}'. Check --rasnet-dir and --elev-err.")
        return 0

    fig = build_figure(df, cells, mrows, mcols, args,
                       full_df=full_df if args.show_full else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f'\nWrote {out}  ({args.width_cm:g}x{args.height_cm:g} cm, '
          f'{len(args.panel)} param panel(s) + {mrows}x{mcols} maps)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
