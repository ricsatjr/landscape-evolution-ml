#!/usr/bin/env python
# coding: utf-8
"""
01d_screen_model_time.py
========================
Screening experiment that fixes the model time multiplier ``k`` used by
``01_generate_landscapes.py`` for the whole production ensemble.

This is step one of the two-step steady-state determination.  The model time
allotted to a landscape must be fixed before that landscape exists, whereas
whether the allotted time sufficed can only be known from the record of the
completed run.  This script performs the screening; ``01d_check_steady_state``
performs the per-landscape verification afterwards.

Although numbered after it, this script runs BEFORE ``01_generate_landscapes``:
its result is the value assigned to ``T_TC`` there.

What is estimated
-----------------
Model time is expressed as a multiple of the characteristic timescale,

    T_ss = k * t_c,     t_c = 1/Ks          (Theodoratos et al., 2018)

so the quantity to estimate is the single dimensionless constant k.  That it
is dimensionless is what makes a small sample informative: k is one number,
and the same number applies across an ensemble spanning orders of magnitude
in t_c.  No estimate made in absolute time could transfer that way.

Design of the sweep
-------------------
A small sample of parameter sets is drawn once, with a fixed seed, and the
SAME sets and the SAME initial elevation fields are reused at every k.  The
only quantity that varies across the sweep is T_ss, so a difference between
panels is attributable to model time alone.

Each k is an independent run, not a continuation of a shorter one: the number
of time steps is held at ``num_ts`` and dt = T_ss / num_ts, so raising k
lengthens the time step rather than adding steps.  A run at k = 25 is
therefore not an extension of the run at k = 20 and cannot be assembled from
it.

The sweep runs on a square grid (``--ly-mult 1``) rather than the production
Lx = 5*Ly.  Sediment transport is directed from the central ridge towards the
two long edges, so the timescale for reaching equilibrium is governed by Ly
and is independent of Lx; a square grid holds Ly fixed while cutting the cell
count, and is therefore far cheaper for the same answer.

The criterion
-------------
The criterion of record in the manuscript is visual: the sweep stops at the
smallest k at which every normalized curve appears flattened towards t* = 1,
and the figure this script writes is what that judgement is made on.

Alongside it, and only as support, the same statistic used in step two is
computed for each curve: the range of the normalized series over its final
window, R_w.  Reporting it here does three things.  It makes the stopping
decision reproducible rather than resting on a reading of the figure, it lets
the screening and the verification be stated in one currency, and it gives
the sweep an automatic early stop.  The figure remains the object a reader
inspects; the number is what lets them check the inspection.

Note that passing here is a weaker statement than passing in step two.  The
sample is small, the grid is square rather than elongated, and a landscape
that satisfies the criterion at this k is evidence about the sample, not a
guarantee for the ensemble.  That guarantee is exactly what step two supplies,
per landscape, and why the two steps are a pair: this screening may be
inexpensive because it is audited afterwards.

Parallelism
-----------
The landscapes of one multiplier are independent and are the unit of work.
Multipliers themselves are run in sequence, because the sweep stops at the
first k that passes and starting a later one would usually be discarded.

Worker count is adaptive rather than fixed.  A landscape's memory footprint
scales with Ly and is not known before it runs, so the pool starts at one
worker and grows while memory and CPU allow, estimating the cost of one more
task from the resident memory of the tasks already running.  It shrinks when
free memory falls below the reserve.  Tasks are submitted as slots free,
which also balances the load: runtimes vary with grid size and time step, so
a static split would leave workers waiting on the slowest share.  Without
``psutil`` the count is fixed at the ceiling and nothing is sampled.

Numerical threading is pinned to one thread per process, set before numpy is
imported.  Nested threading inside workers oversubscribes the machine and
breaks determinism, which here rests on process isolation.

Results are keyed by landscape index and emitted in index order, so output
does not depend on worker assignment or completion order; ``--jobs 1`` runs
the same code path in this process and produces identical output.

Usage
-----
    python pipeline/01c_screen_model_time.py --out-dir data/landscapes/model-time

    python pipeline/01d_screen_model_time.py --k-values 5 10 15 20 25 30 \
        --n-landscapes 30 --out-dir data/landscapes/model-time

    # run the whole sweep rather than stopping at the first k that passes
    python pipeline/01d_screen_model_time.py --no-early-stop \
        --out-dir data/landscapes/model-time

Output
------
    screen-model-time-n{n}-ts{num_ts}-{hash}.csv
        One row per (k, landscape): the LE parameters, dt, the diffusion
        stability ratio, R_w, and whether that curve passed.

    screen-model-time-n{n}-ts{num_ts}-{hash}.npz
        The normalized mean-elevation series themselves, one
        (n_landscapes, num_ts) array per multiplier, keyed ``k5``, ``k10``
        and so on in the order of ``landscape_idx``.  Written before the
        figure, so a sweep survives a plotting failure and the series can be
        re-examined or redrawn without re-running.

    screen-model-time-n{n}-ts{num_ts}-{hash}.png
        One panel per k in a grid of ``--fig-cols`` columns, each showing all
        normalized mean-elevation curves against normalized time.  This is
        the figure the visual criterion is applied to.

    The filenames encode the sample size, the step count and the git hash so
    runs under different settings coexist in one directory.

Notes
-----
Because num_ts is held fixed, dt grows in proportion to k.  Landlab's
``LinearDiffuser`` subdivides a step that exceeds its own stability criterion,
so a large dt costs more rather than producing a visibly wrong answer, and run
time grows with k even though the step count does not.  The ``cfl`` column
reports dt against the explicit two-dimensional diffusion limit dx^2/(4*Kh);
values above 1 indicate where that subdivision is occurring.

Dependencies
------------
    landlab >= 2.8.0, numpy, pandas, matplotlib
    See environment.yml for pinned versions.
"""

import argparse
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

# Single-threaded numerics, set before numpy is imported so the BLAS and
# OpenMP pools are sized correctly at load. Nested threading inside parallel
# worker processes oversubscribes the machine and, per the pipeline's own
# findings, breaks determinism: reproducibility here rests on process
# isolation with one thread each. ``setdefault`` leaves an existing value
# alone, so a shell that already sets these keeps control.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

try:
    import psutil
except ImportError:                       # adaptive sizing unavailable
    psutil = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from pipeline_utils import _git_hash  # noqa: E402
except ImportError:                       # standalone use outside pipeline/
    def _git_hash(short: bool = True) -> str:
        """Fallback when pipeline_utils is not importable."""
        return 'unknown'


# =============================================================================
# SIBLING MODULE LOADING
# =============================================================================

def _load_sibling(stem: str):
    """Import a pipeline module whose name begins with a digit.

    Parameters
    ----------
    stem : str
        File stem of a module in this directory, without the .py suffix,
        e.g. ``'01_generate_landscapes'``.

    Returns
    -------
    module

    Raises
    ------
    SystemExit
        If the file is absent, with the path that was tried.

    Notes
    -----
    ``import 01_generate_landscapes`` is a syntax error, since an identifier
    cannot begin with a digit, so the pipeline's numbered scripts cannot be
    imported by name.  Loading them from their path is the way to reuse their
    functions rather than restating them here, which is what keeps this script
    and the forward model from drifting apart.
    """
    path = Path(__file__).resolve().parent / f'{stem}.py'
    if not path.exists():
        sys.exit(f"ERROR: cannot find {path}. 01d must sit in pipeline/ "
                 f"alongside the scripts it reuses.")
    spec = importlib.util.spec_from_file_location(stem.replace('-', '_'), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# FIXED CONSTANTS
# =============================================================================

# Multipliers swept by default. The manuscript reports this sequence, stopped
# at the first value that passed.
K_VALUES = (5, 10, 15, 20, 25)

# Landscapes in the screening sample.
N_SCREEN = 30

# Grid length multiplier for the sweep. 1 gives the square grid (Lx = Ly) the
# manuscript describes; the production ensemble uses 5.
LY_MULT_SCREEN = 1

# Seed for the screening sample. Kept distinct from any production job_id so a
# screening run cannot be mistaken for job 0 of the main ensemble.
SCREEN_SEED = 9001

# Trailing points forming the final window, matching --final-window in 01c so
# the two steps report the same statistic.
FINAL_WINDOW = 5

# Column order for the output table.
COLUMN_ORDER = (
    'k', 'landscape_idx', 'u', 'kh', 'ks', 'ly', 'dx',
    'T_ss', 'dt', 'cfl', 'z_final', 'R_w', 'passed',
)

# Memory held back from the worker pool, in GiB. Landlab's grid and the
# depression router dominate a worker's footprint, and that footprint scales
# with Ly, so it is not known until landscapes have actually run.
MEM_RESERVE_GB = 2.0

# Seconds between load samples. Long relative to sampling cost, short
# relative to a landscape run.
PROBE_INTERVAL_S = 5.0

# Fraction of the cores that must be idle before the limiter adds a worker.
CPU_HEADROOM = 0.15

# Panel columns in the figure. Rows follow from the number of multipliers.
FIG_COLS = 3


_WORKER_MODS = {}


def _worker_modules():
    """Return the loaded sibling modules, loading them once per process.

    Returns
    -------
    (module, module)
        ``01_generate_landscapes`` and ``01d_check_steady_state``.

    Notes
    -----
    Called in the parent before the pool is created, so that forked workers
    inherit the loaded modules rather than each re-executing them.
    """
    if not _WORKER_MODS:
        _WORKER_MODS['gen'] = _load_sibling('01_generate_landscapes')
        _WORKER_MODS['chk'] = _load_sibling('01d_check_steady_state')
    return _WORKER_MODS['gen'], _WORKER_MODS['chk']


# =============================================================================
# UNIT OF WORK
# =============================================================================

def _task(job: dict) -> tuple:
    """Evolve one landscape at one multiplier and measure its curve.

    Parameters
    ----------
    job : dict
        Plain picklable payload: ``k``, ``landscape_idx``, ``u``, ``kh``,
        ``ks``, ``ly``, ``dx``, ``num_ts``, ``final_window``, ``ly_mult``,
        ``seed`` and ``delta``.

    Returns
    -------
    (int, dict, np.ndarray)
        Landscape index, its result record, and its normalized series.

    Notes
    -----
    Module level and taking only plain data, so it is picklable and runs
    unchanged in the parent or in a worker. The elevation seed depends on the
    landscape index and the screening seed but not on k, and numpy's global
    seed is set inside ``initialize_grid``, so results depend on neither the
    worker a task lands on nor the order tasks complete in.

    The screening needs the elevation series only, not the grids.
    ``evolve_landscape`` writes a snapshot when ``(i + 1) % save_every == 0``,
    and ``i + 1`` never exceeds ``num_ts``, so any modulus above ``num_ts``
    suppresses writing entirely. Asserted rather than assumed: were it to
    fail, snapshots would land in the working directory named
    ``elevts-None-None-*.npy``, since no identifier or output directory is
    passed.
    """
    gen, chk = _worker_modules()
    num_ts = job['num_ts']

    mg, _, _ = gen.initialize_grid(
        ly=job['ly'], dx=job['dx'], ly_mult=job['ly_mult'],
        elev_seed=100 * job['seed'] + job['landscape_idx'],
    )

    no_snapshots = num_ts + 1
    assert no_snapshots > num_ts, 'snapshot suppression broken'

    mean_elevs, _ = gen.evolve_landscape(
        mg, u=job['u'], kh=job['kh'], ks=job['ks'],
        t_tc=job['k'], num_ts=num_ts, save_every=no_snapshots,
    )

    T_ss = job['k'] / job['ks']
    dt = T_ss / num_ts
    z_ref = chk.reference_level(mean_elevs, 'final', job['final_window'])
    zn = mean_elevs / z_ref
    R_w = float(chk.running_range(zn)[num_ts - job['final_window']])

    row = {
        'k': job['k'], 'landscape_idx': job['landscape_idx'],
        'u': job['u'], 'kh': job['kh'], 'ks': job['ks'],
        'ly': job['ly'], 'dx': job['dx'],
        'T_ss': T_ss, 'dt': dt,
        'cfl': dt / (job['dx'] ** 2 / (4.0 * job['kh'])),
        'z_final': float(mean_elevs[-1]),
        'R_w': R_w, 'passed': bool(R_w < job['delta']),
    }
    return job['landscape_idx'], row, zn


# =============================================================================
# ADAPTIVE CONCURRENCY
# =============================================================================

class AdaptiveLimiter:
    """Concurrency target that follows memory headroom and CPU load.

    Parameters
    ----------
    max_jobs : int
        Hard ceiling; the target never exceeds it.
    reserve_bytes : int
        Memory held back. The target is reduced when available memory falls
        below this, and raised only when there is this much plus a further
        task's worth of headroom.
    interval : float
        Seconds between samples.
    verbose : bool
        Print each change in the target.

    Notes
    -----
    Starts at one worker and grows, rather than starting wide and backing
    off. A landscape's memory footprint scales with Ly and is not known in
    advance, so the only honest estimate comes from watching tasks that are
    already running: the limiter divides the pool's resident memory by the
    number in flight and treats that as the cost of one more.

    The target moves by at most one worker per interval, in one direction
    per sample, which keeps it from oscillating between the memory and CPU
    conditions. Without psutil the target is fixed at ``max_jobs`` and
    nothing is sampled.
    """

    def __init__(self, max_jobs: int, reserve_bytes: int, interval: float,
                 verbose: bool = True):
        self.max_jobs = max(1, int(max_jobs))
        self.reserve = reserve_bytes
        self.interval = interval
        self.verbose = verbose
        self.enabled = psutil is not None and self.max_jobs > 1
        self.target = 1 if self.enabled else self.max_jobs
        self._next_probe = time.monotonic() + interval
        self._peak_per_task = 0

    def _pool_rss(self) -> int:
        """Resident memory of this process and its children, in bytes."""
        me = psutil.Process()
        total = me.memory_info().rss
        for c in me.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:          # exited between listing and query
                continue
        return total

    def update(self, in_flight: int) -> int:
        """Resample if the interval has elapsed and return the target.

        Parameters
        ----------
        in_flight : int
            Tasks currently running, used to estimate per-task memory.

        Returns
        -------
        int
            Number of tasks that may be in flight.
        """
        if not self.enabled or time.monotonic() < self._next_probe:
            return self.target
        self._next_probe = time.monotonic() + self.interval

        avail = psutil.virtual_memory().available
        if in_flight > 0:
            self._peak_per_task = max(self._peak_per_task,
                                      self._pool_rss() // max(1, in_flight))
        load = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
        cores = psutil.cpu_count() or self.max_jobs

        old = self.target
        if avail < self.reserve:
            self.target = max(1, self.target - 1)
            why = 'memory below reserve'
        elif (self.target < self.max_jobs
                and avail > self.reserve + self._peak_per_task
                and load < cores * (1.0 - CPU_HEADROOM)):
            self.target += 1
            why = 'headroom available'
        else:
            return self.target

        if self.verbose and self.target != old:
            print(f"\n    [workers {old} -> {self.target}: {why}; "
                  f"{avail / 2**30:.1f} GiB free, load {load:.1f}]",
                  end='', flush=True)
        return self.target


# =============================================================================
# SCREENING SAMPLE
# =============================================================================

def sample_screening_sets(gen, n_landscapes: int, seed: int) -> pd.DataFrame:
    """Draw the parameter sets used at every k of the sweep.

    Parameters
    ----------
    gen : module
        The loaded ``01_generate_landscapes`` module.
    n_landscapes : int
        Number of parameter sets to retain.
    seed : int
        Seed set before sampling, for reproducibility.

    Returns
    -------
    pd.DataFrame
        The retained sets, indexed 0..n_landscapes-1, carrying the same
        columns ``01_generate_landscapes`` produces.

    Raises
    ------
    SystemExit
        If the constraints leave fewer sets than requested.

    Notes
    -----
    Sampling and filtering are delegated to the forward model's own
    ``sample_le_parameters`` and ``apply_constraints``, so the screening
    sample is drawn from exactly the parameter space the production ensemble
    is drawn from.  Were it drawn any other way the multiplier estimated here
    would not transfer to the ensemble it is meant to govern.
    """
    np.random.seed(seed)
    df = gen.apply_constraints(gen.sample_le_parameters())
    if len(df) < n_landscapes:
        sys.exit(f"ERROR: only {len(df)} parameter sets survived the "
                 f"constraints; {n_landscapes} requested.")
    df = df.iloc[:n_landscapes].copy().reset_index(drop=True)
    df['landscape_idx'] = df.index.values
    return df


# =============================================================================
# SWEEP
# =============================================================================

def _build_jobs(sets: pd.DataFrame, k: float, args) -> list:
    """Assemble one picklable payload per landscape for this multiplier.

    Parameters
    ----------
    sets : pd.DataFrame
        Screening parameter sets from ``sample_screening_sets``.
    k : float
        Model time multiplier, T_ss = k / Ks.
    args : argparse.Namespace

    Returns
    -------
    list of dict
    """
    jobs = []
    for idx in sets.index:
        u, kh, ks, dx, ly = sets.loc[idx, ['u', 'kh', 'ks', 'dx', 'ly']]
        jobs.append({
            'k': float(k), 'landscape_idx': int(idx),
            'u': float(u), 'kh': float(kh), 'ks': float(ks),
            'ly': float(ly), 'dx': float(dx),
            'num_ts': args.num_ts, 'final_window': args.final_window,
            'ly_mult': args.ly_mult, 'seed': args.seed,
            'delta': args.delta,
        })
    return jobs


def screen_one_k(sets: pd.DataFrame, k: float, args, limiter=None) -> tuple:
    """Evolve the screening sample at one multiplier and measure each curve.

    Parameters
    ----------
    sets : pd.DataFrame
        Screening parameter sets from ``sample_screening_sets``.
    k : float
        Model time multiplier for this pass, T_ss = k / Ks.
    args : argparse.Namespace
    limiter : AdaptiveLimiter or None
        Concurrency controller. None, or a maximum of one job, runs the
        landscapes in this process.

    Returns
    -------
    rows : list of dict
        One record per landscape, ordered by landscape index.
    curves : np.ndarray, shape (n_landscapes, num_ts)
        Mean elevation series normalized by their own final value, in the
        same order.

    Notes
    -----
    Landscapes within one multiplier are independent and are the unit of
    work. Multipliers are not run concurrently: the sweep stops at the first
    k that passes, so starting a later one would mean speculative work that
    is usually discarded.

    Results are collected into a dictionary keyed by landscape index and
    emitted in index order, so the output does not depend on which worker
    ran a task or on completion order. With ``--jobs 1`` the same ``_task``
    runs in this process, so serial and parallel runs produce identical
    output.
    """
    jobs = _build_jobs(sets, k, args)

    if limiter is None or limiter.max_jobs <= 1:
        done = [_task(j) for j in jobs]
        for _ in done:
            print('.', end='', flush=True)
    else:
        done = _run_parallel(jobs, limiter)

    done.sort(key=lambda r: r[0])
    rows = [r[1] for r in done]
    series = [r[2] for r in done]
    return rows, np.vstack(series)


def _run_parallel(jobs: list, limiter) -> list:
    """Run the payloads across a pool, keeping the target number in flight.

    Parameters
    ----------
    jobs : list of dict
        Payloads for ``_task``.
    limiter : AdaptiveLimiter
        Consulted before each submission.

    Returns
    -------
    list of tuple
        ``_task`` results, in completion order.

    Notes
    -----
    The pool is created at the hard ceiling, but only ``limiter.target``
    tasks are allowed in flight at once; concurrency is therefore varied by
    withholding submissions rather than by resizing the pool, which
    ``ProcessPoolExecutor`` does not support. Idle workers are forked
    children sharing the parent's pages, so holding a few costs little.

    Submitting as slots free also balances the load on its own: landscape
    runtimes vary with grid size and with the time step, so a static split
    would leave workers idle behind the slowest task in their share.
    """
    results, pending = [], list(jobs)
    in_flight = set()

    with ProcessPoolExecutor(max_workers=limiter.max_jobs) as ex:
        while pending or in_flight:
            target = limiter.update(len(in_flight))
            while pending and len(in_flight) < target:
                in_flight.add(ex.submit(_task, pending.pop(0)))

            # The target never falls below one, so work is always in flight
            # here while any remains. The timeout returns control even when
            # nothing finishes, which is how the limiter resamples during
            # long tasks.
            finished, in_flight = wait(in_flight, timeout=limiter.interval,
                                       return_when=FIRST_COMPLETED)
            for fut in finished:
                results.append(fut.result())
                print('.', end='', flush=True)

    return results


def run_sweep(sets: pd.DataFrame, args, limiter=None) -> tuple:
    """Evolve the screening sample at each k until one passes.

    Parameters
    ----------
    sets : pd.DataFrame
    args : argparse.Namespace
    limiter : AdaptiveLimiter or None

    Returns
    -------
    res : pd.DataFrame
        One row per (k, landscape).
    curves : dict
        ``{k: (n_landscapes, num_ts) array}`` of normalized series.
    k_chosen : float or None
        Smallest k at which every curve passed, or None if none did.

    Notes
    -----
    The sweep stops at the first k that passes, which is the procedure the
    manuscript reports; ``--no-early-stop`` runs every requested value so the
    panels above the chosen one can be inspected too.
    """
    rows, curves, k_chosen = [], {}, None

    for k in args.k_values:
        print(f"  k = {k:<5g} ", end='', flush=True)
        r, c = screen_one_k(sets, k, args, limiter)
        rows.extend(r)
        curves[k] = c

        n_pass = sum(d['passed'] for d in r)
        worst = max(d['R_w'] for d in r)
        print(f"  {n_pass}/{len(r)} passed, worst R_w = {worst:.2e}")

        if n_pass == len(r) and k_chosen is None:
            k_chosen = k
            if not args.no_early_stop:
                break

    res = pd.DataFrame(rows)
    return res[[c for c in COLUMN_ORDER if c in res.columns]], curves, k_chosen


# =============================================================================
# FIGURE
# =============================================================================

def make_figure(curves: dict, args, k_chosen=None):
    """Plot the normalized curves, one panel per multiplier.

    Parameters
    ----------
    curves : dict
        ``{k: (n_landscapes, num_ts) array}`` from ``run_sweep``.
    args : argparse.Namespace
    k_chosen : float or None
        Multiplier that passed, marked in its panel title.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Every curve is drawn individually rather than as an envelope.  The sample
    is small enough that the curves are legible, and the criterion is whether
    each one has flattened, which an envelope would conceal.

    Panels are laid out in a grid of ``--fig-cols`` columns, so six
    multipliers fall into two rows of three rather than one row of six,
    which fits a page without shrinking each panel to a sliver.  Axes beyond
    the number of multipliers are removed rather than left blank.  Axis
    labels are drawn only on the outer edges, since the panels share both
    scales.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    chk = _load_sibling('01d_check_steady_state')
    chk._style()

    ks_sorted = sorted(curves)
    n = len(ks_sorted)
    ncols = max(1, min(args.fig_cols, n))
    nrows = -(-n // ncols)                # ceiling division
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.9 * ncols + 0.6, 2.6 * nrows + 0.3),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()

    t_w = 1.0 - args.final_window / args.num_ts
    for ax, k in zip(flat, ks_sorted):
        Z = curves[k]
        tn = (np.arange(Z.shape[1]) + 1) / Z.shape[1]
        ax.add_patch(Rectangle((t_w, 0), 1 - t_w, 1,
                               transform=ax.get_xaxis_transform(),
                               facecolor=chk.C_WIN, edgecolor='none',
                               zorder=1))
        ax.plot(tn, Z.T, color=chk.C_ENS, lw=0.7, alpha=0.55, zorder=2)
        ax.axhline(1.0, color=chk.INK2, lw=0.6, ls=(0, (4, 3)), zorder=3)
        ax.grid(True, color=chk.GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(0, 1)
        mark = '  (chosen)' if k_chosen is not None and k == k_chosen else ''
        ax.set_title(f'$k$ = {k:g}{mark}')

    for ax in flat[n:]:                   # grid wider than the sweep
        ax.remove()

    # Outer edges only: the panels share both scales, so repeating the
    # labels inside the grid adds ink without adding information.
    for ax in flat[:n][-ncols:]:
        ax.set_xlabel('$t^*$')
        # sharex labels only the grid's last row; a panel left exposed by a
        # removed neighbour needs its tick labels restored.
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        if ax in set(flat[:n]):
            ax.set_ylabel('$\\bar z\\,/\\,\\bar z(t^*{=}1)$')
    fig.tight_layout()
    return fig


# =============================================================================
# REPORTING
# =============================================================================

def derive_out_path(args, git_hash: str, ext: str) -> Path:
    """Build an output path from the CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
    git_hash : str
        Short commit hash, or 'unknown'.
    ext : str
        File extension without the dot.

    Returns
    -------
    Path

    Notes
    -----
    The hash is last, immediately before the extension, matching the
    convention in ``01d_check_steady_state``.
    """
    stem = args.out_stem or (f'screen-model-time-n{args.n_landscapes}'
                             f'-ts{args.num_ts}-{git_hash}')
    return Path(args.out_dir) / f'{stem}.{ext}'


def print_summary(res: pd.DataFrame, args, k_chosen):
    """Report the sweep, one line per multiplier, then the verdict.

    Parameters
    ----------
    res : pd.DataFrame
    args : argparse.Namespace
    k_chosen : float or None
    """
    print(f"\nScreening sample: {args.n_landscapes} landscapes | "
          f"grid Lx = {args.ly_mult}*Ly | steps: {args.num_ts} | "
          f"seed: {args.seed}")
    print(f"Support criterion: R_w < {args.delta:g} over the final "
          f"{args.final_window} points\n")

    print(f"  {'k':>6} {'passed':>9} {'median R_w':>12} {'worst R_w':>12} "
          f"{'max cfl':>10}")
    for k, g in res.groupby('k', sort=True):
        print(f"  {k:>6g} {int(g['passed'].sum()):>4d}/{len(g):<4d} "
              f"{g['R_w'].median():>12.3e} {g['R_w'].max():>12.3e} "
              f"{g['cfl'].max():>10.2f}")

    if k_chosen is None:
        print(f"\nNo multiplier in {list(args.k_values)} passed for every "
              f"landscape. Extend --k-values upward, or inspect the figure: "
              f"the curves may be flattening while R_w remains above delta.")
        return

    print(f"\nSmallest multiplier at which every curve passed: k = "
          f"{k_chosen:g}")
    print(f"  Set T_TC = {k_chosen:g} in 01_generate_landscapes.py.")

    g = res[res['k'] == k_chosen]
    if (g['cfl'] > 1).any():
        n_sub = int((g['cfl'] > 1).sum())
        print(f"  Note: dt exceeds the explicit diffusion limit for "
              f"{n_sub} of {len(g)} landscapes (max cfl "
              f"{g['cfl'].max():.1f}); LinearDiffuser subdivides those steps, "
              f"so they cost more than the step count suggests.")

    print("\n  The sweep is evidence about this sample on a square grid, not "
          "a guarantee for the production ensemble. Every landscape is "
          "verified individually by 01c after generation.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--k-values', type=float, nargs='+', default=list(K_VALUES),
                   help='Model time multipliers to sweep, ascending. '
                        f'Default: {" ".join(str(k) for k in K_VALUES)}.')
    p.add_argument('--n-landscapes', type=int, default=N_SCREEN,
                   help=f'Screening sample size. Default: {N_SCREEN}.')
    p.add_argument('--seed', type=int, default=SCREEN_SEED,
                   help='Seed for parameter sampling and, with the landscape '
                        f'index, for the initial surfaces. Default: '
                        f'{SCREEN_SEED}.')
    p.add_argument('--ly-mult', type=int, default=LY_MULT_SCREEN,
                   help='Grid length multiplier, Lx = ly_mult * Ly. 1 is the '
                        'square grid the screening uses; the production '
                        f'ensemble uses 5. Default: {LY_MULT_SCREEN}.')
    p.add_argument('--num-ts', type=int, default=None,
                   help='Time steps per run. Default: NUM_TS from '
                        '01_generate_landscapes, so the screening series has '
                        'the same length as the production series.')
    p.add_argument('--final-window', type=int, default=FINAL_WINDOW,
                   help='Trailing points over which R_w is measured. Match '
                        f'--final-window in 01c. Default: {FINAL_WINDOW}.')
    p.add_argument('--delta', type=float, default=1e-3,
                   help='Threshold for the support criterion, R_w < delta. '
                        'Default: 1e-3.')
    p.add_argument('--jobs', type=int, default=0,
                   help='Maximum worker processes for the landscapes of one '
                        'multiplier. 0 uses the CPU count, 1 runs in this '
                        'process. The pool starts at one worker and grows '
                        'towards this ceiling as memory and CPU allow. '
                        'Default: 0.')
    p.add_argument('--mem-reserve-gb', type=float, default=MEM_RESERVE_GB,
                   help='Memory held back from the pool, in GiB. The worker '
                        'count is reduced below this and raised only with '
                        'this much plus a further task to spare. Default: '
                        f'{MEM_RESERVE_GB}.')
    p.add_argument('--probe-interval', type=float, default=PROBE_INTERVAL_S,
                   help='Seconds between load samples. Default: '
                        f'{PROBE_INTERVAL_S}.')
    p.add_argument('--no-early-stop', action='store_true',
                   help='Sweep every requested k instead of stopping at the '
                        'first that passes.')
    p.add_argument('--fig-cols', type=int, default=FIG_COLS,
                   help='Panel columns in the figure; rows follow from the '
                        'number of multipliers. Default: '
                        f'{FIG_COLS}.')
    p.add_argument('--no-plot', dest='plot', action='store_false',
                   help='Write the CSV only.')
    p.add_argument('--formats', nargs='+', default=['png'],
                   help='Figure formats to write. Default: png.')
    p.add_argument('--out-dir', default='data/screening',
                   help='Directory for output files. Default: data/screening.')
    p.add_argument('--out-stem', default=None,
                   help='Override the derived filename stem.')
    args = p.parse_args()

    args.k_values = sorted(args.k_values)
    if args.final_window < 3:
        sys.exit('ERROR: --final-window must be at least 3.')

    # Loaded here so forked workers inherit them rather than re-executing.
    gen, _ = _worker_modules()
    if args.num_ts is None:
        args.num_ts = gen.NUM_TS
    if args.final_window >= args.num_ts:
        sys.exit('ERROR: --final-window must be smaller than --num-ts.')

    max_jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    max_jobs = max(1, min(max_jobs, args.n_landscapes))
    limiter = AdaptiveLimiter(max_jobs,
                              int(args.mem_reserve_gb * 2 ** 30),
                              args.probe_interval)

    print(f"Screening model time over k = "
          f"{', '.join(f'{k:g}' for k in args.k_values)}")
    if max_jobs == 1:
        print("  Running serially.")
    elif limiter.enabled:
        print(f"  Up to {max_jobs} workers, starting at 1 and growing while "
              f"more than {args.mem_reserve_gb:g} GiB stays free.")
    else:
        print(f"  {max_jobs} workers, fixed: psutil is not installed, so "
              f"load cannot be sampled.")

    sets = sample_screening_sets(gen, args.n_landscapes, args.seed)
    res, curves, k_chosen = run_sweep(sets, args, limiter)

    git_hash = _git_hash()
    csv_path = derive_out_path(args, git_hash, 'csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(csv_path, index=False)

    # Written before the figure: a sweep is expensive, and the curves are
    # the only part that cannot be recovered from the CSV. With these on
    # disk the series can be re-inspected, and the figure redrawn, without
    # evolving a single landscape again.
    npz_path = derive_out_path(args, git_hash, 'npz')
    np.savez_compressed(npz_path,
                        **{f'k{k:g}': c for k, c in curves.items()})

    print_summary(res, args, k_chosen)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {npz_path}")

    if args.plot:
        fig = make_figure(curves, args, k_chosen)
        for ext in args.formats:
            fig_path = derive_out_path(args, git_hash, ext)
            fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
            print(f"Wrote {fig_path}")


if __name__ == '__main__':
    main()
