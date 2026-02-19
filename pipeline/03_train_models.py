"""
03_train_models.py
------------------
Nested cross-validation training pipeline for the landscape-evolution ML
framework.  Consumes feature DataFrames produced by ``02_extract_features.py``
and writes a results pickle that is later consumed by
``04_test_hypotheses.py``.

Workflow
--------
1. Load and concatenate ``features-{job_id}.pkl`` files from ``--data-dir``.
2. Split into features (X) and log₁₀-transformed labels (y).
3. Hold out a final test set (``--test-fraction``).
4. Run nested cross-validation on the remaining training data:
   - Outer loop: ShuffleSplit with ``--n-outer`` folds evaluates
     generalisation performance.
   - Inner loop: ShuffleSplit with ``--n-inner`` folds tunes hyperparameters
     via ``RandomizedSearchCV`` (``--n-iter`` candidates per fold).
5. Select final hyperparameters across folds (frequency → score tiebreaker).
6. Train final models on the full training set; evaluate on held-out test set.
7. Save results pickle and diagnostic figures to ``--output-dir``.

Usage
-----
    python 03_train_models.py \\
        --data-dir data/features \\
        --output-dir outputs \\
        --labels log_u_ks log_kh_ks \\
        --n-outer 10 --n-inner 5 --n-iter 20

    # Train on individual parameters instead of ratios
    python 03_train_models.py --labels u kh ks

    # Restrict to specific job IDs
    python 03_train_models.py --job-ids 1001 1002 1003
"""

import argparse
import glob
import os
import pickle
import subprocess
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler




from display_labels import TARGET_DISPLAY as _TARGET_DISPLAY

def _target_label(col: str) -> str:
    """Return the LaTeX display label for target column *col*."""
    return _TARGET_DISPLAY.get(col, {}).get('label', col)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _git_hash(short: bool = True) -> str:
    """Return current git commit hash, or 'unknown' if git is unavailable."""
    try:
        cmd = ['git', 'rev-parse', '--short' if short else '', 'HEAD']
        return subprocess.check_output(
            [c for c in cmd if c], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def load_features(data_dir: str, job_ids: list = None) -> pd.DataFrame:
    """Load and concatenate ``features-{job_id}.pkl`` files.

    Parameters
    ----------
    data_dir : str
        Directory containing feature DataFrames written by
        ``02_extract_features.py``.
    job_ids : list of int or str, optional
        If given, load only those job IDs.  Otherwise load all matching files.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with LE parameters and topographic features.
    """
    if job_ids:
        paths = [os.path.join(data_dir, f'features-{jid}.pkl') for jid in job_ids]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Feature files not found: {missing}"
            )
    else:
        pattern = os.path.join(data_dir, 'features-*.pkl')
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No feature files found matching '{pattern}'"
            )

    dfs = []
    for p in paths:
        with open(p, 'rb') as fh:
            dfs.append(pickle.load(fh))

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} landscapes from {len(paths)} file(s).")
    return df


def split_features_labels(df: pd.DataFrame, label_cols: list):
    """Split a combined DataFrame into feature matrix and label matrix.

    Parameters
    ----------
    df : pd.DataFrame
    label_cols : list of str
        Target columns to use as y.  Both raw LE parameters (``u``, ``kh``,
        ``ks``) and pre-computed log-ratios (``log_u_ks``, ``log_kh_ks``) are
        valid.

    Returns
    -------
    X : pd.DataFrame  — topographic features (all columns not recognised as
                         LE parameters or derived labels)
    y : pd.DataFrame  — selected labels, log₁₀-transformed where needed
    """
    # Columns that are LE parameters or derived quantities (not features)
    non_feature_cols = {'u', 'kh', 'ks', 'log_u_ks', 'log_kh_ks', 'u_kh',
                        'u_ks', 'kh_ks'}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols]

    # Derived log-ratio columns are already in log₁₀ space; raw LE parameters
    # need to be log-transformed here.
    raw_cols   = [c for c in label_cols if c in {'u', 'kh', 'ks', 'u_kh', 'u_ks', 'kh_ks'}]
    ratio_cols = [c for c in label_cols if c in {'log_u_ks', 'log_kh_ks'}]

    y_parts = []
    if ratio_cols:
        y_parts.append(df[ratio_cols])
    if raw_cols:
        y_parts.append(np.log10(df[raw_cols]))

    y = pd.concat(y_parts, axis=1)[label_cols]   # preserve requested order
    return X, y


# ---------------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------------

def nested_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_outer_splits: int = 10,
    n_inner_splits: int = 5,
    random_state: int = 42,
    n_iter: int = 20,
) -> dict:
    """Run nested cross-validation for all eight algorithms.

    Parameters
    ----------
    X : pd.DataFrame  — feature matrix (training set only)
    y : pd.DataFrame  — log₁₀ target matrix (training set only)
    n_outer_splits : int — number of outer ShuffleSplit folds
    n_inner_splits : int — number of inner ShuffleSplit folds for tuning
    random_state : int
    n_iter : int — number of RandomizedSearchCV candidates per inner fold

    Returns
    -------
    dict  — nested results keyed by algorithm shortname
    """
    models    = get_random_search_params()
    toolkit   = MultiOutputRegressionToolkit(models)
    reg_names = list(models.keys())

    results = {}

    for reg_name in reg_names:
        print(f"\n{'='*60}\n  Algorithm: {reg_name}\n{'='*60}")

        outer_cv = ShuffleSplit(
            n_splits=n_outer_splits, test_size=0.2, random_state=random_state
        )
        inner_cv = ShuffleSplit(
            n_splits=n_inner_splits, test_size=0.2, random_state=random_state
        )

        results[reg_name] = {
            'test_r2':        [],
            'test_mse':       [],
            'test_rmse':      [],
            'test_mae':       [],
            'best_params':    [],
            'per_target_r2':  [],
            'per_target_mse': [],
            'per_target_rmse':[],
            'per_target_mae': [],
        }

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
            print(f"  Outer fold {fold_idx}/{n_outer_splits}", end='', flush=True)

            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            regressor  = toolkit.get_regressor(reg_name)
            param_dist = toolkit.get_param_dict(reg_name)

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('piped',  regressor),
            ])

            if param_dist:
                search = RandomizedSearchCV(
                    pipe,
                    param_dist,
                    n_iter=n_iter,
                    cv=inner_cv,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=random_state,
                )
                search.fit(X_tr, y_tr)
                best_params    = search.best_params_
                best_estimator = search.best_estimator_
            else:
                # LinearRegression: no tunable params beyond the tiny discrete grid
                best_params    = {}
                best_estimator = pipe
                best_estimator.fit(X_tr, y_tr)

            y_pred = best_estimator.predict(X_te)
            if isinstance(y_te, pd.DataFrame) and isinstance(y_pred, np.ndarray):
                y_pred = pd.DataFrame(y_pred, index=y_te.index, columns=y_te.columns)

            # Overall metrics
            r2   = r2_score(y_te, y_pred, multioutput='uniform_average')
            mse  = mean_squared_error(y_te, y_pred, multioutput='uniform_average')
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(y_te, y_pred, multioutput='uniform_average')
            print(f"  →  R² = {r2:.3f}")

            # Per-target metrics
            pt_r2   = r2_score(y_te, y_pred, multioutput='raw_values')
            pt_mse  = mean_squared_error(y_te, y_pred, multioutput='raw_values')
            pt_rmse = np.sqrt(pt_mse)
            pt_mae  = mean_absolute_error(y_te, y_pred, multioutput='raw_values')

            results[reg_name]['test_r2'].append(r2)
            results[reg_name]['test_mse'].append(mse)
            results[reg_name]['test_rmse'].append(rmse)
            results[reg_name]['test_mae'].append(mae)
            results[reg_name]['best_params'].append(best_params)
            results[reg_name]['per_target_r2'].append(pt_r2)
            results[reg_name]['per_target_mse'].append(pt_mse)
            results[reg_name]['per_target_rmse'].append(pt_rmse)
            results[reg_name]['per_target_mae'].append(pt_mae)

        results[reg_name]['mean_test_r2']   = np.mean(results[reg_name]['test_r2'])
        results[reg_name]['std_test_r2']    = np.std(results[reg_name]['test_r2'])
        results[reg_name]['mean_test_mse']  = np.mean(results[reg_name]['test_mse'])
        results[reg_name]['mean_test_rmse'] = np.mean(results[reg_name]['test_rmse'])
        results[reg_name]['mean_test_mae']  = np.mean(results[reg_name]['test_mae'])

        print(
            f"  {reg_name}: mean R² = {results[reg_name]['mean_test_r2']:.3f} "
            f"± {results[reg_name]['std_test_r2']:.3f}"
        )

    return results


# ---------------------------------------------------------------------------
# Final model selection and evaluation
# ---------------------------------------------------------------------------

def select_final_hyperparameters(reg_results: dict) -> tuple:
    """Choose final hyperparameters from outer-fold best-params lists.

    Strategy (hierarchical):
    1. If any complete parameter set occurs in more than one fold, select the
       most frequent.  Ties broken by the outer-fold R² score.
    2. Otherwise select the parameter set from the highest-scoring fold.

    Parameters
    ----------
    reg_results : dict
        Results dict for a *single* algorithm (one entry of the outer results
        dict, i.e. ``results[reg_name]``).

    Returns
    -------
    final_params : dict
    selection_method : str
        One of ``'frequency'``, ``'frequency_with_score_tiebreaker'``,
        ``'best_score'``.
    """
    occurrences = {}
    for params in reg_results['best_params']:
        key = tuple(sorted(params.items()))
        occurrences[key] = occurrences.get(key, 0) + 1

    max_count = max(occurrences.values())

    if max_count > 1:
        candidates = [k for k, v in occurrences.items() if v == max_count]
        if len(candidates) == 1:
            return dict(candidates[0]), 'frequency'

        # Tie-break by score
        best_score, best_params = -np.inf, None
        for cand in candidates:
            cand_dict = dict(cand)
            for params, score in zip(
                reg_results['best_params'], reg_results['test_r2']
            ):
                if set(params.items()) == set(cand):
                    if score > best_score:
                        best_score  = score
                        best_params = cand_dict
        return best_params, 'frequency_with_score_tiebreaker'

    best_idx = int(np.argmax(reg_results['test_r2']))
    return reg_results['best_params'][best_idx], 'best_score'


def evaluate_final_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_test:  pd.DataFrame,
    results: dict,
    output_dir: str,
    git_hash: str,
    figure_width_cm: float = 14.0,
) -> dict:
    """Train final models on full training set; evaluate on held-out test set.

    Hyperparameters are selected from the nested-CV results via
    ``select_final_hyperparameters``.  Trained model objects, metrics, and
    figures are stored back into *results* and written to *output_dir*.

    Parameters
    ----------
    X_train, y_train : training data
    X_test,  y_test  : held-out test data (never seen during CV)
    results : dict   — output of ``nested_cv()``; mutated in-place
    output_dir : str
    git_hash : str
    figure_width_cm : float

    Returns
    -------
    results : dict  (same object, updated with 'final_model' key per algorithm)
    """
    models  = get_random_search_params()
    toolkit = MultiOutputRegressionToolkit(models)
    labels  = y_train.columns
    fw      = figure_width_cm / 2.54

    # Per-target comparison figures (all models on one canvas per target)
    fig_per_target, ax_per_target = {}, {}
    for col in labels:
        n_models = len(results)
        ncols = min(4, n_models)
        nrows = int(np.ceil(n_models / ncols))
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            sharex=True, sharey=True,
            figsize=(fw, fw * 0.6 * nrows / 2),
        )
        fig_per_target[col] = fig
        ax_per_target[col]  = axes.flatten()

    for model_idx, reg_name in enumerate(results):
        print(f"\nFinal model: {reg_name}")

        final_params, method = select_final_hyperparameters(results[reg_name])
        print(f"  Hyperparameter selection: {method}")

        # Build pipeline with selected hyperparameters
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('piped',  toolkit.get_regressor(reg_name, clone_regressor=True)),
        ])
        for param, value in final_params.items():
            try:
                pipe.set_params(**{param: value})
            except ValueError as exc:
                warnings.warn(
                    f"Could not set {param}={value!r} for {reg_name}: {exc}"
                )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Per-target metrics (append overall mean as last element)
        pt_r2   = list(r2_score(y_test, y_pred, multioutput='raw_values'))
        pt_rmse = list(mean_squared_error(y_test, y_pred, multioutput='raw_values') ** 0.5)
        pt_mae  = list(mean_absolute_error(y_test, y_pred, multioutput='raw_values'))

        results[reg_name]['final_model'] = {
            'hyperparams':    final_params,
            'selection_method': method,
            'regressor':      pipe,
            'test_set_r2':    pt_r2   + [r2_score(y_test, y_pred)],
            'test_set_rmse':  pt_rmse + [mean_squared_error(y_test, y_pred) ** 0.5],
            'test_set_mae':   pt_mae  + [mean_absolute_error(y_test, y_pred)],
        }

        # ----------------------------------------------------------------
        # Per-model figure: predicted vs true + residuals for each target
        # ----------------------------------------------------------------
        n_targets = y_train.shape[1]
        fig_m, ax_m = plt.subplots(
            nrows=n_targets, ncols=2,
            figsize=(9 / 2.54, 1.8 * n_targets),
        )
        if n_targets == 1:
            ax_m = ax_m[np.newaxis, :]  # ensure 2-D indexing

        scatter_kw = dict(alpha=0.2, lw=0, marker='o', ms=2)
        fs_ax = 7

        for t_idx, col in enumerate(labels):
            display = _target_label(col)
            y_t = y_test.loc[:, col].values
            y_p = y_pred[:, t_idx] if y_pred.ndim > 1 else y_pred

            lims = [y_t.min(), y_t.max()]

            # Predicted vs true
            ax_m[t_idx, 0].plot(y_p, y_t, **scatter_kw)
            ax_m[t_idx, 0].plot(lims, lims, ls=':', c='k', label='1:1')
            ax_m[t_idx, 0].set_xlabel(f'predicted log {display}', fontsize=fs_ax)
            ax_m[t_idx, 0].set_ylabel(f'true log {display}', fontsize=fs_ax)
            ax_m[t_idx, 0].tick_params(labelsize=fs_ax)
            ax_m[t_idx, 0].text(
                0.05, 0.95, reg_name, fontsize=9,
                va='top', transform=ax_m[t_idx, 0].transAxes,
            )
            metric_str = '\n'.join([
                f'$R^2$={results[reg_name]["final_model"]["test_set_r2"][t_idx]:.3f}',
                f'RMSE={results[reg_name]["final_model"]["test_set_rmse"][t_idx]:.3f}',
                f'MAE={results[reg_name]["final_model"]["test_set_mae"][t_idx]:.3f}',
            ])
            ax_m[t_idx, 0].text(
                0.95, 0.05, metric_str, fontsize=fs_ax,
                va='bottom', ha='right', transform=ax_m[t_idx, 0].transAxes,
            )

            # Residuals
            ax_m[t_idx, 1].plot(y_p, y_t - y_p, **scatter_kw)
            ax_m[t_idx, 1].axhline(0, ls=':', c='k')
            ax_m[t_idx, 1].set_xlabel(f'predicted log {display}', fontsize=fs_ax)
            ax_m[t_idx, 1].set_ylabel(f'residual log {display}', fontsize=fs_ax)
            ax_m[t_idx, 1].tick_params(labelsize=fs_ax)

            # Add to per-target comparison canvas
            cur_ax = ax_per_target[col][model_idx]
            cur_ax.plot(y_p, y_t, **scatter_kw)
            cur_ax.plot(lims, lims, ls=':', c='k')
            cur_ax.tick_params(labelsize=fs_ax)
            cur_ax.text(
                0.02, 0.98, reg_name, fontsize=9,
                va='top', transform=cur_ax.transAxes,
            )
            metric_str2 = '\n'.join([
                f'$R^2$={results[reg_name]["final_model"]["test_set_r2"][t_idx]:.3f}',
                f'RMSE={results[reg_name]["final_model"]["test_set_rmse"][t_idx]:.3f}',
            ])
            cur_ax.text(
                0.95, 0.05, metric_str2, fontsize=fs_ax,
                va='bottom', ha='right', transform=cur_ax.transAxes,
            )

        fig_m.tight_layout()
        stem = os.path.join(output_dir, f'pred_vs_true-{reg_name}-{git_hash}')
        fig_m.savefig(stem + '.png', dpi=300)
        fig_m.savefig(stem + '.svg')
        plt.close(fig_m)

    # Save per-target comparison figures
    for col in labels:
        display = _target_label(col)
        fig = fig_per_target[col]
        fig.tight_layout()
        stem = os.path.join(output_dir, f'pred_vs_true-all_models-{col}-{git_hash}')
        fig.savefig(stem + '.png', dpi=300)
        fig.savefig(stem + '.svg')
        plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# Plotting: nested CV results
# ---------------------------------------------------------------------------

def plot_nested_cv_results(
    results: dict,
    y: pd.DataFrame,
    output_dir: str,
    git_hash: str,
    figure_width_cm: float = 14.0,
    box: bool = True,
):
    """Box-plot or scatter-plot nested-CV performance across folds.

    Parameters
    ----------
    results : dict   — output of ``nested_cv()``
    y : pd.DataFrame — training targets (used only for column names)
    output_dir : str
    git_hash : str
    figure_width_cm : float
    box : bool       — True → box plots, False → scatter + mean marker
    """
    fw     = figure_width_cm / 2.54
    fasp   = 0.3
    col0   = list(results.keys())
    col1   = ['r2', 'rmse', 'mae']
    col1_L = [r'$R^2$', r'$RMSE$', r'$MAE$']
    col2   = list(range(len(y.columns))) + ['mean']

    # Assemble results DataFrame
    idx  = pd.MultiIndex.from_product([col0, col1, col2])
    rdf  = pd.DataFrame(columns=idx)
    for c0 in col0:
        for c1 in col1:
            for c2 in col2:
                if c2 == 'mean':
                    rdf[(c0, c1, c2)] = results[c0][f'test_{c1}']
                else:
                    rdf[(c0, c1, c2)] = np.array(
                        results[c0][f'per_target_{c1}']
                    ).T[c2]

    # Sort algorithms by mean R²
    mean_r2 = [results[c0]['mean_test_r2'] for c0 in col0]
    col0_sorted = list(np.array(col0)[np.argsort(mean_r2)])

    box_kw = dict(
        boxprops    ={'linewidth': 0.5, 'color': '0.5'},
        whiskerprops={'linewidth': 0.5, 'color': '0.5'},
        capprops    ={'linewidth': 0.5, 'color': '0.5'},
        flierprops  ={'marker': 'o', 'mew': 0.5, 'ms': 2, 'color': '0.5'},
        medianprops ={'color': '0.5'},
        meanprops   ={'marker': '^', 'mfc': 'k', 'mew': 0, 'mec': 'k',
                      'alpha': 1, 'ms': 5},
        showmeans=True,
    )

    for c_idx, (c1, c1_label) in enumerate(zip(col1, col1_L)):
        fig, axes = plt.subplots(
            ncols=len(col2), sharey=True,
            figsize=(fw, fw * fasp),
        )
        for i, c2 in enumerate(col2):
            ax = axes[i]
            data = [rdf[(c0, c1, c2)] for c0 in col0_sorted]

            if box:
                ax.boxplot(data, tick_labels=col0_sorted, **box_kw)
            else:
                has_leg = False
                for j, c0 in enumerate(col0_sorted):
                    kw = dict(marker='.', mfc='0.5', mec='none', alpha=0.5, lw=0)
                    if not has_leg:
                        ax.plot(np.full(len(rdf[(c0, c1, c2)]), j),
                                rdf[(c0, c1, c2)], label='cv scores', **kw)
                        ax.plot(j, rdf[(c0, c1, c2)].mean(),
                                marker='_', mec='k', mfc='none', lw=0,
                                label='mean')
                        has_leg = True
                    else:
                        ax.plot(np.full(len(rdf[(c0, c1, c2)]), j),
                                rdf[(c0, c1, c2)], **kw)
                        ax.plot(j, rdf[(c0, c1, c2)].mean(),
                                marker='_', mec='k', mfc='none', lw=0)
                ax.set_xticks(range(len(col0_sorted)))
                ax.set_xticklabels(col0_sorted)

            ax.tick_params(axis='both', labelsize=7)
            ax.tick_params(axis='x', rotation=60)

            if c2 == 'mean':
                text = 'mean'
            else:
                col_name = y.columns[c2]
                text = f'log {_target_label(col_name)}'

            ha = 'left' if c1 == 'r2' else 'right'
            va = 'top'
            tx = 0.05 if c1 == 'r2' else 0.95
            ax.text(tx, 0.98, text, va=va, ha=ha,
                    transform=ax.transAxes, fontsize=10)

            if not box and c2 == 'mean':
                ax.legend(fontsize=6, loc='lower right')

        axes[0].set_ylabel(c1_label, rotation=90, ha='center', fontsize=8)
        fig.tight_layout()

        suffix = '-box' if box else ''
        stem = os.path.join(
            output_dir, f'nested-cv-performance-{c1}{suffix}-{git_hash}'
        )
        fig.savefig(stem + '.png', dpi=300)
        fig.savefig(stem + '.svg')
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Nested CV training for the landscape-evolution ML framework.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--data-dir', default='data/features',
        help='Directory containing features-{job_id}.pkl files.',
    )
    p.add_argument(
        '--output-dir', default='outputs',
        help='Directory for results pickle and figures.',
    )
    p.add_argument(
        '--job-ids', nargs='+', default=None,
        help='Restrict to specific job IDs (e.g. --job-ids 1001 1002).',
    )
    p.add_argument(
        '--labels', nargs='+',
        default=['log_u_ks', 'log_kh_ks'],
        help=(
            'Target columns to train on.  Use log_u_ks / log_kh_ks for '
            'dimensionless ratios (recommended) or u / kh / ks for individual '
            'parameters.'
        ),
    )
    p.add_argument('--test-fraction', type=float, default=0.1,
                   help='Fraction of total data held out as the final test set.')
    p.add_argument('--n-outer',  type=int, default=10,
                   help='Number of outer ShuffleSplit folds.')
    p.add_argument('--n-inner',  type=int, default=5,
                   help='Number of inner ShuffleSplit folds.')
    p.add_argument('--n-iter',   type=int, default=20,
                   help='RandomizedSearchCV candidates per inner fold.')
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument(
        '--skip-cv', action='store_true',
        help='Skip nested CV and load an existing results pickle for the '
             'final-model and plotting steps.',
    )
    p.add_argument(
        '--results-pkl', default=None,
        help='Path to existing results pickle (used with --skip-cv).',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    git_hash = _git_hash()

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    df = load_features(args.data_dir, job_ids=args.job_ids)
    X, y = split_features_labels(df, args.labels)

    print(f"\nFeatures : {X.shape[1]} columns, {X.shape[0]:,} samples")
    print(f"Targets  : {list(y.columns)}")
    print(f"y range  : {y.min().to_dict()} … {y.max().to_dict()}")

    # -----------------------------------------------------------------------
    # 2. Train / test split  (deterministic last-N-rows split preserved from
    #    original notebook; ShuffleSplit version available via --test-fraction)
    # -----------------------------------------------------------------------
    n_test  = max(1, int(len(df) * args.test_fraction))
    n_train = len(df) - n_test

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_test  = X.iloc[n_train:]
    y_test  = y.iloc[n_train:]

    print(f"\nTrain : {len(X_train):,}   Test : {len(X_test):,}")

    # -----------------------------------------------------------------------
    # 3. Nested CV
    # -----------------------------------------------------------------------
    pkl_path = args.results_pkl or os.path.join(
        output_dir := args.output_dir,
        f'nested-cv-results-{git_hash}.pkl',
    )

    if args.skip_cv and pkl_path and os.path.exists(pkl_path):
        print(f"\nLoading existing results from {pkl_path}")
        with open(pkl_path, 'rb') as fh:
            results = pickle.load(fh)
    else:
        results = nested_cv(
            X_train, y_train,
            n_outer_splits=args.n_outer,
            n_inner_splits=args.n_inner,
            random_state=args.random_state,
            n_iter=args.n_iter,
        )
        with open(pkl_path, 'wb') as fh:
            pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nNested-CV results saved → {pkl_path}")

    # -----------------------------------------------------------------------
    # 4. Plot CV results
    # -----------------------------------------------------------------------
    plt.close('all')
    plot_nested_cv_results(results, y_train, args.output_dir, git_hash, box=False)
    plot_nested_cv_results(results, y_train, args.output_dir, git_hash, box=True)

    # -----------------------------------------------------------------------
    # 5. Train final models and evaluate on held-out test set
    # -----------------------------------------------------------------------
    plt.close('all')
    results = evaluate_final_model(
        X_train, y_train, X_test, y_test,
        results, args.output_dir, git_hash,
    )

    # Save updated results (now includes 'final_model' for each algorithm)
    with open(pkl_path, 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nFinal results saved → {pkl_path}")

    # Quick summary table
    print("\n── Final model test-set performance ──")
    for reg_name, reg_res in results.items():
        fm = reg_res.get('final_model', {})
        if fm:
            r2_vals = fm['test_set_r2'][:-1]   # per-target (exclude mean)
            r2_mean = fm['test_set_r2'][-1]
            per = '  '.join(
                f'{c}:{v:.3f}' for c, v in zip(y.columns, r2_vals)
            )
            print(f"  {reg_name:4s}  mean R²={r2_mean:.3f}   [{per}]")


if __name__ == '__main__':
    main()
