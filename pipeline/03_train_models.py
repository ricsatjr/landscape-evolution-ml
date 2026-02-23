"""
03_train_models.py
------------------
Nested cross-validation training pipeline for the landscape-evolution ML
framework.  Consumes feature DataFrames produced by ``02_extract_features.py``
and writes a results pickle that is later consumed by
``04_feature_importance.py``.

Workflow
--------
1. Load and concatenate ``features-{job_id}.pkl`` files from ``--data-dir``.
2. Split into features (X) and log₁₀-transformed labels (y).
3. Hold out a final test set (``--test-fraction``).
4. Run nested cross-validation on the full feature set:
   - Outer loop: ShuffleSplit with ``--n-outer`` folds evaluates
     generalisation performance.
   - Inner loop: ShuffleSplit with ``--n-inner`` folds tunes hyperparameters
     via ``RandomizedSearchCV`` (``--n-iter`` candidates per fold).
5. Select final hyperparameters per algorithm; train on full training set.
6. Evaluate final models on held-out test set; save figures and pickle.

Usage
-----
    python 03_train_models.py \\
        --data-dir data/features \\
        --output-dir outputs \\
        --labels u_ks kh_ks \\
        --n-outer 10 --n-inner 5 --n-iter 20

    # Train on individual parameters
    python 03_train_models.py --labels u kh ks

    # Skip nested CV; re-run final-model and plots from an existing pickle
    python 03_train_models.py --skip-cv --results-pkl outputs/nested-cv-results-full-abc1234.pkl
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from display_labels import TARGET_DISPLAY as _TARGET_DISPLAY
from ml_core import nested_cv, train_final_model
from pipeline_utils import _git_hash, load_features, split_features_labels


# =============================================================================
# Display helper
# =============================================================================

def _target_label(col: str) -> str:
    """Return the LaTeX display label for target column *col*."""
    return _TARGET_DISPLAY.get(col, {}).get('label', col)


# =============================================================================
# Final model evaluation with plotting
# =============================================================================

def evaluate_final_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_test:  pd.DataFrame,
    results: dict,
    output_dir: str,
    git_hash: str,
    figure_width_cm: float = 14.0,
) -> dict:
    """Train final models and produce predicted-vs-true figures.

    Calls ``train_final_model()`` from ``ml_core`` for each algorithm, then
    generates per-model and per-target comparison figures.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test   : held-out test data (never seen during CV)
    results : dict   — output of ``nested_cv()``; mutated in-place
    output_dir : str
    git_hash : str
    figure_width_cm : float

    Returns
    -------
    results : dict  (updated with ``'final_model'`` key per algorithm)
    """
    labels = y_train.columns
    fw     = figure_width_cm / 2.54

    # Per-target comparison canvases (all models on one figure per target)
    fig_per_target, ax_per_target = {}, {}
    for col in labels:
        n_models = len(results)
        ncols = min(4, n_models)
        nrows = int(np.ceil(n_models / ncols))
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=True, sharey=True,
            figsize=(fw, fw * 0.6 * nrows / 2),
        )
        fig_per_target[col] = fig
        ax_per_target[col]  = np.array(axes).flatten()

    scatter_kw = dict(alpha=0.2, lw=0, marker='o', ms=2)
    fs_ax = 7

    for model_idx, reg_name in enumerate(results):
        print(f"\nFinal model: {reg_name}")

        results[reg_name]['final_model'] = train_final_model(
            reg_name, results[reg_name],
            X_train, y_train, X_test, y_test,
        )
        fm     = results[reg_name]['final_model']
        y_pred = fm['regressor'].predict(X_test)

        # Per-model figure: predicted vs true + residuals for each target
        n_targets = y_train.shape[1]
        fig_m, ax_m = plt.subplots(
            nrows=n_targets, ncols=2,
            figsize=(9 / 2.54, 1.8 * n_targets),
        )
        if n_targets == 1:
            ax_m = ax_m[np.newaxis, :]

        for t_idx, col in enumerate(labels):
            display = _target_label(col)
            y_t  = y_test.loc[:, col].values
            y_p  = y_pred[:, t_idx] if y_pred.ndim > 1 else y_pred
            lims = [y_t.min(), y_t.max()]

            metric_str = '\n'.join([
                f'$R^2$={fm["test_set_r2"][t_idx]:.3f}',
                f'RMSE={fm["test_set_rmse"][t_idx]:.3f}',
                f'MAE={fm["test_set_mae"][t_idx]:.3f}',
            ])

            # Predicted vs true
            ax_m[t_idx, 0].plot(y_p, y_t, **scatter_kw)
            ax_m[t_idx, 0].plot(lims, lims, ls=':', c='k', label='1:1')
            ax_m[t_idx, 0].set_xlabel(f'predicted log {display}', fontsize=fs_ax)
            ax_m[t_idx, 0].set_ylabel(f'true log {display}', fontsize=fs_ax)
            ax_m[t_idx, 0].tick_params(labelsize=fs_ax)
            ax_m[t_idx, 0].text(0.05, 0.95, reg_name, fontsize=9, va='top',
                                 transform=ax_m[t_idx, 0].transAxes)
            ax_m[t_idx, 0].text(0.95, 0.05, metric_str, fontsize=fs_ax,
                                 va='bottom', ha='right',
                                 transform=ax_m[t_idx, 0].transAxes)

            # Residuals
            ax_m[t_idx, 1].plot(y_p, y_t - y_p, **scatter_kw)
            ax_m[t_idx, 1].axhline(0, ls=':', c='k')
            ax_m[t_idx, 1].set_xlabel(f'predicted log {display}', fontsize=fs_ax)
            ax_m[t_idx, 1].set_ylabel(f'residual log {display}', fontsize=fs_ax)
            ax_m[t_idx, 1].tick_params(labelsize=fs_ax)

            # Add to per-target comparison canvas
            metric_str2 = '\n'.join([
                f'$R^2$={fm["test_set_r2"][t_idx]:.3f}',
                f'RMSE={fm["test_set_rmse"][t_idx]:.3f}',
            ])
            cur_ax = ax_per_target[col][model_idx]
            cur_ax.plot(y_p, y_t, **scatter_kw)
            cur_ax.plot(lims, lims, ls=':', c='k')
            cur_ax.tick_params(labelsize=fs_ax)
            cur_ax.text(0.02, 0.98, reg_name, fontsize=9, va='top',
                        transform=cur_ax.transAxes)
            cur_ax.text(0.95, 0.05, metric_str2, fontsize=fs_ax,
                        va='bottom', ha='right', transform=cur_ax.transAxes)

        fig_m.tight_layout()
        stem = os.path.join(output_dir, f'pred_vs_true-{reg_name}-{git_hash}')
        fig_m.savefig(stem + '.png', dpi=300)
        fig_m.savefig(stem + '.svg')
        plt.close(fig_m)

    for col in labels:
        fig = fig_per_target[col]
        fig.tight_layout()
        stem = os.path.join(output_dir, f'pred_vs_true-all_models-{col}-{git_hash}')
        fig.savefig(stem + '.png', dpi=300)
        fig.savefig(stem + '.svg')
        plt.close(fig)

    return results


# =============================================================================
# Plotting: nested CV results
# =============================================================================

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
    results : dict
    y : pd.DataFrame — training targets (column names used for annotations)
    output_dir : str
    git_hash : str
    figure_width_cm : float
    box : bool — True → box plots, False → scatter + mean marker
    """
    fw   = figure_width_cm / 2.54
    col0 = list(results.keys())
    col1 = ['r2', 'rmse', 'mae']
    col1_labels = [r'$R^2$', r'$RMSE$', r'$MAE$']
    col2 = list(range(len(y.columns))) + ['mean']

    idx = pd.MultiIndex.from_product([col0, col1, col2])
    rdf = pd.DataFrame(columns=idx)
    for c0 in col0:
        for c1 in col1:
            for c2 in col2:
                if c2 == 'mean':
                    rdf[(c0, c1, c2)] = results[c0][f'test_{c1}']
                else:
                    rdf[(c0, c1, c2)] = np.array(
                        results[c0][f'per_target_{c1}']
                    ).T[c2]

    col0_sorted = list(
        np.array(col0)[np.argsort([results[c]['mean_test_r2'] for c in col0])]
    )

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

    for c1, c1_label in zip(col1, col1_labels):
        fig, axes = plt.subplots(
            ncols=len(col2), sharey=True, figsize=(fw, fw * 0.3)
        )
        for i, c2 in enumerate(col2):
            ax   = axes[i]
            data = [rdf[(c0, c1, c2)] for c0 in col0_sorted]

            if box:
                ax.boxplot(data, tick_labels=col0_sorted, **box_kw)
            else:
                has_leg = False
                for j, c0 in enumerate(col0_sorted):
                    kw = dict(marker='.', mfc='0.5', mec='none', alpha=0.5, lw=0)
                    lbl_s = 'cv scores' if not has_leg else '_'
                    lbl_m = 'mean'      if not has_leg else '_'
                    ax.plot(np.full(len(rdf[(c0, c1, c2)]), j),
                            rdf[(c0, c1, c2)], label=lbl_s, **kw)
                    ax.plot(j, rdf[(c0, c1, c2)].mean(),
                            marker='_', mec='k', mfc='none', lw=0, label=lbl_m)
                    has_leg = True
                ax.set_xticks(range(len(col0_sorted)))
                ax.set_xticklabels(col0_sorted)

            ax.tick_params(axis='both', labelsize=7)
            ax.tick_params(axis='x', rotation=60)

            text = ('mean' if c2 == 'mean'
                    else f'log {_target_label(y.columns[c2])}')
            ha = 'left' if c1 == 'r2' else 'right'
            tx = 0.05   if c1 == 'r2' else 0.95
            ax.text(tx, 0.98, text, va='top', ha=ha,
                    transform=ax.transAxes, fontsize=10)

            if not box and c2 == 'mean':
                ax.legend(fontsize=6, loc='lower right')

        axes[0].set_ylabel(c1_label, rotation=90, ha='center', fontsize=8)
        fig.tight_layout()

        suffix = '-box' if box else ''
        stem = os.path.join(
            output_dir, f'nested-cv-performance-full-{c1}{suffix}-{git_hash}'
        )
        fig.savefig(stem + '.png', dpi=300)
        fig.savefig(stem + '.svg')
        plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Nested CV training on the full feature set.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-dir',     default='data/features',
                   help='Directory containing features-{job_id}.pkl files.')
    p.add_argument('--output-dir',   default='outputs',
                   help='Directory for results pickle and figures.')
    p.add_argument('--job-ids',      nargs='+', default=None,
                   help='Restrict to specific job IDs.')
    p.add_argument('--labels',       nargs='+', default=['u_ks', 'kh_ks'],
                   help='Target columns (log₁₀-transformed at training time).')
    p.add_argument('--test-fraction', type=float, default=0.1,
                   help='Fraction of data held out as the final test set.')
    p.add_argument('--n-outer',      type=int, default=10)
    p.add_argument('--n-inner',      type=int, default=5)
    p.add_argument('--n-iter',       type=int, default=20,
                   help='RandomizedSearchCV candidates per inner fold.')
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--skip-cv',      action='store_true',
                   help='Skip nested CV; load existing pickle for plotting.')
    p.add_argument('--results-pkl',  default=None,
                   help='Path to existing results pickle (used with --skip-cv).')
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    git_hash = _git_hash()

    # 1. Load data
    df = load_features(args.data_dir, job_ids=args.job_ids)
    X, y = split_features_labels(df, args.labels)
    print(f"\nFeatures : {X.shape[1]} columns, {X.shape[0]:,} samples")
    print(f"Targets  : {list(y.columns)}")

    # 2. Train / test split (deterministic tail split)
    n_test  = max(1, int(len(df) * args.test_fraction))
    n_train = len(df) - n_test
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    print(f"Train : {len(X_train):,}   Test : {len(X_test):,}")

    # 3. Nested CV
    pkl_path = args.results_pkl or os.path.join(
        args.output_dir, f'nested-cv-results-full-{git_hash}.pkl'
    )

    if args.skip_cv and os.path.exists(pkl_path):
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

    # 4. Plot CV results
    plt.close('all')
    plot_nested_cv_results(results, y_train, args.output_dir, git_hash, box=False)
    plot_nested_cv_results(results, y_train, args.output_dir, git_hash, box=True)

    # 5. Train final models; evaluate on held-out test set
    plt.close('all')
    results = evaluate_final_models(
        X_train, y_train, X_test, y_test,
        results, args.output_dir, git_hash,
    )
    with open(pkl_path, 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nFinal results saved → {pkl_path}")

    # Summary table
    print("\n── Final model test-set performance ──")
    for reg_name, reg_res in results.items():
        fm = reg_res.get('final_model', {})
        if fm:
            r2_mean = fm['test_set_r2'][-1]
            per = '  '.join(
                f'{c}:{v:.3f}' for c, v in zip(y.columns, fm['test_set_r2'][:-1])
            )
            print(f"  {reg_name:4s}  mean R²={r2_mean:.3f}   [{per}]")


if __name__ == '__main__':
    main()
