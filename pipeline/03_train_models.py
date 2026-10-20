"""
03_train_models.py
------------------
Nested cross-validation training pipeline for the landscape-evolution ML
framework.  Consumes feature DataFrames produced by ``02_extract_features.py``
and writes a results pickle that is later consumed by
``04_feature_importance.py``.

Workflow
--------
1. Load and concatenate ``features-{job_id}.pkl`` files from ``--data-dir``,
   optionally restricted to the landscapes verified as being at steady state
   (``--retained-csv``).
2. Split into features (X) and log₁₀-transformed labels (y).
3. Hold out a final test set (``--test-fraction``), recording its membership
   by landscape identity in ``_meta['train_ids']`` / ``_meta['test_ids']``.
4. Run nested cross-validation on the full feature set:
   - Outer loop: ShuffleSplit with ``--n-outer`` folds evaluates
     generalisation performance.
   - Inner loop: ShuffleSplit with ``--n-inner`` folds tunes hyperparameters
     via ``RandomizedSearchCV`` (``--n-iter`` candidates per fold).
5. Select final hyperparameters per algorithm; train on full training set.
6. Evaluate final models on held-out test set; save figures and pickle,
   under an ``-ss`` filename tag when the run was filtered.

Usage
-----
    python pipeline/03_train_models.py \\
        --data-dir data/features \\
        --output-dir outputs \\
        --labels u_ks kh_ks \\
        --n-outer 10 --n-inner 5 --n-iter 20

    # Train on individual parameters
    python pipeline/03_train_models.py --labels u kh ks

    # Skip nested CV; re-run final-model and plots from an existing pickle
    python pipeline/03_train_models.py --skip-cv --results-pkl outputs/nested-cv-results-full-u_ks-kh_ks-abc1234.pkl --labels u_ks kh_ks

    # Train on the steady-state subset only
    python pipeline/03_train_models.py \\
        --retained-csv data/landscapes/steady-state/steady-state-ts_mean_elevs-f5-final-abc1234-retained.csv

Notes
-----
The train / test split is positional (``X.iloc[:n_train]``), so restricting
the ensemble with ``--retained-csv`` changes which landscapes fall in the test
set.  Results obtained with and without the filter are therefore not directly
comparable: the two differ in the train / test split and in every CV fold,
since ``ShuffleSplit`` permutes a different number of rows.  A filtered run
therefore writes its pickle and figures under an ``-ss`` filename tag, and
records the manifest it was trained under in ``_meta['retained_csv']``.

``04_feature_importance.py`` and ``03b_compare_features.py`` reload the
features and reapply the split from ``_meta['train_ids']`` /
``_meta['test_ids']``, which name landscapes by their ``job_id``,
``landscape_idx`` and ``ts_index`` and are therefore unaffected by the job
selection or glob order of that reload.  A results pickle written before
these keys existed cannot be analysed by the current ``04``; retrain to
produce one.
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — prevents tkinter threading
                       # crashes when parallel CV workers are active.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from display_labels import TARGET_DISPLAY as _TARGET_DISPLAY
from ml_core import nested_cv, train_final_model
from pipeline_utils import (_git_hash, identity_keys, landscape_ids,
                            load_features, split_features_labels)


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
    label_tag: str,
    random_state: int = 42,
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
    label_tag : str  — filename tag: the hyphen-joined target columns, with
        ``-ss`` appended when the run was restricted to steady-state
        landscapes, so that filtered and unfiltered figures of the same git
        hash do not collide
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

    for model_idx, reg_name in enumerate(k for k in results if k != '_meta'):
        print(f"\nFinal model: {reg_name}")

        results[reg_name]['final_model'] = train_final_model(
            reg_name, results[reg_name],
            X_train, y_train, X_test, y_test,
            random_state=random_state,
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
        stem = os.path.join(output_dir, f'pred_vs_true-{reg_name}-{label_tag}-{git_hash}')
        fig_m.savefig(stem + '.png', dpi=300)
        fig_m.savefig(stem + '.svg')
        plt.close(fig_m)

    for col in labels:
        fig = fig_per_target[col]
        fig.tight_layout()
        stem = os.path.join(output_dir, f'pred_vs_true-all_models-{col}-{label_tag}-{git_hash}')
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
    label_tag: str,
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
    label_tag : str  — filename tag; see ``evaluate_final_models``
    figure_width_cm : float
    box : bool — True → box plots, False → scatter + mean marker
    """
    fw   = figure_width_cm / 2.54
    col0 = [k for k in results.keys() if k != '_meta']
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
            output_dir, f'nested-cv-performance-full-{c1}{suffix}-{label_tag}-{git_hash}'
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
    p.add_argument('--features-hash', default=None,
                   help='Git hash suffix of feature pkl files to load '
                        '(e.g. abc1234). Required if multiple hashed versions '
                        'exist in --data-dir.')
    p.add_argument('--retained-csv', default=None,
                   help='Path to a *-retained.csv manifest from '
                        '01c_check_steady_state.py. Restricts training to '
                        'the landscapes verified as being at steady state. '
                        'Changes test-set membership, since the split is '
                        'positional. Output filenames gain an -ss tag so '
                        'that filtered and unfiltered runs of the same git '
                        'hash do not overwrite one another.')
    p.add_argument('--test-fraction', type=float, default=0.2,
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
    df = load_features(args.data_dir, job_ids=args.job_ids,
                       features_hash=args.features_hash,
                       retained_csv=args.retained_csv)

    X, y = split_features_labels(df, args.labels)
    print(f"\nFeatures : {X.shape[1]} columns, {X.shape[0]:,} samples")
    print(f"Targets  : {list(y.columns)}")

    # 2. Train / test split (deterministic tail split)
    #
    # The split is positional and relies on the concatenated frame carrying no
    # ordering with respect to the LE parameters or the labels: under that
    # assumption the final rows are an exchangeable sample and no seeding is
    # required. The assumption also makes --retained-csv benign: the filter
    # is a lookup on the manifest, but which landscapes the manifest omits is
    # determined by their own steady-state outcome and hence by their
    # parameters, so under the same assumption exclusion is independent of
    # row position. It then thins the train and test blocks in the same
    # proportion, leaving the cut at nearly the same place in the unfiltered
    # ordering. Both properties fail if the frame is ever sorted by a
    # parameter, or if feature files are added such that their glob order
    # groups landscapes systematically.
    n_test  = max(1, int(len(df) * args.test_fraction))
    n_train = len(df) - n_test

    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    # Record the split by landscape identity. This is the contract that
    # 04_feature_importance.py and 03b_compare_features.py reapply, via
    # pipeline_utils.select_by_ids: an identity denotes the same landscape
    # whatever else was loaded alongside it, so the split survives a changed
    # job selection, glob order or steady-state filter.
    id_keys  = identity_keys(df)
    train_ids = landscape_ids(df.iloc[:n_train], id_keys)
    test_ids  = landscape_ids(df.iloc[n_train:], id_keys)

    print(f"Train : {len(X_train):,}   Test : {len(X_test):,}")


    # 3. Nested CV
    label_tag = '-'.join(args.labels)
    # label_tag appears only in output filenames. Mark a steady-state-filtered
    # run so that it cannot overwrite the unfiltered results and figures of
    # the same git hash: the two differ in every stored quantity, since the
    # train / test split and every CV fold are drawn from a different number
    # of rows.
    if args.retained_csv:
        label_tag += '-ss'
    pkl_path  = args.results_pkl or os.path.join(
        args.output_dir, f'nested-cv-results-full-{label_tag}-{git_hash}.pkl'
    )

    if args.skip_cv and os.path.exists(pkl_path):
        print(f"\nLoading existing results from {pkl_path}")
        with open(pkl_path, 'rb') as fh:
            results = pickle.load(fh)
        # Validate that the pkl was trained on the same labels
        pkl_labels = results.get('_meta', {}).get('label_names', [])
        if pkl_labels and pkl_labels != args.labels:
            raise ValueError(
                f"Label mismatch between --results-pkl and --labels.\n"
                f"  pkl  : {pkl_labels}\n"
                f"  CLI  : {args.labels}\n"
                f"  Check that --results-pkl corresponds to the intended labels."
            )
        # ... and on the same training set. The plots below pair the stored
        # results with y_train as loaded now, so the two must describe the
        # same landscapes. Comparing the recorded identities tests that
        # directly, and so also catches a changed --retained-csv, a changed
        # --job-ids and feature files added since training.
        pkl_ids = results.get('_meta', {}).get('train_ids')
        if pkl_ids is None:
            raise ValueError(
                f"--results-pkl has no train_ids in _meta: {pkl_path}\n"
                f"  It predates identity-based split records and cannot be "
                f"paired with a freshly loaded frame. Re-run without "
                f"--skip-cv to retrain."
            )
        if [tuple(i) for i in pkl_ids] != train_ids:
            raise ValueError(
                f"Training-set mismatch between --results-pkl and the "
                f"features loaded now.\n"
                f"  pkl  : {len(pkl_ids):,} landscapes "
                f"(retained_csv={results['_meta'].get('retained_csv')})\n"
                f"  now  : {len(train_ids):,} landscapes "
                f"(retained_csv={args.retained_csv})\n"
                f"  Check --retained-csv, --job-ids, --features-hash and "
                f"whether feature files have been added to --data-dir."
            )
    else:
        results = nested_cv(
            X_train, y_train,
            n_outer_splits=args.n_outer,
            n_inner_splits=args.n_inner,
            random_state=args.random_state,
            n_iter=args.n_iter,
        )
        results['_meta'] = {
            'feature_names':  list(X_train.columns),
            'label_names':    list(y_train.columns),
            'job_ids':        sorted(df['job_id'].unique().tolist()),
            'retained_csv':   args.retained_csv,
            'test_fraction':  args.test_fraction,
            'id_keys':        id_keys,
            'train_ids':      train_ids,
            'test_ids':       test_ids,
            'n_outer_splits': args.n_outer,
            'n_inner_splits': args.n_inner,
            'n_iter':         args.n_iter,
            'random_state':   args.random_state,
            'git_hash':       git_hash,
            'features_hash':  args.features_hash,
        }
        with open(pkl_path, 'wb') as fh:
            pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nNested-CV results saved → {pkl_path}")

    # 4. Plot CV results
    plt.close('all')
    plot_nested_cv_results(results, y_train, args.output_dir, git_hash, label_tag, box=False)
    plot_nested_cv_results(results, y_train, args.output_dir, git_hash, label_tag, box=True)

    # 5. Train final models; evaluate on held-out test set
    plt.close('all')
    results = evaluate_final_models(
        X_train, y_train, X_test, y_test,
        results, args.output_dir, git_hash, label_tag,
        random_state=args.random_state,
    )
    with open(pkl_path, 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nFinal results saved → {pkl_path}")

    # Summary table
    print("\n── Final model test-set performance ──")
    for reg_name, reg_res in results.items():
        if reg_name == '_meta':
            continue
        fm = reg_res.get('final_model', {})
        if fm:
            r2_mean = fm['test_set_r2'][-1]
            per = '  '.join(
                f'{c}:{v:.3f}' for c, v in zip(y.columns, fm['test_set_r2'][:-1])
            )
            print(f"  {reg_name:4s}  mean R²={r2_mean:.3f}   [{per}]")


if __name__ == '__main__':
    main()
