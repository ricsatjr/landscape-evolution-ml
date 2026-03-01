"""
04_feature_importance.py
------------------------
Feature importance analysis for the landscape-evolution ML framework.
Identifies which topographic features are most informative for estimating
dimensionless landscape evolution parameter ratios (U/Ks, Kh/Ks).

This script is the computational counterpart to the feature importance
sections of the associated JGR:ES article (Figs. 10–12).

Workflow
--------
1. Load ``features-{job_id}.pkl`` files (same data as ``03_train_models.py``).
2. Identify multicollinear features via hierarchical clustering on Spearman
   correlation distances (``get_feature_clusters``).
3. Select one representative feature per cluster, either:
   - **Domain-driven** (default): manually specified list that prioritises
     interpretable, easy-to-compute features.
   - **Data-driven** (``--feature-selection random``): random representative
     drawn from within each cluster, fully reproducible via ``--random-state``.
4. Run nested cross-validation on the reduced feature set, saving
   ``nested-cv-results-reduced-{hash}.pkl``.
5. Compare full-feature (from ``03``) vs reduced-feature generalisation
   performance with Welch t-tests (``--full-results-pkl`` required).
6. Compute permutation feature importance on the reduced final models.
7. Save ranked importance table (CSV) and all figures.

Usage
-----
    # Full run (clustering → CV → comparison → importance)
    python 04_feature_importance.py \\
        --data-dir data/features \\
        --output-dir outputs \\
        --full-results-pkl outputs/nested-cv-results-full-abc1234.pkl

    # Data-driven feature selection
    python 04_feature_importance.py --feature-selection random

    # Skip CV; reload existing reduced results for importance analysis only
    python 04_feature_importance.py \\
        --skip-cv \\
        --results-pkl outputs/nested-cv-results-reduced-abc1234.pkl \\
        --full-results-pkl outputs/nested-cv-results-full-abc1234.pkl
"""

import argparse
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from display_labels import FEATURE_DISPLAY, TARGET_DISPLAY
from ml_core import nested_cv, train_final_model
from pipeline_utils import _git_hash, load_features, split_features_labels


# =============================================================================
# Display helpers
# =============================================================================

def _feature_label(col: str) -> str:
    return FEATURE_DISPLAY.get(col, {}).get('label', col)


def _target_label(col: str) -> str:
    return TARGET_DISPLAY.get(col, {}).get('label', col)


# =============================================================================
# Domain-driven reduced feature set
# =============================================================================

# Features chosen to represent each correlation cluster while maximising
# physical interpretability and ease of computation.
# Must be a subset of the 39 features written by 02_extract_features.py.
# Update this list if the clustering results change substantially.
DOMAIN_REDUCED_FEATURES = [
    'n0',          # zero-order channel count  — primary finding
    'Z_mean',      # mean elevation            — scaling
    'Z_cv',        # elevation variability     — relief organisation
    'Z_skew',      # elevation skewness        — hypsometry proxy
    'grd_mean',    # mean gradient             — hillslope steepness
    'htcrv_med',   # median hilltop curvature  — erosion-uplift balance
    'crv_mean',    # mean curvature            — overall curvature
    'crv_kurt',    # curvature kurtosis        — curvature heterogeneity
    'htcrv_min',   # minimum hilltop curvature — extreme curvature signal
    'htcrv_max',   # maximum hilltop curvature — extreme curvature signal
    'hyp_int',     # hypsometric integral      — landscape maturity
    'Rb',          # mean bifurcation ratio    — network structure
    'Rb0',         # zero-order bifurcation    — primary finding
    'Rl0',         # zero-order length ratio   — network geometry
]


# =============================================================================
# Feature clustering
# =============================================================================

def get_feature_clusters(
    X: pd.DataFrame,
    dist_thresh: float = 0.25,
    output_dir: str = 'outputs',
    git_hash: str = 'latest',
    feature_selection: str = 'domain',
    random_state: int = 42,
) -> tuple:
    """Identify multicollinear feature clusters via hierarchical clustering.

    Uses Spearman rank correlation converted to a distance matrix, then
    Ward linkage.  Produces a dendrogram figure with selected features marked.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (training set only).
    dist_thresh : float
        Distance threshold for cutting the dendrogram into clusters.
        Features within ``dist_thresh`` of each other are grouped.
    output_dir : str
    git_hash : str
    feature_selection : str
        ``'domain'`` — use ``DOMAIN_REDUCED_FEATURES`` (default).
        ``'random'`` — draw one random representative per cluster.
    random_state : int
        Seed for random representative selection (only used when
        ``feature_selection='random'``).

    Returns
    -------
    reduced_features : list of str
    cluster_id_to_feature_ids : dict
        Maps cluster ID → list of feature column indices within that cluster.
    """
    # Spearman correlation → distance matrix
    corr = spearmanr(X).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr)
    dist_linkage    = hierarchy.ward(squareform(distance_matrix))

    # Dendrogram
    fig, ax = plt.subplots(figsize=(14 / 2.54, 8))
    dendro = hierarchy.dendrogram(
        dist_linkage,
        labels=X.columns.to_list(),
        ax=ax,
        leaf_rotation=360,
        orientation='right',
        color_threshold=dist_thresh,
    )
    ax.axvline(x=dist_thresh, ls='--', c='k', label='threshold distance')
    ax.set_xlabel('distance', fontsize=8)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(-0.01,)

    # Cluster assignments
    cluster_ids = hierarchy.fcluster(dist_linkage, dist_thresh, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cid].append(idx)

    # Select representative features
    if feature_selection == 'random':
        rng = np.random.default_rng(random_state)
        selected_idx = [
            int(rng.choice(v)) for v in cluster_id_to_feature_ids.values()
        ]
        reduced_features = [X.columns[s] for s in selected_idx]
        print(f"Data-driven selection ({len(reduced_features)} features): {reduced_features}")
    else:
        reduced_features = DOMAIN_REDUCED_FEATURES
        missing = [f for f in reduced_features if f not in X.columns]
        if missing:
            raise ValueError(
                f"DOMAIN_REDUCED_FEATURES contains columns not in X: {missing}"
            )
        print(f"Domain-driven selection ({len(reduced_features)} features): {reduced_features}")

    # Annotate dendrogram with selected features
    tick_labels = [item.get_text() for item in ax.get_yticklabels()]
    annotated   = [f'*{l}' if l in reduced_features else l for l in tick_labels]
    ax.set_yticklabels(annotated)
    ax.tick_params(axis='both', labelsize=8)

    fig.tight_layout()
    stem = os.path.join(output_dir, f'feature-clustering-{git_hash}')
    fig.savefig(stem + '.png', dpi=300)
    fig.savefig(stem + '.svg')
    plt.close(fig)

    return reduced_features, cluster_id_to_feature_ids


# =============================================================================
# Full vs reduced performance comparison
# =============================================================================

def compare_full_vs_reduced(
    full_results: dict,
    red_results:  dict,
    y_labels:     list,
    output_dir:   str,
    git_hash:     str,
    figure_width_cm: float = 12.0,
) -> None:
    """Compare generalisation performance of full- and reduced-feature models.

    Runs Welch t-tests on outer-fold R² and RMSE distributions and plots
    box-plot comparisons for each algorithm and metric.  Prints a LaTeX-ready
    summary table to stdout.

    Parameters
    ----------
    full_results : dict  — nested-CV results from ``03_train_models.py``
    red_results  : dict  — nested-CV results from this script (reduced features)
    y_labels : list of str — target column names (for figure titles)
    output_dir : str
    git_hash : str
    figure_width_cm : float
    """
    models   = [k for k in full_results.keys() if k != '_meta']
    metrics  = ['r2', 'rmse']
    metric_labels = [r'$R^2$', r'$RMSE$']
    fw = figure_width_cm / 2.54

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

    ncols = max(1, len(models) // 2)

    for metric, metric_label in zip(metrics, metric_labels):
        fig, axes = plt.subplots(
            nrows=2, ncols=ncols, sharey=True,
            figsize=(fw, fw * 0.5),
        )
        axes = axes.flatten()

        print(f"\n── Full vs Reduced: {metric} ──")
        for i, model in enumerate(models):
            full_scores = full_results[model][f'test_{metric}']
            red_scores  = red_results[model][f'test_{metric}']
            ttest = st.ttest_ind(full_scores, red_scores, equal_var=False)
            decision = 'H0' if ttest.pvalue > 0.05 else 'H1'
            print(
                f"  {model} & "
                f"{np.mean(full_scores):.3f} ({np.std(full_scores):.3f}) & "
                f"{np.mean(red_scores):.3f} ({np.std(red_scores):.3f}) & "
                f"{ttest.pvalue:.3f} & {decision} \\\\"
            )

            df_plot = pd.DataFrame({'full': full_scores, 'reduced': red_scores})
            ax = axes[i]
            ax.boxplot(df_plot, **box_kw)

            # x-tick labels only on bottom row
            if i >= ncols:
                ax.set_xticks([1, 2])
                ax.set_xticklabels(['full', 'reduced'], fontsize=6)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis='both', labelsize=6)
            va = 'bottom' if metric == 'r2' else 'top'
            y_pos = 0.04 if metric == 'r2' else 0.96
            ax.text(0.98, y_pos, model, ha='right', va=va,
                    transform=ax.transAxes, fontsize=8)

            if i in [0, ncols]:
                ax.set_ylabel(metric_label, fontsize=8)

        fig.tight_layout()
        stem = os.path.join(output_dir, f'full-vs-reduced-{metric}-{git_hash}')
        fig.savefig(stem + '.png', dpi=300)
        fig.savefig(stem + '.svg')
        plt.close(fig)


# =============================================================================
# Permutation importance
# =============================================================================

def compute_permutation_importance(
    reg,
    X: pd.DataFrame,
    y: pd.DataFrame,
    ax,
    n_repeats: int = 10,
    random_state: int = 42,
    box: bool = False,
) -> tuple:
    """Compute and plot permutation importance for one fitted model.

    Parameters
    ----------
    reg : fitted Pipeline
    X : pd.DataFrame — test features (reduced set)
    y : pd.DataFrame — log₁₀ test targets
    ax : matplotlib Axes
    n_repeats : int
    random_state : int
    box : bool — True → box plots, False → scatter + mean marker

    Returns
    -------
    ax : matplotlib Axes
    importances_mean : np.ndarray — mean importance per feature
    """
    result = permutation_importance(
        reg, X, y, n_repeats=n_repeats, random_state=random_state,
        n_jobs=-1, scoring='r2',
    )
    sorted_idx = result.importances_mean.argsort()

    ax.axvline(x=0, color='0.5', linestyle='--')

    if box:
        ax.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            boxprops    ={'linewidth': 0.5},
            whiskerprops={'linewidth': 0.5, 'color': '0.5'},
            capprops    ={'linewidth': 0.5, 'color': '0.5'},
            flierprops  ={'marker': 'o', 'mew': 0.5, 'ms': 2, 'color': '0.5'},
            medianprops ={'color': '0.5'},
            meanprops   ={'marker': '^', 'mfc': 'k', 'mew': 0, 'mec': 'k',
                          'alpha': 1, 'ms': 5},
            showmeans=True,
        )
    else:
        for rank, i in enumerate(sorted_idx):
            scores = result.importances[i]
            kw = dict(marker='.', mec='none', mfc='0.5', lw=0, alpha=0.5)
            label_s = 'score' if rank == 0 else '_'
            label_m = 'mean'  if rank == 0 else '_'
            ax.plot(scores, np.full(len(scores), rank), label=label_s, **kw)
            ax.plot(scores.mean(), rank, marker='|', mec='k', mfc='none',
                    lw=0, alpha=1, label=label_m)

    ax.set_yticks(np.arange(len(X.columns)))
    ax.set_yticklabels(
        [_feature_label(X.columns[i]) for i in sorted_idx], fontsize=7
    )

    return ax, result.importances_mean


def analyze_feature_importance(
    results: dict,
    all_features: list,
    reduced_features: list,
    cluster_id_to_feature_ids: dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    output_dir: str,
    git_hash: str,
    n_repeats: int = 10,
    random_state: int = 42,
    box: bool = False,
) -> pd.DataFrame:
    """Compute permutation importance for all models and aggregate ranks.

    Parameters
    ----------
    results : dict — reduced-feature nested-CV results (with ``final_model``)
    all_features : list — full list of 39 feature names
    reduced_features : list — selected feature subset
    cluster_id_to_feature_ids : dict
    X_test : pd.DataFrame — reduced test features
    y_test : pd.DataFrame — log₁₀ test targets
    output_dir : str
    git_hash : str
    n_repeats : int
    random_state : int
    box : bool

    Returns
    -------
    pd.DataFrame
        Importance ranks per model and median rank, sorted by median rank.
        Includes a ``'related features'`` column listing cluster companions.
    """
    models = [k for k in results.keys() if k != '_meta']
    fig, axes = plt.subplots(
        nrows=2, ncols=max(1, len(models) // 2),
        sharex=False, figsize=(19 / 2.54, 4),
    )
    axes = axes.flatten()

    feat_imp_df = pd.DataFrame(index=reduced_features)

    for i, model in enumerate(models):
        print(f"\nPermutation importance: {model}")
        reg = results[model]['final_model']['regressor']

        axes[i], imp_mean = compute_permutation_importance(
            reg, X_test, y_test, axes[i],
            n_repeats=n_repeats, random_state=random_state, box=box,
        )
        feat_imp_df[model] = imp_mean
        feat_imp_df[model] = feat_imp_df[model].rank(ascending=False)

        axes[i].text(0.95, 0.05, model, va='bottom', ha='right',
                     transform=axes[i].transAxes, fontsize=9)
        axes[i].tick_params(labelsize=7)
        xlim = axes[i].get_xlim()
        axes[i].set_xlim(-0.1 * (xlim[1] - xlim[0]),)

        if model == 'mlp':
            axes[i].legend(fontsize=7, loc='center right')

    fig.supxlabel(r'feature importance ($R^2 - R^2_\mathrm{perm}$)', fontsize=10)
    fig.tight_layout()
    stem = os.path.join(output_dir, f'feature-importance-{git_hash}')
    fig.savefig(stem + '.png', dpi=300)
    fig.savefig(stem + '.svg')
    plt.close(fig)

    # Aggregate: median rank and std of ranks across models
    rank_cols = [c for c in feat_imp_df.columns
                 if c not in ('median_rank', 'related features')]
    feat_imp_df['median_rank'] = feat_imp_df[rank_cols].median(axis=1)
    feat_imp_df['rank_std']    = feat_imp_df[rank_cols].std(axis=1)
    feat_imp_df = np.round(
        feat_imp_df.sort_values(['median_rank', 'rank_std']), 1
    )

    # Annotate with cluster companions
    for rfeat in feat_imp_df.index:
        for key, val in cluster_id_to_feature_ids.items():
            cluster_names = [all_features[v] for v in val]
            if rfeat in cluster_names:
                companions = [n for n in cluster_names if n != rfeat]
                feat_imp_df.loc[rfeat, 'related features'] = ', '.join(companions)

    feat_imp_df.to_csv(
        os.path.join(output_dir, f'feature-importance-{git_hash}.csv')
    )
    return feat_imp_df


# =============================================================================
# Top-feature scatter plots  (publication figures — user-controlled)
# =============================================================================

def plot_top_features_vs_targets(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    feat_imp_df: pd.DataFrame,
    top_n: int = 5,
    output_dir: str = 'outputs',
    git_hash: str = 'latest',
    figure_width_cm: float = 19.0,
):
    """Scatter plots of the top-N features against each target.

    Parameters
    ----------
    X_train : pd.DataFrame — reduced training features
    y_train : pd.DataFrame — log₁₀ training targets
    feat_imp_df : pd.DataFrame — output of ``analyze_feature_importance``
    top_n : int
    output_dir : str
    git_hash : str
    figure_width_cm : float
    """
    labels   = y_train.columns
    top_feat = feat_imp_df.index[:top_n]
    fw       = figure_width_cm / 2.54
    fig_kw   = dict(alpha=0.2, lw=0, marker='.', mew=0)

    fig, ax = plt.subplots(
        nrows=len(labels), ncols=top_n,
        figsize=(fw, fw * 0.4),
    )

    for f, feat in enumerate(top_feat):
        for t, label in enumerate(labels):
            ax[t, f].plot(X_train[feat], y_train[label], **fig_kw)

            ax[t, f].tick_params(labelsize=7)
            if f != 0:
                ax[t, f].set_yticklabels([])
            if f == 0:
                ax[t, f].set_ylabel(
                    f'log {_target_label(label)}', fontsize=8
                )
            if t == len(labels) - 1:
                ax[t, f].set_xlabel(feat, fontsize=8)
            else:
                ax[t, f].set_xticklabels([])

    fig.tight_layout()
    stem = os.path.join(output_dir, f'top-features-vs-targets-{git_hash}')
    fig.savefig(stem + '.png', dpi=300)
    fig.savefig(stem + '.svg')
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Feature importance analysis for the landscape-evolution ML framework.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-dir',      default='data/features',
                   help='Directory containing features-{job_id}.pkl files.')
    p.add_argument('--output-dir',    default='outputs')
    p.add_argument('--job-ids',       nargs='+', default=None)
    p.add_argument('--labels',        nargs='+', default=['u_ks', 'kh_ks'],
                   help='Target columns (log₁₀-transformed at training time).')
    p.add_argument('--test-fraction', type=float, default=0.1)
    p.add_argument('--dist-thresh',   type=float, default=0.25,
                   help='Spearman distance threshold for feature clustering.')
    p.add_argument('--feature-selection', choices=['domain', 'random'],
                   default='domain',
                   help=(
                       '"domain" uses the hand-picked DOMAIN_REDUCED_FEATURES list; '
                       '"random" draws one feature per cluster at random.'
                   ))
    p.add_argument('--n-outer',       type=int, default=10)
    p.add_argument('--n-inner',       type=int, default=5)
    p.add_argument('--n-iter',        type=int, default=20)
    p.add_argument('--n-repeats',     type=int, default=10,
                   help='Permutation repeats per feature.')
    p.add_argument('--random-state',  type=int, default=42)
    p.add_argument('--top-n',         type=int, default=5,
                   help='Number of top features to plot against targets.')
    p.add_argument('--skip-cv',       action='store_true',
                   help='Skip nested CV; load existing reduced pickle.')
    p.add_argument('--results-pkl',   default=None,
                   help='Path to existing REDUCED results pickle (--skip-cv).')
    p.add_argument('--full-results-pkl', default=None,
                   help='Path to FULL-feature results pickle from 03 '
                        '(required for full-vs-reduced comparison).')
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    git_hash = _git_hash()

    # 1. Load full-feature results pickle if provided (paired mode)
    full_results = None
    if args.full_results_pkl:
        if not os.path.exists(args.full_results_pkl):
            raise FileNotFoundError(
                f"--full-results-pkl not found: {args.full_results_pkl}"
            )
        with open(args.full_results_pkl, 'rb') as fh:
            full_results = pickle.load(fh)
        meta = full_results.get('_meta', {})
        print(f"\nPaired mode: inheriting parameters from {args.full_results_pkl}")
        print(f"  labels       : {meta.get('label_names')}")
        print(f"  job_ids      : {len(meta.get('job_ids', []))} jobs")
        print(f"  random_state : {meta.get('random_state')}")
        print(f"  train/test   : {len(meta.get('train_idx', []))}"
              f" / {len(meta.get('test_idx', []))}")
    else:
        meta = {}
        print("\nStandalone mode: using CLI parameters.")

    # 2. Load data
    # In paired mode, load all available feature files from --data-dir
    # without filtering by job_ids — the user is responsible for pointing
    # --data-dir to the same data used in 03. Job IDs are verified below.
    df = load_features(args.data_dir, job_ids=args.job_ids)

    # In paired mode, verify loaded job_ids match _meta
    if meta.get('job_ids'):
        loaded_job_ids   = sorted(df['job_id'].unique().tolist())
        expected_job_ids = meta['job_ids']
        if loaded_job_ids != expected_job_ids:
            raise ValueError(
                f"Loaded data does not match full-feature results.\n"
                f"  Expected {len(expected_job_ids)} jobs from _meta\n"
                f"  Loaded   {len(loaded_job_ids)} jobs from {args.data_dir}\n"
                f"  Check that --data-dir points to the same feature files "
                f"used in 03."
            )

    # In paired mode, labels are inherited from _meta
    label_names = meta.get('label_names', args.labels)
    X, y = split_features_labels(df, label_names)
    all_features = list(X.columns)
    print(f"\nFeatures : {len(all_features)} columns, {X.shape[0]:,} samples")
    print(f"Targets  : {list(y.columns)}")

    # 3. Train / test split
    # In paired mode, inherit exact split indices from _meta
    if meta.get('train_idx') and meta.get('test_idx'):
        train_idx = meta['train_idx']
        test_idx  = meta['test_idx']
        X_train = X.loc[train_idx]
        X_test  = X.loc[test_idx]
        y_train = y.loc[train_idx]
        y_test  = y.loc[test_idx]
    else:
        n_test  = max(1, int(len(df) * args.test_fraction))
        n_train = len(df) - n_test
        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    print(f"Train : {len(X_train):,}   Test : {len(X_test):,}")

    # In paired mode, random_state is inherited from _meta
    random_state = meta.get('random_state', args.random_state)

    # 4. Feature clustering
    reduced_features, cluster_id_to_feature_ids = get_feature_clusters(
        X_train,
        dist_thresh=args.dist_thresh,
        output_dir=args.output_dir,
        git_hash=git_hash,
        feature_selection=args.feature_selection,
        random_state=random_state,
    )

    X_train_red = X_train[reduced_features]
    X_test_red  = X_test[reduced_features]

    # 5. Nested CV on reduced features
    pkl_path = args.results_pkl or os.path.join(
        args.output_dir, f'nested-cv-results-reduced-{git_hash}.pkl'
    )

    if args.skip_cv and os.path.exists(pkl_path):
        print(f"\nLoading existing reduced results from {pkl_path}")
        with open(pkl_path, 'rb') as fh:
            results = pickle.load(fh)
    else:
        results = nested_cv(
            X_train_red, y_train,
            n_outer_splits=args.n_outer,
            n_inner_splits=args.n_inner,
            random_state=random_state,
            n_iter=args.n_iter,
        )
        results['_meta'] = {
            'feature_names':     list(X_train_red.columns),
            'label_names':       list(y_train.columns),
            'job_ids':           sorted(df['job_id'].unique().tolist()),
            'train_idx':         list(X_train_red.index),
            'test_idx':          list(X_test_red.index),
            'random_state':      random_state,
            'git_hash':          git_hash,
            'feature_selection': args.feature_selection,
            'paired_with':       args.full_results_pkl,
        }
        with open(pkl_path, 'wb') as fh:
            pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nReduced nested-CV results saved → {pkl_path}")

    # 6. Train final reduced models
    print("\n── Training final reduced-feature models ──")
    for reg_name in (k for k in results if k != '_meta'):
        results[reg_name]['final_model'] = train_final_model(
            reg_name, results[reg_name],
            X_train_red, y_train, X_test_red, y_test,
        )

    with open(pkl_path, 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Updated reduced results saved → {pkl_path}")

    # 7. Full vs reduced comparison (paired mode only)
    if full_results is not None:
        plt.close('all')
        compare_full_vs_reduced(
            full_results, results, label_names,
            args.output_dir, git_hash,
        )
    else:
        print("\nStandalone mode: skipping full-vs-reduced comparison.")

    # 8. Permutation importance
    plt.close('all')
    feat_imp_df = analyze_feature_importance(
        results,
        all_features=all_features,
        reduced_features=reduced_features,
        cluster_id_to_feature_ids=cluster_id_to_feature_ids,
        X_test=X_test_red,
        y_test=y_test,
        output_dir=args.output_dir,
        git_hash=git_hash,
        n_repeats=args.n_repeats,
        random_state=random_state,
    )

    print("\n── Feature importance (median rank) ──")
    print(feat_imp_df[['median_rank', 'rank_std', 'related features']].to_string())

    # 9. Top-feature scatter plots
    plt.close('all')
    plot_top_features_vs_targets(
        X_train_red, y_train, feat_imp_df,
        top_n=args.top_n,
        output_dir=args.output_dir,
        git_hash=git_hash,
    )


if __name__ == '__main__':
    main()
