"""
04_feature_importance.py
------------------------
Feature importance analysis for the landscape-evolution ML framework.
Identifies which topographic features are most informative for estimating
dimensionless landscape evolution parameter ratios (U/Ks, Kh/Ks).

This script is the computational counterpart to the feature importance
sections of the associated JGR:ES article (Figs. 10-12).

Modes (--mode)
--------------
explore
    Load features from ``--features-dir`` and analyse inter-feature
    correlations via hierarchical clustering on Spearman rank distances.
    Produces a dendrogram figure annotated with the selected representative
    features.  No models are trained; no labels are required.  Use this
    mode to inspect feature redundancy and tune ``--cluster-threshold``
    and ``--cluster-selection`` before committing to a reduced feature set.

reduced
    Full pipeline from feature clustering through to permutation importance
    on a reduced feature set.  Requires ``--models-pkl`` pointing to a
    full-feature results pkl produced by ``03_train_models.py``.  Inherits
    train/test split indices, label names, and random state from that pkl's
    ``_meta``.  Runs nested CV on the reduced feature set, trains final
    reduced-feature models, compares full vs reduced generalisation
    performance, and runs permutation importance.  Saves the reduced-feature
    results to ``--reduced-models-pkl``.

importance
    Permutation importance analysis only.  Requires ``--models-pkl`` pointing
    to a reduced-feature results pkl (i.e. the output of a previous ``reduced``
    run).  Inherits all metadata from that pkl.  No clustering or model
    training is performed.  Runs permutation importance sequentially for each
    individual target and then for all targets combined (mean R2); the
    combined run is omitted when only one target is present.

Output filename conventions
---------------------------
All output files include a ``{label_tag}`` (hyphen-joined target column names)
and ``{git_hash}`` suffix for provenance tracking.

Permutation importance outputs carry an additional target suffix:
    feature-importance-{target_col}-{label_tag}-{hash}   <- per-target
    feature-importance-all-{label_tag}-{hash}             <- combined

Usage
-----
    # Explore feature correlations only
    python 04_feature_importance.py \\
        --mode explore \\
        --features-dir data/features \\
        --output-dir outputs/explore

    # Full reduced-feature pipeline
    python 04_feature_importance.py \\
        --mode reduced \\
        --features-dir data/features \\
        --models-pkl outputs/models/nested-cv-results-full-abc1234.pkl \\
        --reduced-models-pkl outputs/models/nested-cv-results-reduced-abc1234.pkl \\
        --output-dir outputs/reduced

    # Permutation importance on existing reduced models
    python 04_feature_importance.py \\
        --mode importance \\
        --features-dir data/features \\
        --models-pkl outputs/models/nested-cv-results-reduced-abc1234.pkl \\
        --output-dir outputs/importance
"""

import argparse
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required when permutation
                       # importance runs parallel workers (n_jobs=-1); tkinter
                       # crashes if figure objects are garbage-collected from
                       # worker threads rather than the main thread.
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
    """Return the display label for a feature column, falling back to col."""
    return FEATURE_DISPLAY.get(col, {}).get('label', col)


def _target_label(col: str) -> str:
    """Return the display label for a target column, falling back to col."""
    return TARGET_DISPLAY.get(col, {}).get('label', col)


def _importance_xlabel(target_col: str | None) -> str:
    """Return the x-axis label for a permutation importance figure.

    The label string for each target is looked up from the
    ``'importance_xlabel'`` key in ``TARGET_DISPLAY``.  This key must be a
    valid LaTeX mathtext string and is authored directly in ``display_labels.py``
    to avoid interpolating raw column names (which may contain underscores)
    into LaTeX at runtime.

    Parameters
    ----------
    target_col : str or None
        A specific target column name, or None for the combined (all-target)
        run.

    Returns
    -------
    str
        LaTeX mathtext string suitable for ``fig.supxlabel``.
    """
    if target_col is None:
        return r'feature importance ($\overline{R^2} - \overline{R^2}_\mathrm{perm}$)'
    entry = TARGET_DISPLAY.get(target_col, {})
    xlabel = entry.get('importance_xlabel')
    if xlabel is not None:
        return xlabel
    # Fallback: generic label that is always valid LaTeX.
    # Add an 'importance_xlabel' key to TARGET_DISPLAY to override this.
    return r'feature importance ($R^2_\mathrm{target} - R^2_\mathrm{perm}$)'


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
    'Rl',          # length ratio              — network geometry
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
# Per-target permutation scorer
# =============================================================================

def _make_single_target_scorer(target_idx: int):
    """Return a scorer that evaluates R2 for one target of a multi-output model.

    ``permutation_importance`` requires a scorer that returns a single scalar.
    For multi-output regressors, the default ``'r2'`` scorer returns the mean
    R2 across all targets (``multioutput='uniform_average'``).  This factory
    wraps ``r2_score`` with ``multioutput='raw_values'`` and selects the
    element at ``target_idx``, allowing per-target importance measurement
    without retraining or modifying the fitted pipeline.

    Parameters
    ----------
    target_idx : int
        Index of the target column in the model's output (0-based).
        Corresponds to the position of the target in ``y.columns``.

    Returns
    -------
    callable
        A scorer compatible with ``sklearn.inspection.permutation_importance``.
        Signature: ``scorer(estimator, X, y) -> float``.
    """
    def _score(estimator, X, y):
        y_pred = estimator.predict(X)
        return r2_score(y, y_pred, multioutput='raw_values')[target_idx]
    return _score


# =============================================================================
# Feature clustering
# =============================================================================

def get_feature_clusters(
    X: pd.DataFrame,
    dist_thresh: float = 0.12,
    output_dir: str = 'outputs',
    git_hash: str = 'latest',
    label_tag: str = '',
    cluster_selection: str = 'domain',
    random_state: int = 42,
) -> tuple:
    """Identify multicollinear feature clusters via hierarchical clustering.

    Computes a Spearman rank correlation matrix, converts it to a distance
    matrix (``1 - |rho|``), applies Ward linkage, and cuts the dendrogram at
    ``dist_thresh`` to define clusters.  One representative feature per cluster
    is selected either from the domain-driven list (``DOMAIN_REDUCED_FEATURES``)
    or at random.  Produces a dendrogram figure with selected features marked
    by a leading asterisk.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.  Should contain only the training set to avoid
        data leakage into the correlation structure.
    dist_thresh : float
        Distance threshold for cutting the dendrogram into clusters.
        Features with Spearman distance <= dist_thresh are grouped into the
        same cluster.  Smaller values produce more clusters (less aggressive
        redundancy removal).  Default 0.12 was tuned on the synthetic
        landscape ensemble.
    output_dir : str
        Directory for saving the dendrogram figure.
    git_hash : str
        Git hash suffix appended to output filenames for provenance.
    label_tag : str
        Hyphen-joined target label names, appended to output filenames.
    cluster_selection : str
        ``'domain'``: use ``DOMAIN_REDUCED_FEATURES`` (default).  Features
        must be a subset of ``X.columns``; raises ``ValueError`` if any are
        missing.
        ``'random'``: draw one representative per cluster uniformly at random,
        seeded by ``random_state``.
    random_state : int
        Random seed for ``'random'`` cluster selection.  Has no effect when
        ``cluster_selection='domain'``.

    Returns
    -------
    reduced_features : list of str
        Selected representative feature names, one per cluster.
    cluster_id_to_feature_ids : dict
        Maps cluster ID (int) to a list of feature column indices (int) within
        that cluster.  Used downstream to annotate importance tables with
        cluster companions.
    """
    corr = np.array(spearmanr(X).correlation)
    # Replace NaNs from constant-valued columns (zero variance) with 0
    corr = np.where(np.isnan(corr), 0, corr)
    # Enforce exact symmetry; floating-point rounding can produce tiny asymmetries
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr)
    # Clip to [0, 1] and re-enforce symmetry before passing to squareform
    distance_matrix = np.clip(distance_matrix, 0, 1)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    dist_linkage    = hierarchy.ward(squareform(distance_matrix))

    fig, ax = plt.subplots(figsize=(14 / 2.54, 8))
    hierarchy.dendrogram(
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

    cluster_ids = hierarchy.fcluster(dist_linkage, dist_thresh, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cid].append(idx)

    if cluster_selection == 'random':
        rng = np.random.default_rng(random_state)
        selected_idx = [
            int(rng.choice(v)) for v in cluster_id_to_feature_ids.values()
        ]
        reduced_features = [X.columns[s] for s in selected_idx]
        print(f"Random cluster selection ({len(reduced_features)} features): "
              f"{reduced_features}")
    else:
        reduced_features = DOMAIN_REDUCED_FEATURES
        missing = [f for f in reduced_features if f not in X.columns]
        if missing:
            raise ValueError(
                f"DOMAIN_REDUCED_FEATURES contains columns not in X: {missing}"
            )
        print(f"Domain-driven cluster selection ({len(reduced_features)} features): "
              f"{reduced_features}")

    tick_labels = [item.get_text() for item in ax.get_yticklabels()]
    annotated   = [f'*{l}' if l in reduced_features else l for l in tick_labels]
    ax.set_yticklabels(annotated)
    ax.tick_params(axis='both', labelsize=8)

    fig.tight_layout()
    stem = os.path.join(output_dir, f'feature-clustering-{label_tag}-{git_hash}')
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
    label_tag:    str = '',
    figure_width_cm: float = 12.0,
) -> None:
    """Compare generalisation performance of full- and reduced-feature models.

    Runs Welch t-tests on outer-fold R2 and RMSE distributions and plots
    side-by-side box plots for each algorithm and metric.  Prints a
    LaTeX-ready summary table to stdout.

    Parameters
    ----------
    full_results : dict
        Nested-CV results from ``03_train_models.py`` (full feature set).
    red_results : dict
        Nested-CV results from this script (reduced feature set).
    y_labels : list of str
        Target column names, used in the figure title.
    output_dir : str
        Directory for saving figures.
    git_hash : str
        Git hash suffix for output filenames.
    label_tag : str
        Hyphen-joined target label names for output filenames.
    figure_width_cm : float
        Figure width in centimetres.
    """
    models        = [k for k in full_results.keys() if k != '_meta']
    metrics       = ['r2', 'rmse']
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
            ttest    = st.ttest_ind(full_scores, red_scores, equal_var=False)
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

            if i >= ncols:
                ax.set_xticks([1, 2])
                ax.set_xticklabels(['full', 'reduced'], fontsize=6)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis='both', labelsize=6)
            va    = 'bottom' if metric == 'r2' else 'top'
            y_pos = 0.04    if metric == 'r2' else 0.96
            ax.text(0.98, y_pos, model, ha='right', va=va,
                    transform=ax.transAxes, fontsize=8)

            if i in [0, ncols]:
                ax.set_ylabel(metric_label, fontsize=8)

        fig.tight_layout()
        stem = os.path.join(
            output_dir, f'full-vs-reduced-{metric}-{label_tag}-{git_hash}'
        )
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
    target_col: str | None = None,
) -> tuple:
    """Compute and plot permutation importance for one fitted model.

    For each feature, the feature column is randomly shuffled ``n_repeats``
    times and the model is re-scored after each shuffle.  The importance of a
    feature is the mean drop in the scorer's value across repeats.  A positive
    value indicates the feature contributed to predictive performance; a value
    near zero or negative indicates the feature is uninformative or redundant.
    The model is never retrained — only prediction and scoring occur.

    Parameters
    ----------
    reg : fitted Pipeline
        A fitted scikit-learn Pipeline (StandardScaler + regressor) as stored
        in ``results[model]['final_model']['regressor']``.
    X : pd.DataFrame
        Reduced test feature matrix.
    y : pd.DataFrame
        log10-transformed test targets.  Always the full multi-column
        DataFrame regardless of ``target_col`` — the model requires all
        target columns for prediction.
    ax : matplotlib Axes
        Axes object to plot importance scores into.
    n_repeats : int
        Number of permutation repeats per feature.  More repeats reduce
        variance in the importance estimates at the cost of computation time.
    random_state : int
        Random seed for permutation shuffling.  Controls reproducibility
        of the importance estimates.
    box : bool
        If True, plot repeat scores as box plots.
        If False (default), plot individual repeat scores as dots with the
        mean marked by a vertical bar.
    target_col : str or None
        If None, scoring uses mean R2 across all targets (``'r2'`` scorer),
        matching the multi-output CV scoring used in nested CV.
        If a column name (e.g. ``'log_u_ks'``), scoring uses R2 for that
        target only, via ``_make_single_target_scorer``.

    Returns
    -------
    ax : matplotlib Axes
        The axes with importance scores plotted.
    importances_mean : np.ndarray of shape (n_features,)
        Mean importance per feature across permutation repeats, ordered to
        match ``X.columns``.
    """
    if target_col is not None:
        target_idx = list(y.columns).index(target_col)
        scoring = _make_single_target_scorer(target_idx)
    else:
        scoring = 'r2'

    result = permutation_importance(
        reg, X, y, n_repeats=n_repeats, random_state=random_state,
        n_jobs=-1, scoring=scoring,
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
            scores  = result.importances[i]
            kw      = dict(marker='.', mec='none', mfc='0.5', lw=0, alpha=0.5)
            label_s = 'score' if rank == 0 else '_'
            label_m = 'mean'  if rank == 0 else '_'
            ax.plot(scores, np.full(len(scores), rank), label=label_s, **kw)
            ax.plot(scores.mean(), rank, marker='|', mec='k', mfc='none',
                    lw=0, alpha=1, label=label_m)

    ax.set_yticks(np.arange(len(X.columns)))
    ax.set_yticklabels([X.columns[i] for i in sorted_idx], fontsize=7)

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
    label_tag: str = '',
    n_repeats: int = 10,
    random_state: int = 42,
    box: bool = False,
    target_col: str | None = None,
) -> pd.DataFrame:
    """Compute permutation importance for all models and aggregate ranks.

    Calls ``compute_permutation_importance`` for each algorithm in ``results``,
    collects per-model importance scores, converts them to ranks (1 = most
    important), and aggregates across models via median rank and rank standard
    deviation.  Annotates each feature with its cluster companions from the
    full feature set.

    Parameters
    ----------
    results : dict
        Reduced-feature nested-CV results with ``final_model`` entries, as
        produced by the ``reduced`` mode of this script.
    all_features : list of str
        Full list of feature names (all 39), used to look up cluster companions.
    reduced_features : list of str
        The selected representative feature subset.
    cluster_id_to_feature_ids : dict
        Maps cluster ID to list of feature column indices, from
        ``get_feature_clusters``.
    X_test : pd.DataFrame
        Reduced test feature matrix (columns = ``reduced_features``).
    y_test : pd.DataFrame
        log10-transformed test targets.  Always the full multi-column
        DataFrame — individual targets are selected via ``target_col``.
    output_dir : str
        Directory for saving figures and CSV.
    git_hash : str
        Git hash suffix for output filenames.
    label_tag : str
        Hyphen-joined target label names for output filenames.
    n_repeats : int
        Number of permutation repeats per feature.
    random_state : int
        Random seed for permutation shuffling.
    box : bool
        Passed to ``compute_permutation_importance``.
    target_col : str or None
        Passed to ``compute_permutation_importance``.
        None -> mean R2 across all targets (combined / 'all' run).
        A column name -> single-target R2.

    Returns
    -------
    pd.DataFrame
        Index: ``reduced_features``.
        Columns: one rank column per algorithm, plus ``median_rank``,
        ``rank_std``, and ``related features`` (comma-separated cluster
        companions).  Sorted by ``median_rank`` then ``rank_std``.
        Also saved to a CSV file in ``output_dir``.
    """
    target_tag = f'-{target_col}' if target_col is not None else '-all'

    models = [k for k in results.keys() if k != '_meta']
    fig, axes = plt.subplots(
        nrows=2, ncols=max(1, len(models) // 2),
        sharex=False, figsize=(19 / 2.54, 4),
    )
    axes = axes.flatten()

    feat_imp_df = pd.DataFrame(index=reduced_features)

    for i, model in enumerate(models):
        print(f"\nPermutation importance [{target_col or 'all'}]: {model}")
        reg = results[model]['final_model']['regressor']

        axes[i], imp_mean = compute_permutation_importance(
            reg, X_test, y_test, axes[i],
            n_repeats=n_repeats, random_state=random_state, box=box,
            target_col=target_col,
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

    fig.supxlabel(_importance_xlabel(target_col), fontsize=10)
    fig.tight_layout()

    stem = os.path.join(
        output_dir, f'feature-importance{target_tag}-{label_tag}-{git_hash}'
    )
    fig.savefig(stem + '.png', dpi=300)
    fig.savefig(stem + '.svg')
    plt.close(fig)

    rank_cols = [c for c in feat_imp_df.columns
                 if c not in ('median_rank', 'rank_std', 'related features')]
    feat_imp_df['median_rank'] = feat_imp_df[rank_cols].median(axis=1)
    feat_imp_df['rank_std']    = feat_imp_df[rank_cols].std(axis=1)
    feat_imp_df = np.round(
        feat_imp_df.sort_values(['median_rank', 'rank_std']), 1
    )

    for rfeat in feat_imp_df.index:
        for key, val in cluster_id_to_feature_ids.items():
            cluster_names = [all_features[v] for v in val]
            if rfeat in cluster_names:
                companions = [n for n in cluster_names if n != rfeat]
                feat_imp_df.loc[rfeat, 'related features'] = ', '.join(companions)

    feat_imp_df.to_csv(
        os.path.join(
            output_dir,
            f'feature-importance{target_tag}-{label_tag}-{git_hash}.csv'
        )
    )
    return feat_imp_df


# =============================================================================
# Top-feature scatter plots
# =============================================================================

def plot_top_features_vs_targets(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    feat_imp_df: pd.DataFrame,
    top_n: int = 5,
    output_dir: str = 'outputs',
    git_hash: str = 'latest',
    label_tag: str = '',
    target_col: str | None = None,
    figure_width_cm: float = 19.0,
):
    """Scatter plots of the top-N features against each target.

    Produces one subplot column per top feature and one subplot row per target.
    When ``target_col`` is set, only that target is plotted (one row).
    When None (combined run), all targets are plotted (one row each).

    Parameters
    ----------
    X_train : pd.DataFrame
        Reduced training features.
    y_train : pd.DataFrame
        log10 training targets — always the full multi-column DataFrame.
    feat_imp_df : pd.DataFrame
        Output of ``analyze_feature_importance``, sorted by median rank.
        The top ``top_n`` rows (index) are used as the feature selection.
    top_n : int
        Number of top-ranked features to include.
    output_dir : str
        Directory for saving figures.
    git_hash : str
        Git hash suffix for output filenames.
    label_tag : str
        Hyphen-joined target label names for output filenames.
    target_col : str or None
        If set, only this target column is plotted (one row).
        If None, all target columns are plotted (one row each).
    figure_width_cm : float
        Figure width in centimetres.
    """
    labels   = [target_col] if target_col is not None else list(y_train.columns)
    top_feat = feat_imp_df.index[:top_n]
    fw       = figure_width_cm / 2.54
    fig_kw   = dict(alpha=0.2, lw=0, marker='.', mew=0)

    fig, ax = plt.subplots(
        nrows=len(labels), ncols=top_n,
        figsize=(fw, fw * 0.4 * len(labels)),
        squeeze=False,
    )

    for f, feat in enumerate(top_feat):
        for t, label in enumerate(labels):
            ax[t, f].plot(X_train[feat], y_train[label], **fig_kw)
            ax[t, f].tick_params(labelsize=7)
            if f != 0:
                ax[t, f].set_yticklabels([])
            if f == 0:
                ax[t, f].set_ylabel(f'log {_target_label(label)}', fontsize=8)
            if t == len(labels) - 1:
                ax[t, f].set_xlabel(feat, fontsize=8)
            else:
                ax[t, f].set_xticklabels([])

    fig.tight_layout()
    target_tag = f'-{target_col}' if target_col is not None else '-all'
    stem = os.path.join(
        output_dir,
        f'top-features-vs-targets{target_tag}-{label_tag}-{git_hash}'
    )
    fig.savefig(stem + '.png', dpi=300)
    fig.savefig(stem + '.svg')
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Feature importance analysis for the landscape-evolution ML framework.\n\n'
            'Three modes are available via --mode:\n'
            '  explore    Analyse inter-feature correlations and produce a\n'
            '             dendrogram.  No models required.\n'
            '  reduced    Full pipeline: clustering, nested CV on reduced\n'
            '             features, full-vs-reduced comparison, and permutation\n'
            '             importance.\n'
            '  importance Permutation importance only on existing reduced-feature\n'
            '             models.  No training or clustering performed.\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Core I/O ──────────────────────────────────────────────────────────────
    p.add_argument(
        '--features-dir',
        required=True,
        help=(
            'Directory containing features-{job_id}.pkl files produced by '
            '02_extract_features.py.  Required in all modes: in explore mode '
            'features are loaded for correlation analysis; in reduced and '
            'importance modes features are loaded to reconstruct the train/test '
            'split for model training and importance evaluation.'
        ),
    )
    p.add_argument(
        '--output-dir',
        required=True,
        help=(
            'Directory where all output files (figures, CSVs, pkl) are saved. '
            'Created automatically if it does not exist.'
        ),
    )
    p.add_argument(
        '--models-pkl',
        default=None,
        help=(
            'Path to an existing results pkl file.  '
            'In reduced mode: must point to the FULL-feature results pkl '
            'produced by 03_train_models.py.  Train/test split indices, label '
            'names, and random state are inherited from its _meta.  '
            'In importance mode: must point to the REDUCED-feature results pkl '
            'produced by a previous reduced run of this script.  '
            'Not used in explore mode.  '
            'Required for reduced and importance modes.'
        ),
    )
    p.add_argument(
        '--reduced-models-pkl',
        default=None,
        help=(
            'Output path for the reduced-feature results pkl produced in '
            'reduced mode.  The pkl stores nested-CV results, final fitted '
            'models, and _meta (label names, split indices, git hash, CV '
            'parameters, cluster selection method).  '
            'Required when --mode reduced is set.'
        ),
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument(
        '--mode',
        choices=['explore', 'reduced', 'importance'],
        default='importance',
        help=(
            'Analysis mode (default: importance).  '
            'explore: correlation analysis and dendrogram only, no models needed.  '
            'reduced: full pipeline from clustering through to permutation '
            'importance, requires --models-pkl (full-feature) and '
            '--reduced-models-pkl (output path).  '
            'importance: permutation importance only on existing reduced-feature '
            'models, requires --models-pkl (reduced-feature).'
        ),
    )

    # ── Feature loading ────────────────────────────────────────────────────────
    p.add_argument(
        '--job-ids',
        nargs='+',
        default=None,
        help=(
            'Whitespace-separated list of job IDs to load from --features-dir. '
            'If omitted, all available feature pkl files in the directory are '
            'loaded.  Use this to restrict analysis to a specific subset of jobs.'
        ),
    )
    p.add_argument(
        '--features-hash',
        default=None,
        help=(
            'Git hash suffix of the feature pkl files to load '
            '(e.g. abc1234 matches features-{job_id}-abc1234.pkl).  '
            'Required only when --features-dir contains multiple versioned '
            'feature files with different hash suffixes.  If the directory '
            'contains a single version, this argument can be omitted.'
        ),
    )

    # ── Feature clustering (explore and reduced modes) ─────────────────────────
    p.add_argument(
        '--cluster-threshold',
        type=float,
        default=0.12,
        help=(
            'Spearman distance threshold for cutting the hierarchical clustering '
            'dendrogram into feature clusters (default: 0.12).  Features with '
            'pairwise Spearman distance <= this value are grouped into the same '
            'cluster.  Smaller values produce more clusters (less aggressive '
            'redundancy removal); larger values produce fewer clusters.  '
            'Only used in explore and reduced modes.'
        ),
    )
    p.add_argument(
        '--cluster-selection',
        choices=['domain', 'random'],
        default='domain',
        help=(
            'Method for selecting one representative feature per cluster '
            '(default: domain).  '
            'domain: use the hand-picked DOMAIN_REDUCED_FEATURES list, which '
            'prioritises physically interpretable and easily computed features.  '
            'random: draw one feature per cluster uniformly at random, seeded '
            'by --random-state.  Fully reproducible but not guided by domain '
            'knowledge.  '
            'Only used in explore and reduced modes.'
        ),
    )

    # ── Permutation importance (reduced and importance modes) ──────────────────
    p.add_argument(
        '--permutation-repeats',
        type=int,
        default=10,
        help=(
            'Number of times each feature is randomly shuffled during permutation '
            'importance computation (default: 10).  More repeats reduce variance '
            'in the importance estimates at the cost of computation time.  '
            'Only used in reduced and importance modes.'
        ),
    )
    p.add_argument(
        '--top-features',
        type=int,
        default=5,
        help=(
            'Number of top-ranked features to include in the scatter plots of '
            'features vs targets (default: 5).  Features are ranked by median '
            'importance rank across all algorithms.  '
            'Only used in reduced and importance modes.'
        ),
    )
    p.add_argument(
        '--random-state',
        type=int,
        default=42,
        help=(
            'Random seed used for (1) permutation shuffling in importance '
            'analysis and (2) random feature selection when '
            '--cluster-selection random is set (default: 42).  '
            'Does not affect train/test splits or nested CV — those inherit '
            'their random state from the input pkl _meta.'
        ),
    )

    args = p.parse_args()

    # ── Argument validation ────────────────────────────────────────────────────
    if args.models_pkl is None:
        p.error("--models-pkl is required for all modes")
    if args.mode == 'reduced' and args.reduced_models_pkl is None:
        p.error("--mode reduced requires --reduced-models-pkl")

    return args


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    git_hash = _git_hash()

    # ── Load models pkl (all modes) ───────────────────────────────────────────
    # --models-pkl is required for all modes. In explore mode it provides the
    # authoritative feature_names list; in reduced mode it provides the
    # full-feature results; in importance mode it provides the reduced-feature
    # results.
    print(f"\nLoading models pkl: {args.models_pkl}")
    with open(args.models_pkl, 'rb') as fh:
        results = pickle.load(fh)
    meta = results.get('_meta', {})

    if not meta.get('feature_names'):
        raise ValueError(
            f"Models pkl _meta has no feature_names: {args.models_pkl}\n"
            f"Cannot determine feature set."
        )

    # ── Load features ──────────────────────────────────────────────────────────
    df = load_features(
        args.features_dir,
        job_ids=args.job_ids,
        features_hash=meta.get('features_hash', args.features_hash),
    )

    # ── EXPLORE mode ───────────────────────────────────────────────────────────
    if args.mode == 'explore':
        print("\n── Mode: explore ──")
        # Use feature_names from _meta to select only feature columns,
        # avoiding contamination from label or metadata columns in df.
        feature_names = meta['feature_names']
        X = df[feature_names]
        print(f"Features : {X.shape[1]} columns, {X.shape[0]:,} samples")
        get_feature_clusters(
            X,
            dist_thresh=args.cluster_threshold,
            output_dir=args.output_dir,
            git_hash=git_hash,
            label_tag='explore',
            cluster_selection=args.cluster_selection,
            random_state=args.random_state,
        )
        print(f"\nDendrogram saved to {args.output_dir}")
        return

    if not meta.get('label_names'):
        raise ValueError(
            f"Models pkl has no label_names in _meta: {args.models_pkl}\n"
            f"Cannot determine targets."
        )
    label_names = meta['label_names']
    print(f"  label_names  : {label_names}")
    print(f"  git_hash     : {meta.get('git_hash')}")
    print(f"  random_state : {meta.get('random_state')}")
    print(f"  train/test   : {len(meta.get('train_idx', []))}"
          f" / {len(meta.get('test_idx', []))}")

    label_tag         = '-'.join(label_names)
    random_state_meta = meta.get('random_state', args.random_state)

    # ── Split features and labels ──────────────────────────────────────────────
    X, y = split_features_labels(df, label_names)
    all_features = list(X.columns)   # full feature set for cluster annotation
    print(f"\nFeatures : {len(all_features)} columns, {X.shape[0]:,} samples")
    print(f"Targets  : {list(y.columns)}")

    # Reconstruct train/test split from _meta indices
    if not (meta.get('train_idx') and meta.get('test_idx')):
        raise ValueError(
            f"Models pkl _meta is missing train_idx or test_idx: {args.models_pkl}\n"
            f"Cannot reconstruct train/test split."
        )
    train_idx = meta['train_idx']
    test_idx  = meta['test_idx']
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    print(f"Train : {len(X_train):,}   Test : {len(X_test):,}")

    # ── REDUCED mode ───────────────────────────────────────────────────────────
    if args.mode == 'reduced':
        print("\n── Mode: reduced ──")

        # 1. Feature clustering
        reduced_features, cluster_id_to_feature_ids = get_feature_clusters(
            X_train,
            dist_thresh=args.cluster_threshold,
            output_dir=args.output_dir,
            git_hash=git_hash,
            label_tag=label_tag,
            cluster_selection=args.cluster_selection,
            random_state=args.random_state,
        )
        X_train_red = X_train[reduced_features]
        X_test_red  = X_test[reduced_features]

        # 2. Nested CV on reduced features
        # n_outer, n_inner, n_iter inherited from full-feature _meta
        n_outer = meta.get('n_outer_splits', 10)
        n_inner = meta.get('n_inner_splits', 5)
        n_iter  = meta.get('n_iter', 20)
        print(f"\nNested CV: n_outer={n_outer}, n_inner={n_inner}, n_iter={n_iter}")

        red_results = nested_cv(
            X_train_red, y_train,
            n_outer_splits=n_outer,
            n_inner_splits=n_inner,
            random_state=random_state_meta,
            n_iter=n_iter,
        )
        red_results['_meta'] = {
            'feature_names':         list(X_train.columns),        # full feature set
            'reduced_feature_names': list(X_train_red.columns),    # reduced subset
            'label_names':           list(y_train.columns),
            'job_ids':               sorted(df['job_id'].unique().tolist()),
            'train_idx':             list(X_train_red.index),
            'test_idx':              list(X_test_red.index),
            'random_state':          random_state_meta,
            'n_outer_splits':        n_outer,
            'n_inner_splits':        n_inner,
            'n_iter':                n_iter,
            'git_hash':              git_hash,
            'cluster_selection':     args.cluster_selection,
            'full_models_pkl':       args.models_pkl,
        }

        # 3. Train final reduced models (skip if already present)
        models_to_train = [
            k for k in red_results if k != '_meta'
            and 'final_model' not in red_results.get(k, {})
        ]
        if models_to_train:
            print("\n── Training final reduced-feature models ──")
            for reg_name in models_to_train:
                red_results[reg_name]['final_model'] = train_final_model(
                    reg_name, red_results[reg_name],
                    X_train_red, y_train, X_test_red, y_test,
                )
        else:
            print("\nFinal reduced-feature models already present — skipping retraining.")

        with open(args.reduced_models_pkl, 'wb') as fh:
            pickle.dump(red_results, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nReduced models saved -> {args.reduced_models_pkl}")

        # 4. Full vs reduced comparison
        plt.close('all')
        compare_full_vs_reduced(
            results, red_results, label_names,
            args.output_dir, git_hash, label_tag,
        )

        results_for_importance = red_results

    # ── IMPORTANCE mode ────────────────────────────────────────────────────────
    else:
        print("\n── Mode: importance ──")

        # Verify the pkl contains final models
        model_keys    = [k for k in results if k != '_meta']
        missing_final = [k for k in model_keys
                         if 'final_model' not in results.get(k, {})]
        if missing_final:
            raise ValueError(
                f"The following models in --models-pkl have no final_model entry: "
                f"{missing_final}\n"
                f"Re-run with --mode reduced to train final reduced-feature models."
            )

        # Recover reduced feature names from _meta
        reduced_features = meta.get('reduced_feature_names')
        if not reduced_features:
            raise ValueError(
                f"Models pkl _meta has no reduced_feature_names: {args.models_pkl}\n"
                f"This pkl was likely not produced by --mode reduced.\n"
                f"Re-run with --mode reduced to generate a valid reduced-feature pkl."
            )

        # Re-run clustering on the FULL feature set to recover
        # cluster_id_to_feature_ids for companion annotation in the importance
        # table.  Clustering parameters are taken from _meta to ensure
        # consistency with the original reduced run.
        # feature_names in _meta is always the full set (consistent with 03).
        X_train_full = X_train[meta['feature_names']]
        _, cluster_id_to_feature_ids = get_feature_clusters(
            X_train_full,
            dist_thresh=args.cluster_threshold,
            output_dir=args.output_dir,
            git_hash=git_hash,
            label_tag=label_tag,
            cluster_selection=meta.get('cluster_selection', args.cluster_selection),
            random_state=args.random_state,
        )
        X_train_red = X_train[reduced_features]
        X_test_red  = X_test[reduced_features]

        results_for_importance = results

    # ── Permutation importance (reduced and importance modes) ──────────────────
    #
    # Run once per individual target, then once combined (None) when there is
    # more than one target.  The combined run uses mean R2 across all targets,
    # matching the scoring used during nested CV.  When only one target is
    # present the combined run is omitted (it would be identical to the
    # single-target run).
    #
    # target_cols sequence:
    #   [col_0, col_1, ..., None]  if len(label_names) > 1
    #   [col_0]                    if len(label_names) == 1
    target_cols = list(y_test.columns)
    if len(target_cols) > 1:
        target_cols = target_cols + [None]

    for target_col in target_cols:
        tag_label = target_col if target_col is not None else 'all'
        print(f"\n{'='*60}")
        print(f"  Permutation importance: {tag_label}")
        print(f"{'='*60}")

        plt.close('all')
        feat_imp_df = analyze_feature_importance(
            results_for_importance,
            all_features=all_features,
            reduced_features=reduced_features,
            cluster_id_to_feature_ids=cluster_id_to_feature_ids,
            X_test=X_test_red,
            y_test=y_test,
            output_dir=args.output_dir,
            git_hash=git_hash,
            label_tag=label_tag,
            n_repeats=args.permutation_repeats,
            random_state=args.random_state,
            target_col=target_col,
        )

        print(f"\n── Feature importance [{tag_label}] (median rank) ──")
        print(feat_imp_df[['median_rank', 'rank_std', 'related features']].to_string())

        plt.close('all')
        plot_top_features_vs_targets(
            X_train_red, y_train, feat_imp_df,
            top_n=args.top_features,
            output_dir=args.output_dir,
            git_hash=git_hash,
            label_tag=label_tag,
            target_col=target_col,
        )


if __name__ == '__main__':
    main()
