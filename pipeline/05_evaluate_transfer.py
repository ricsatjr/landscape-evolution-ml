"""
05_evaluate_transfer.py
=======================
Within-domain and cross-domain transfer evaluation for steady-state (SS),
transient (TR), and optionally mixed-domain (MX) landscape evolution ML
models.

Assumptions
-----------
- SS, TR, and MX models and features were produced under identical
  conditions (same feature set, label set, train/test split logic) except
  for the landscape regime. The _meta dicts of all bundles must agree on
  feature_names and label_names; an error is raised otherwise.
- The held-out test set is identified by integer indices stored in
  _meta['test_idx'] of each nested-CV results pkl. These indices are used
  to slice the corresponding features dataframe.
- Feature names and label names are read from _meta of the SS bundle
  (single source of truth after validation).

Evaluation conditions
---------------------
  Without MX:
    SS->SS  : SS model on SS held-out test features  (within-domain)
    TR->TR  : TR model on TR held-out test features  (within-domain)
    TR->SS  : TR model on SS held-out test features  (cross-domain)
    SS->TR  : SS model on TR held-out test features  (cross-domain)

  With MX (--models-mx / --features-mx supplied):
    MX->MX  : MX model on MX held-out test features  (within-domain)
    MX->SS  : MX model on SS held-out test features  (mixed transfer)
    MX->TR  : MX model on TR held-out test features  (mixed transfer)

Output
------
A pickle file with keys 'within', 'cross', optionally 'mixed', and '_meta':
    results[alg][condition][target] = (mean_r2, lower_ci, upper_ci)
    target is one of: 'u_ks', 'kh_ks'

Output filename: transfer-{git_hash}.pkl
where git_hash is read from _meta['git_hash'] of the SS model bundle.

Usage
-----
python 05_evaluate_transfer.py \\
    --models-ss   path/to/ss/nested*.pkl \\
    --models-tr   path/to/tr/nested*.pkl \\
    --features-ss path/to/ss/features*.pkl \\
    --features-tr path/to/tr/features*.pkl \\
    [--models-mx   path/to/mx/nested*.pkl] \\
    [--features-mx path/to/mx/features*.pkl] \\
    [--n-bootstrap 1000] \\
    [--ci 95] \\
    [--seed INT] \\
    [--output-dir results/transfer/]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.utils import resample


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_glob(pattern: str) -> Path:
    """Resolve a glob pattern or direct path to exactly one file."""
    p = Path(pattern)
    if p.exists():
        return p
    matches = sorted(Path(".").glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    if len(matches) > 1:
        raise ValueError(
            f"Pattern '{pattern}' matched {len(matches)} files; "
            f"provide a more specific path.\nMatches: {matches}"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Meta validation
# ---------------------------------------------------------------------------

# Keys that must be identical between SS and TR _meta dicts.
REQUIRED_MATCHING_KEYS = ["feature_names", "label_names"]


def validate_meta(meta_ref: dict, meta_other: dict, label: str) -> None:
    """
    Raise ValueError if any REQUIRED_MATCHING_KEYS differ between
    meta_ref (SS, used as reference) and meta_other.
    Raise KeyError if a required key is absent from either.
    """
    for key in REQUIRED_MATCHING_KEYS:
        if key not in meta_ref:
            raise KeyError(f"'_meta' of SS models is missing required key: '{key}'")
        if key not in meta_other:
            raise KeyError(
                f"'_meta' of {label} models is missing required key: '{key}'"
            )

    mismatches = [
        key for key in REQUIRED_MATCHING_KEYS
        if meta_ref[key] != meta_other[key]
    ]

    if mismatches:
        lines = [
            f"SS and {label} _meta dicts differ on required keys. "
            f"Models must be trained under identical conditions.\n"
        ]
        for key in mismatches:
            lines.append(f"  Key   : '{key}'")
            lines.append(f"    SS  : {meta_ref[key]}")
            lines.append(f"    {label:4s}: {meta_other[key]}")
        raise ValueError("\n".join(lines))


def get_algorithms(models: dict) -> list:
    """Return sorted algorithm keys, excluding '_meta'."""
    return sorted(k for k in models.keys() if k != "_meta")


# ---------------------------------------------------------------------------
# Bootstrap R2
# ---------------------------------------------------------------------------

def bootstrap_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Bootstrapped R2 mean and percentile CI for each target column.

    Parameters
    ----------
    y_true, y_pred : ndarray of shape (n_samples, n_targets)
    n_bootstrap    : int
    ci             : float, confidence level in percent
    rng            : numpy Generator for reproducibility

    Returns
    -------
    mean, lower, upper : each ndarray of shape (n_targets,)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples, n_targets = y_true.shape
    scores  = np.zeros((n_bootstrap, n_targets))
    idx_all = np.arange(n_samples)

    for i in range(n_bootstrap):
        idx = resample(
            idx_all, replace=True,
            random_state=int(rng.integers(0, 2**31)),
        )
        for t in range(n_targets):
            scores[i, t] = r2_score(y_true[idx, t], y_pred[idx, t])

    alpha = (100.0 - ci) / 2.0
    lower = np.percentile(scores, alpha, axis=0)
    upper = np.percentile(scores, 100.0 - alpha, axis=0)
    mean  = scores.mean(axis=0)
    return mean, lower, upper


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def get_test_arrays(
    features_df,
    test_idx: list,
    feature_names: list,
    label_names: list,
) -> tuple:
    """
    Slice features_df to the held-out test indices and return
    X (feature matrix) and y_true (log10-transformed labels).
    """
    df_test = features_df.iloc[test_idx]
    X       = df_test[feature_names]
    y_true  = np.log10(df_test[label_names].values)
    return X, y_true


def evaluate_condition(
    model,
    X,
    y_true: np.ndarray,
    n_bootstrap: int,
    ci: float,
    rng: np.random.Generator,
) -> dict:
    """
    Predict with model on X, then compute bootstrapped R2 against y_true.

    Returns
    -------
    dict with keys 'u_ks' and 'kh_ks', each a (mean, lower, upper) tuple.
    """
    y_pred = model.predict(X)
    mean, lo, hi = bootstrap_r2(
        y_true, y_pred, n_bootstrap=n_bootstrap, ci=ci, rng=rng,
    )
    return {
        "u_ks":  (mean[0], lo[0], hi[0]),
        "kh_ks": (mean[1], lo[1], hi[1]),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results(results: dict, section_title: str) -> None:
    print(f"\n{'#' * 60}")
    print(f"  {section_title}")
    print(f"{'#' * 60}")
    for alg, conditions in results.items():
        print(f"\n{'=' * 50}\n{alg}")
        for condition, targets in conditions.items():
            src, dst = condition.split("_on_")
            print(f"  {src.upper()}->{dst.upper()}")
            for target_key, (m, lo, hi) in targets.items():
                target_label = (
                    "log(U/Ks): " if target_key == "u_ks" else "log(Kh/Ks):"
                )
                print(f"    {target_label}  {m:.4f}  [{lo:.4f}, {hi:.4f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Within-domain and cross-domain transfer evaluation "
            "for SS and TR landscape evolution ML models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-ss", required=True, metavar="PATH",
        help="Path (or glob) to the SS nested-CV model pkl file.",
    )
    parser.add_argument(
        "--models-tr", required=True, metavar="PATH",
        help="Path (or glob) to the TR nested-CV model pkl file.",
    )
    parser.add_argument(
        "--features-ss", required=True, metavar="PATH",
        help="Path (or glob) to the SS features pkl file.",
    )
    parser.add_argument(
        "--features-tr", required=True, metavar="PATH",
        help="Path (or glob) to the TR features pkl file.",
    )
    parser.add_argument(
        "--models-mx", default=None, metavar="PATH",
        help="Path (or glob) to the MX nested-CV model pkl file (optional).",
    )
    parser.add_argument(
        "--features-mx", default=None, metavar="PATH",
        help="Path (or glob) to the MX features pkl file (optional).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000, metavar="N",
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--ci", type=float, default=95.0, metavar="LEVEL",
        help="Confidence interval level in percent.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="INT",
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir", default=".", metavar="DIR",
        help="Directory for the output pkl file.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print("Loading model bundles...")
    path_models_ss   = resolve_glob(args.models_ss)
    path_models_tr   = resolve_glob(args.models_tr)
    path_features_ss = resolve_glob(args.features_ss)
    path_features_tr = resolve_glob(args.features_tr)

    models_ss      = load_pickle(path_models_ss)
    models_tr      = load_pickle(path_models_tr)

    # MX is optional; validate that both --models-mx and --features-mx
    # are either both supplied or both absent.
    use_mx = args.models_mx is not None or args.features_mx is not None
    if use_mx and (args.models_mx is None or args.features_mx is None):
        parser.error(
            "--models-mx and --features-mx must be supplied together."
        )

    if use_mx:
        path_models_mx   = resolve_glob(args.models_mx)
        path_features_mx = resolve_glob(args.features_mx)
        models_mx        = load_pickle(path_models_mx)
    else:
        path_models_mx = path_features_mx = models_mx = None

    print("Loading feature dataframes...")
    features_ss_df = load_pickle(path_features_ss)
    features_tr_df = load_pickle(path_features_tr)
    features_mx_df = load_pickle(path_features_mx) if use_mx else None

    # ------------------------------------------------------------------
    # Validate _meta consistency
    # ------------------------------------------------------------------
    print("Validating _meta consistency...")
    meta_ss = models_ss["_meta"]
    meta_tr = models_tr["_meta"]
    validate_meta(meta_ss, meta_tr, label="TR")
    if use_mx:
        meta_mx = models_mx["_meta"]
        validate_meta(meta_ss, meta_mx, label="MX")
    else:
        meta_mx = None

    # Single source of truth for shared metadata
    feature_names = meta_ss["feature_names"]
    label_names   = meta_ss["label_names"]
    test_idx_ss   = meta_ss["test_idx"]
    test_idx_tr   = meta_tr["test_idx"]
    test_idx_mx   = meta_mx["test_idx"] if use_mx else None

    print(f"  Features : {len(feature_names)}")
    print(f"  Labels   : {label_names}")
    print(f"  SS test n: {len(test_idx_ss)}")
    print(f"  TR test n: {len(test_idx_tr)}")
    if use_mx:
        print(f"  MX test n: {len(test_idx_mx)}")

    # ------------------------------------------------------------------
    # Reconcile algorithms
    # ------------------------------------------------------------------
    algs_ss = set(get_algorithms(models_ss))
    algs_tr = set(get_algorithms(models_tr))
    all_alg_sets = {"SS": algs_ss, "TR": algs_tr}
    if use_mx:
        algs_mx = set(get_algorithms(models_mx))
        all_alg_sets["MX"] = algs_mx
    else:
        algs_mx = algs_ss  # unused, set for intersection logic

    common = algs_ss & algs_tr & (algs_mx if use_mx else algs_ss)
    for domain, alg_set in all_alg_sets.items():
        only_this = alg_set - common
        if only_this:
            print(
                f"WARNING: algorithms only in {domain} bundle: "
                f"{sorted(only_this)}",
                file=sys.stderr,
            )
    algorithms = sorted(common)
    print(f"  Algorithms: {algorithms}")

    # ------------------------------------------------------------------
    # Pre-slice held-out test arrays
    # ------------------------------------------------------------------
    X_ss, y_ss = get_test_arrays(
        features_ss_df, test_idx_ss, feature_names, label_names,
    )
    X_tr, y_tr = get_test_arrays(
        features_tr_df, test_idx_tr, feature_names, label_names,
    )
    if use_mx:
        X_mx, y_mx = get_test_arrays(
            features_mx_df, test_idx_mx, feature_names, label_names,
        )

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Within-domain evaluation  (SS->SS, TR->TR, and optionally MX->MX)
    # ------------------------------------------------------------------
    print(f"\nRunning within-domain evaluation "
          f"(n_bootstrap={args.n_bootstrap}, CI={args.ci}%)...")

    results_within = {}
    for alg in algorithms:
        print(f"  {alg}...", end=" ", flush=True)
        results_within[alg] = {
            "ss_on_ss": evaluate_condition(
                models_ss[alg]["final_model"]["regressor"],
                X_ss, y_ss, args.n_bootstrap, args.ci, rng,
            ),
            "tr_on_tr": evaluate_condition(
                models_tr[alg]["final_model"]["regressor"],
                X_tr, y_tr, args.n_bootstrap, args.ci, rng,
            ),
        }
        if use_mx:
            results_within[alg]["mx_on_mx"] = evaluate_condition(
                models_mx[alg]["final_model"]["regressor"],
                X_mx, y_mx, args.n_bootstrap, args.ci, rng,
            )
        print("done")

    # ------------------------------------------------------------------
    # Cross-domain evaluation  (TR->SS, SS->TR)
    # ------------------------------------------------------------------
    print(f"\nRunning cross-domain evaluation "
          f"(n_bootstrap={args.n_bootstrap}, CI={args.ci}%)...")

    results_cross = {}
    for alg in algorithms:
        print(f"  {alg}...", end=" ", flush=True)
        results_cross[alg] = {
            "tr_on_ss": evaluate_condition(
                models_tr[alg]["final_model"]["regressor"],
                X_ss, y_ss, args.n_bootstrap, args.ci, rng,
            ),
            "ss_on_tr": evaluate_condition(
                models_ss[alg]["final_model"]["regressor"],
                X_tr, y_tr, args.n_bootstrap, args.ci, rng,
            ),
        }
        print("done")

    # ------------------------------------------------------------------
    # Mixed transfer evaluation  (MX->SS, MX->TR)
    # ------------------------------------------------------------------
    results_mixed = {}
    if use_mx:
        print(f"\nRunning mixed-domain transfer evaluation "
              f"(n_bootstrap={args.n_bootstrap}, CI={args.ci}%)...")
        for alg in algorithms:
            print(f"  {alg}...", end=" ", flush=True)
            results_mixed[alg] = {
                "mx_on_ss": evaluate_condition(
                    models_mx[alg]["final_model"]["regressor"],
                    X_ss, y_ss, args.n_bootstrap, args.ci, rng,
                ),
                "mx_on_tr": evaluate_condition(
                    models_mx[alg]["final_model"]["regressor"],
                    X_tr, y_tr, args.n_bootstrap, args.ci, rng,
                ),
            }
            print("done")

    # ------------------------------------------------------------------
    # Print summaries
    # ------------------------------------------------------------------
    print_results(results_within, "WITHIN-DOMAIN RESULTS")
    print_results(results_cross,  "CROSS-DOMAIN RESULTS")
    if use_mx:
        print_results(results_mixed, "MIXED TRANSFER RESULTS (MX->SS, MX->TR)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    git_hash = meta_ss["git_hash"]

    output = {
        "within": results_within,
        "cross":  results_cross,
        "_meta": {
            "algorithms":    algorithms,
            "n_bootstrap":   args.n_bootstrap,
            "ci":            args.ci,
            "seed":          args.seed,
            "git_hash":      git_hash,
            "feature_names": feature_names,
            "label_names":   label_names,
            "test_idx_ss":   test_idx_ss,
            "test_idx_tr":   test_idx_tr,
            "models_ss":     str(path_models_ss),
            "models_tr":     str(path_models_tr),
            "features_ss":   str(path_features_ss),
            "features_tr":   str(path_features_tr),
        },
    }
    if use_mx:
        output["mixed"] = results_mixed
        output["_meta"].update({
            "test_idx_mx": test_idx_mx,
            "models_mx":   str(path_models_mx),
            "features_mx": str(path_features_mx),
        })

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"transfer-{git_hash}.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
