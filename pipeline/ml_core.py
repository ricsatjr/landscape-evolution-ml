"""
ml_core.py
----------
Shared ML components for the landscape-evolution training pipeline.

Imported by both ``03_train_models.py`` (full-feature training) and
``04_feature_importance.py`` (reduced-feature training + importance analysis).

Contents
--------
get_random_search_params()      Hyperparameter search spaces for 8 algorithms
MultiOutputRegressionToolkit    Multi-output routing and param-prefix handling
nested_cv()                     Nested cross-validation loop
select_final_hyperparameters()  Frequency/score-based hyperparameter selection
train_final_model()             Fit a single final pipeline; return metrics

References
----------
Hyperparameter ranges follow Bergstra & Bengio (2012) and scikit-learn docs.
AI-assisted development: https://claude.ai/chat/bd51aa3b-3ef2-4820-916e-fcb29843f920
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, MultiTaskLasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# =============================================================================
# Hyperparameter search spaces
# =============================================================================

def get_random_search_params() -> dict:
    """Return hyperparameter search spaces for all eight algorithms.

    Returns
    -------
    dict
        ``{shortname: (estimator, param_distributions)}``

    Shortnames
    ----------
    lin : LinearRegression
    lso : MultiTaskLasso
    knn : KNeighborsRegressor
    svm : SVR              (wrapped in MultiOutputRegressor downstream)
    dtr : DecisionTreeRegressor
    rfo : RandomForestRegressor
    gbs : GradientBoostingRegressor  (wrapped in MultiOutputRegressor)
    mlp : MLPRegressor
    """
    linear_params = {
        'fit_intercept': [True, False],
        'positive':      [True, False],
    }

    mtlasso_params = {
        'alpha':         loguniform(0.001, 10.0),
        'tol':           [1e-4, 1e-3],
        'max_iter':      [1000, 5000],
        'fit_intercept': [True, False],
        'warm_start':    [True, False],
    }

    kn_params = {
        'n_neighbors': randint(1, 31),
        'weights':     ['uniform', 'distance'],
        'p':           [1, 2],
        'leaf_size':   randint(10, 101),
    }

    svr_params = {
        'kernel':  ['linear', 'rbf', 'poly'],
        'C':       loguniform(0.001, 1000),
        'epsilon': loguniform(0.01, 1.0),
        'gamma':   ['scale', 'auto'],
    }

    dt_params = {
        'max_depth':         randint(3, 21),
        'min_samples_split': randint(2, 21),
        'min_samples_leaf':  randint(1, 21),
        'max_features':      ['sqrt', 'log2', None],
    }

    rf_params = {
        'n_estimators':      randint(50, 501),
        'max_depth':         randint(3, 21),
        'min_samples_split': randint(2, 21),
        'min_samples_leaf':  randint(1, 21),
        'max_features':      ['sqrt', 'log2', None],
    }

    gb_params = {
        'n_estimators':      randint(50, 501),
        'learning_rate':     loguniform(0.001, 0.5),
        'max_depth':         randint(2, 11),
        'subsample':         uniform(0.5, 0.5),
        'min_samples_split': randint(2, 21),
    }

    mlp_params = {
        'hidden_layer_sizes': [
            (16,), (64,), (128,),
            (16, 8), (64, 32), (128, 64),
            (64, 32, 8), (128, 64, 32),
            (64, 32, 16, 8), (128, 64, 32, 16, 8),
        ],
        'activation':         ['relu', 'tanh', 'logistic'],
        'solver':             ['adam', 'sgd', 'lbfgs'],
        'learning_rate':      ['constant', 'adaptive', 'invscaling'],
        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
        'alpha':              [0.0001, 0.001, 0.01, 0.1],
        'batch_size':         [8, 32, 128],
        'max_iter':           [200, 500, 1000],
        'early_stopping':     [True],
        'tol':                [1e-4, 1e-3],
        'momentum':           [0.0, 0.5, 0.9],
    }

    return {
        'lin': (LinearRegression(),          linear_params),
        'lso': (MultiTaskLasso(),            mtlasso_params),
        'knn': (KNeighborsRegressor(),       kn_params),
        'svm': (SVR(max_iter=int(10e4)),     svr_params),
        'dtr': (DecisionTreeRegressor(),     dt_params),
        'rfo': (RandomForestRegressor(),     rf_params),
        'gbs': (GradientBoostingRegressor(), gb_params),
        'mlp': (MLPRegressor(verbose=False), mlp_params),
    }


# =============================================================================
# Multi-output routing
# =============================================================================

class MultiOutputRegressionToolkit:
    """Manage multi-output–compatible regressors and their parameter spaces.

    Some estimators (SVR, GradientBoostingRegressor) do not natively support
    multi-output targets and must be wrapped in ``MultiOutputRegressor``.
    This class encapsulates that routing and constructs the correct Pipeline
    parameter-key prefixes for downstream ``RandomizedSearchCV`` calls.

    Parameters
    ----------
    models : dict
        Output of ``get_random_search_params()``.
    """

    _NATIVE  = {'lin', 'lso', 'dtr', 'rfo', 'knn', 'mlp'}
    _WRAPPED = {'svm', 'gbs'}

    def __init__(self, models: dict):
        self.models = models
        self.native_multi_output = {
            m: models[m][0] for m in self._NATIVE if m in models
        }
        self.wrapped_multi_output = {
            m: MultiOutputRegressor(models[m][0]) for m in self._WRAPPED if m in models
        }
        self.regressors = {**self.native_multi_output, **self.wrapped_multi_output}

    def get_regressor(self, name: str, clone_regressor: bool = True):
        """Return a (cloned) regressor by shortname."""
        if name not in self.regressors:
            raise ValueError(
                f"Unknown regressor '{name}'. Available: {sorted(self.regressors)}"
            )
        reg = self.regressors[name]
        return clone(reg) if clone_regressor else reg

    def is_wrapped(self, name: str) -> bool:
        return name in self.wrapped_multi_output

    def get_param_dict(self, name: str) -> dict:
        """Return param distributions prefixed for use inside a Pipeline.

        The pipeline step is named ``'piped'``, so native estimators use the
        ``'piped__'`` prefix and wrapped estimators use ``'piped__estimator__'``.
        """
        if name not in self.models:
            raise ValueError(
                f"Unknown algorithm '{name}'. Available: {sorted(self.models)}"
            )
        prefix = 'piped__estimator__' if self.is_wrapped(name) else 'piped__'
        return {f"{prefix}{k}": v for k, v in self.models[name][1].items()}


# =============================================================================
# Nested cross-validation
# =============================================================================

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
    X : pd.DataFrame
        Feature matrix (training set only).
    y : pd.DataFrame
        log₁₀-transformed target matrix (training set only).
    n_outer_splits : int
        Number of outer ShuffleSplit folds for generalisation estimation.
    n_inner_splits : int
        Number of inner ShuffleSplit folds for hyperparameter tuning.
    random_state : int
    n_iter : int
        Number of RandomizedSearchCV candidates per inner fold.

    Returns
    -------
    dict
        Nested results keyed by algorithm shortname.  Each value contains
        per-fold and summary metrics, best hyperparameters per fold, and
        per-target breakdowns.
    """
    models  = get_random_search_params()
    toolkit = MultiOutputRegressionToolkit(models)
    results = {}

    for reg_name in models:
        print(f"\n{'='*60}\n  Algorithm: {reg_name}\n{'='*60}")

        outer_cv = ShuffleSplit(
            n_splits=n_outer_splits, test_size=0.2, random_state=random_state
        )
        # TODO: use random_state + 1 for inner_cv to ensure outer and inner
        # splits are independently randomised. Currently both use the same
        # random_state, which introduces correlation between fold assignments.
        # Changing this will alter CV fold assignments and produce different
        # (though not necessarily worse) results — defer to v2 retraining.
        inner_cv = ShuffleSplit(
            n_splits=n_inner_splits, test_size=0.2, random_state=random_state
        )

        results[reg_name] = {
            'test_r2':         [],
            'test_mse':        [],
            'test_rmse':       [],
            'test_mae':        [],
            'best_params':     [],
            'per_target_r2':   [],
            'per_target_mse':  [],
            'per_target_rmse': [],
            'per_target_mae':  [],
        }

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
            print(f"  Outer fold {fold_idx}/{n_outer_splits}", end='', flush=True)

            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('piped',  toolkit.get_regressor(reg_name)),
            ])
            param_dist = toolkit.get_param_dict(reg_name)

            if param_dist:
                search = RandomizedSearchCV(
                    pipe, param_dist,
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
                best_params    = {}
                best_estimator = pipe
                best_estimator.fit(X_tr, y_tr)

            y_pred = best_estimator.predict(X_te)
            if isinstance(y_te, pd.DataFrame) and isinstance(y_pred, np.ndarray):
                y_pred = pd.DataFrame(y_pred, index=y_te.index, columns=y_te.columns)

            r2   = r2_score(y_te, y_pred, multioutput='uniform_average')
            mse  = mean_squared_error(y_te, y_pred, multioutput='uniform_average')
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(y_te, y_pred, multioutput='uniform_average')
            print(f"  →  R² = {r2:.3f}")

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
            f"  {reg_name}: mean R² = {results[reg_name]['mean_test_r2']:.3f}"
            f" ± {results[reg_name]['std_test_r2']:.3f}"
        )

    return results


# =============================================================================
# Final model selection and training
# =============================================================================

def select_final_hyperparameters(reg_results: dict) -> tuple:
    """Choose final hyperparameters from outer-fold best-params lists.

    Strategy (hierarchical):

    1. If any complete parameter set appears in more than one fold, select the
       most frequent.  Ties broken by the outer-fold R² score.
    2. Otherwise select the parameter set from the highest-scoring fold.

    Parameters
    ----------
    reg_results : dict
        Single-algorithm entry from the nested-CV results dict.

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

        best_score, best_params = -np.inf, None
        for cand in candidates:
            for params, score in zip(
                reg_results['best_params'], reg_results['test_r2']
            ):
                if set(params.items()) == set(cand) and score > best_score:
                    best_score  = score
                    best_params = dict(cand)
        return best_params, 'frequency_with_score_tiebreaker'

    best_idx = int(np.argmax(reg_results['test_r2']))
    return reg_results['best_params'][best_idx], 'best_score'


def train_final_model(
    reg_name: str,
    reg_results: dict,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> dict:
    """Fit a final pipeline for one algorithm and return metrics.

    Hyperparameters are selected from the nested-CV results via
    ``select_final_hyperparameters``.  The fitted pipeline is trained on the
    full training set and evaluated on the held-out test set.

    This function is intentionally free of plotting logic so it can be called
    by both ``03_train_models.py`` and ``04_feature_importance.py``.

    Parameters
    ----------
    reg_name : str
        Algorithm shortname (e.g. ``'rfo'``).
    reg_results : dict
        Single-algorithm entry from the nested-CV results dict.
    X_train, y_train : training data
    X_test, y_test : held-out test data

    Returns
    -------
    dict with keys:
        ``hyperparams``      — selected hyperparameter dict
        ``selection_method`` — how hyperparams were chosen
        ``regressor``        — fitted Pipeline object
        ``test_set_r2``      — per-target R² values + overall mean (last element)
        ``test_set_rmse``    — per-target RMSE + overall (last element)
        ``test_set_mae``     — per-target MAE + overall (last element)
    """
    models  = get_random_search_params()
    toolkit = MultiOutputRegressionToolkit(models)

    final_params, method = select_final_hyperparameters(reg_results)
    print(f"  {reg_name}: hyperparameter selection = {method}")

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('piped',  toolkit.get_regressor(reg_name, clone_regressor=True)),
    ])
    for param, value in final_params.items():
        try:
            pipe.set_params(**{param: value})
        except ValueError as exc:
            warnings.warn(f"Could not set {param}={value!r} for {reg_name}: {exc}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    pt_r2   = list(r2_score(y_test, y_pred, multioutput='raw_values'))
    pt_rmse = list(mean_squared_error(y_test, y_pred, multioutput='raw_values') ** 0.5)
    pt_mae  = list(mean_absolute_error(y_test, y_pred, multioutput='raw_values'))

    return {
        'hyperparams':      final_params,
        'selection_method': method,
        'regressor':        pipe,
        'test_set_r2':      pt_r2   + [r2_score(y_test, y_pred)],
        'test_set_rmse':    pt_rmse + [mean_squared_error(y_test, y_pred) ** 0.5],
        'test_set_mae':     pt_mae  + [mean_absolute_error(y_test, y_pred)],
    }
