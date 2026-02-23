# Trained Models

The trained machine learning model objects are archived on Zenodo and are not
stored in this repository.

## Download

**Zenodo record:** https://doi.org/10.5281/zenodo.18739197

Three pickle files are available:

| File | Features | Targets | Mean R² |
|------|----------|---------|---------|
| `trained-models-full-features-ratios.pkl` | All 39 | U/Ks, Kh/Ks | 0.999 |
| `trained-models-reduced-features-ratios.pkl` | 14 (clustered) | U/Ks, Kh/Ks | 0.935 |
| `trained-models-full-features-individual-params.pkl` | All 39 | U, Kh, Ks | 0.092 |

The individual-parameter pickle is included to demonstrate that individual LE
parameters cannot be reliably estimated from topography alone, in contrast to
the dimensionless ratios (R² > 0.99 vs R² < 0.10).

---

## File Structure

Each pickle file is a nested dictionary with the following structure:

```python
results[algorithm]                        # keyed by shortname: lin, lso, knn, svm, dtr, rfo, gbs, mlp
├── 'test_r2'          list[float]        # overall R² per outer CV fold
├── 'test_mse'         list[float]
├── 'test_rmse'        list[float]
├── 'test_mae'         list[float]
├── 'best_params'      list[dict]         # best hyperparameters per outer fold
├── 'per_target_r2'    list[array]        # per-target R² per outer fold
├── 'per_target_mse'   list[array]
├── 'per_target_rmse'  list[array]
├── 'per_target_mae'   list[array]
├── 'mean_test_r2'     float              # mean R² across outer folds
├── 'std_test_r2'      float
├── 'mean_test_mse'    float
├── 'mean_test_rmse'   float
├── 'mean_test_mae'    float
└── 'final_model'      dict
    ├── 'hyperparams'        dict         # selected hyperparameters
    ├── 'regressor'          Pipeline     # fitted StandardScaler + estimator
    ├── 'test_set_r2'        list[float]  # per-target R² + overall mean (last element)
    ├── 'test_set_rmse'      list[float]
    └── 'test_set_mae'       list[float]
```

The fitted pipeline is stored under `results[algorithm]['final_model']['regressor']`
and can be used directly for prediction:

```python
import pickle
import numpy as np

with open('trained-models-full-features-ratios.pkl', 'rb') as f:
    results = pickle.load(f)

# Use the MLP final model for prediction
model = results['mlp']['final_model']['regressor']

# X must be a DataFrame or array with the 39 topographic features
# in the column order written by 02_extract_features.py
y_pred = model.predict(X)   # returns log10(U/Ks), log10(Kh/Ks)
```

---

## Software Requirements

Models were trained with **scikit-learn 1.6.1** and **Python 3.12.2**.
Loading these files with a different scikit-learn version may produce
`InconsistentVersionWarning` but should remain functional for minor version
differences. See `environment.yml` for the full pinned software environment.

---

## Retraining

To retrain the models from scratch, run:

```bash
python pipeline/03_train_models.py \
    --data-dir data/features \
    --output-dir outputs \
    --labels u_ks kh_ks \
    --n-outer 10 --n-inner 5 --n-iter 20
```

Note that retraining will produce numerically similar but not bit-identical
results due to randomness in the training algorithms (weight initialisation,
bootstrap sampling). The archived pickles are the authoritative trained models
used to produce the results in the paper.

---

## Citation

If you use these models, please cite:

> Saturay, R.L., Ramos, N.T., & Bantang, J.Y. (2025). [Title].
> Journal of Geophysical Research: Earth Surface. [DOI]

And the Zenodo record directly:

> Saturay, R.L., Ramos, N.T., & Bantang, J.Y. (2025). Trained machine learning
> models for estimating landscape evolution parameters from topographic features.
> Zenodo. https://doi.org/10.5281/zenodo.18739197
