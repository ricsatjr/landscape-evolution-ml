"""
02b_generate_sstr_features.py

Generate a mixed steady-state / transient (sstr) features dataset by sampling
rows from the steady-state and transient features DataFrames using a binary
mask. The mask determines, for each landscape (row), whether its features come
from the steady-state or the transient snapshot. The two subsets are mutually
exclusive and together cover all rows.

Directory structure assumed:
    <commit-hash>/
    ├── steady-state/
    │   └── features/
    │       └── features-<label_tag>-<git_hash>.pkl
    └── transient/
        └── features/
            └── features-<label_tag>-<git_hash>.pkl

Output written to:
    <commit-hash>/
    └── sstr/
        ├── features/
        │   ├── features-<label_tag>-<git_hash>.pkl   (mixed DataFrame)
        │   └── mask-<label_tag>-<git_hash>.npy        (binary mask, 1=ss, 0=tr)
        └── models/
            ├── full-features/
            └── reduced-features/

Usage
-----
    python 02b_generate_sstr_features.py \\
        --ss-features  <path/to/steady-state/features/features*.pkl> \\
        --tr-features  <path/to/transient/features/features*.pkl> \\
        --output-dir   <path/to/commit-hash/sstr> \\
        [--ss-fraction 0.5] \\
        [--seed 42]

Arguments
---------
--ss-features   Path to the steady-state features pickle file.
--tr-features   Path to the transient features pickle file.
--output-dir    Root of the sstr output folder (will be created if absent).
--ss-fraction   Fraction of rows drawn from steady-state (default: 0.5).
                The complement (1 - ss_fraction) comes from transient.
--seed          Random seed for reproducibility (default: 42).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(path: Path) -> pd.DataFrame:
    """Load a features DataFrame from a pickle file."""
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in {path}, got {type(df)}")
    return df


def extract_git_hash(path: Path) -> str:
    """
    Extract the git hash from a features filename.

    Expected filename pattern: features-<git_hash>.pkl
    Returns the portion after the 'features-' prefix.
    """
    stem = path.stem  # e.g. "features-abc1234"
    prefix = "features-"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    raise ValueError(
        f"Cannot extract git hash from filename '{path.name}'. "
        f"Expected pattern: features-<git_hash>.pkl"
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def generate_sstr(
    ss_features_path: Path,
    tr_features_path: Path,
    output_dir: Path,
    ss_fraction: float = 0.5,
    seed: int = 42,
) -> None:

    # --- Validate fraction ------------------------------------------------
    if not (0.0 < ss_fraction < 1.0):
        raise ValueError(
            f"--ss-fraction must be strictly between 0 and 1, got {ss_fraction}"
        )

    # --- Load DataFrames --------------------------------------------------
    print(f"Loading steady-state features from: {ss_features_path}")
    ss_df = load_features(ss_features_path)

    print(f"Loading transient features from:    {tr_features_path}")
    tr_df = load_features(tr_features_path)

    n_ss, n_cols_ss = ss_df.shape
    n_tr, n_cols_tr = tr_df.shape

    if n_ss != n_tr:
        raise ValueError(
            f"Row count mismatch: steady-state has {n_ss} rows, "
            f"transient has {n_tr} rows. They must be equal."
        )
    if n_cols_ss != n_cols_tr:
        raise ValueError(
            f"Column count mismatch: steady-state has {n_cols_ss} columns, "
            f"transient has {n_cols_tr} columns."
        )
    if list(ss_df.columns) != list(tr_df.columns):
        raise ValueError(
            "Column names differ between steady-state and transient DataFrames."
        )

    N = n_ss
    print(f"\nDataFrame shape: {N} rows × {n_cols_ss} columns")
    print(f"ss_fraction = {ss_fraction:.4f}  →  "
          f"~{int(round(ss_fraction * N))} rows from steady-state, "
          f"~{N - int(round(ss_fraction * N))} rows from transient")

    # --- Generate binary mask ---------------------------------------------
    rng = np.random.default_rng(seed)

    # Build a mask with exactly round(ss_fraction * N) ones
    n_ss_rows = int(round(ss_fraction * N))
    mask = np.zeros(N, dtype=np.int8)
    ss_indices = rng.choice(N, size=n_ss_rows, replace=False)
    mask[ss_indices] = 1

    n_actual_ss = mask.sum()
    n_actual_tr = N - n_actual_ss
    print(f"\nMask generated (seed={seed}):")
    print(f"  1s (steady-state rows): {n_actual_ss}")
    print(f"  0s (transient rows):    {n_actual_tr}")

    # --- Mix the DataFrames -----------------------------------------------
    mixed_df = ss_df.copy()
    tr_rows = (mask == 0)
    mixed_df.iloc[tr_rows] = tr_df.iloc[tr_rows].values

    print(f"\nMixed DataFrame shape: {mixed_df.shape}")

    # --- Build output paths -----------------------------------------------
    ss_hash = extract_git_hash(ss_features_path)
    tr_hash = extract_git_hash(tr_features_path)
    if ss_hash != tr_hash:
        raise ValueError(
            f"Git hash mismatch: steady-state file has '{ss_hash}', "
            f"transient file has '{tr_hash}'. Inputs must come from the same pipeline run."
        )
    git_hash = ss_hash
    ss_pct = int(round(ss_fraction * 100))  # e.g. 0.5 -> 50, 0.7 -> 70
    fraction_tag = f"ss{ss_pct}"            # e.g. "ss50", "ss70"

    features_dir = output_dir / "features"
    models_full_dir = output_dir / "models" / "full-features"
    models_reduced_dir = output_dir / "models" / "reduced-features"

    for d in [features_dir, models_full_dir, models_reduced_dir]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Directory ensured: {d}")

    features_out = features_dir / f"features-{fraction_tag}-{git_hash}.pkl"
    mask_out = features_dir / f"mask-{fraction_tag}-{git_hash}.npy"

    # --- Save outputs -----------------------------------------------------
    mixed_df.to_pickle(features_out)
    print(f"\nSaved mixed features → {features_out}")

    np.save(mask_out, mask)
    print(f"Saved binary mask    → {mask_out}")

    # --- Summary ----------------------------------------------------------
    print("\n--- Summary ---")
    print(f"  Total rows:           {N}")
    print(f"  From steady-state:    {n_actual_ss} ({100 * n_actual_ss / N:.1f}%)")
    print(f"  From transient:       {n_actual_tr} ({100 * n_actual_tr / N:.1f}%)")
    print(f"  Random seed:          {seed}")
    print(f"  Features file:        {features_out}")
    print(f"  Mask file:            {mask_out}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sstr (mixed steady-state/transient) features dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ss-features", required=True, type=Path,
        help="Path to steady-state features pickle file.",
    )
    parser.add_argument(
        "--tr-features", required=True, type=Path,
        help="Path to transient features pickle file.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Root output directory for sstr (e.g. <commit-hash>/sstr).",
    )
    parser.add_argument(
        "--ss-fraction", type=float, default=0.5,
        help="Fraction of rows drawn from steady-state (0 < value < 1).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve and validate input paths
    ss_path = args.ss_features.resolve()
    tr_path = args.tr_features.resolve()
    out_dir = args.output_dir.resolve()

    if not ss_path.exists():
        print(f"ERROR: Steady-state features file not found: {ss_path}", file=sys.stderr)
        sys.exit(1)
    if not tr_path.exists():
        print(f"ERROR: Transient features file not found: {tr_path}", file=sys.stderr)
        sys.exit(1)

    generate_sstr(
        ss_features_path=ss_path,
        tr_features_path=tr_path,
        output_dir=out_dir,
        ss_fraction=args.ss_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
