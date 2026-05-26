"""
Utility functions for data loading and result persistence.

Functions
---------
load_representation  — load a preprocessed taxa .npz and return train/test arrays
save_rep_to_excel    — write per-representation result sheets to the Excel workbook
"""

import os
import numpy as np
import pandas as pd

from config import RESULTS_XLSX


def load_representation(taxa_path):
    """Load a preprocessed taxa .npz file and return train/test splits.

    Args:
        taxa_path : str — path to .npz file (e.g. preprocessed_binary.npz)

    Returns:
        X_train : float32 array — training images
        X_test  : float32 array — test images
        y_train : float32 array — training labels
        y_test  : float32 array — test labels
    """
    npz     = np.load(taxa_path)
    X_train = npz['X_train'].astype('float32')
    X_test  = npz['X_test'].astype('float32')
    y_train = npz['y_train'].astype('float32')
    y_test  = npz['y_test'].astype('float32')
    return X_train, X_test, y_train, y_test


def save_rep_to_excel(rep_name, gs_df, cv_metrics, test_metrics,
                      perm_df=None, grad_df=None, results_path=None):
    """Write result sheets for one representation to the Excel workbook.

    Appends to an existing file or creates a new one. Existing sheets with the
    same name are replaced (safe to re-run after a crash).

    Sheets written:
      gs_{rep_name}    — grid search results
      cv_{rep_name}    — per-fold CV metrics (fold column prepended)
      test_{rep_name}  — test set metrics
      perm_{rep_name}  — permutation importance (if provided)
      grad_{rep_name}  — gradient importance (if provided)

    Args:
        rep_name     : str       — sheet name suffix (e.g. 'binary', 'log_io')
        gs_df        : DataFrame — from run_grid_search
        cv_metrics   : list[dict] — from run_kfold_cv
        test_metrics : dict      — from run_final_model_*
        perm_df      : DataFrame or None — from run_importance_analysis
        grad_df      : DataFrame or None — from run_importance_analysis
        results_path : str or None — override default RESULTS_XLSX
    """
    path   = results_path or RESULTS_XLSX
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode   = 'a' if os.path.exists(path) else 'w'
    kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}

    with pd.ExcelWriter(path, engine='openpyxl', mode=mode, **kwargs) as writer:
        gs_df.to_excel(writer, sheet_name=f'gs_{rep_name}', index=False)

        cv_df = pd.DataFrame(cv_metrics)
        cv_df.insert(0, 'fold', range(1, len(cv_metrics) + 1))
        cv_df.to_excel(writer, sheet_name=f'cv_{rep_name}', index=False)

        pd.DataFrame([test_metrics]).to_excel(
            writer, sheet_name=f'test_{rep_name}', index=False)

        if perm_df is not None:
            perm_df.to_excel(writer, sheet_name=f'perm_{rep_name}', index=False)
        if grad_df is not None:
            grad_df.to_excel(writer, sheet_name=f'grad_{rep_name}', index=False)

    sheets = 'gs, cv, test'
    if perm_df is not None:
        sheets += ', perm, grad'
    print(f"  Saved -> {path}  ({sheets}  for {rep_name})")
