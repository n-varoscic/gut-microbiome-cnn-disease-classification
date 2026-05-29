"""
Mirrors CNN_pipeline.ipynb exactly, with two differences:
  • Uses the Agg matplotlib backend (no display — figures are saved to disk as normal)
  • Runs from the project root, so all imports resolve without path gymnastics

Usage (local or SLURM):
    python run_pipeline.py

To skip representations after a mid-run crash, edit SKIP_REPS_IO / SKIP_REPS_META
at the top of each section below.
"""

# ── Backend must be set before any other matplotlib/pyplot import ─────────────
import matplotlib
matplotlib.use('Agg')

import sys
import os
import gc
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

from config import (
    BASE_SEED, IMG_SHAPE, METADATA_PATH, RESULTS_XLSX, VIZ_DIR,
    REPRESENTATIONS, META_FEATURE_NAMES, META_FEATURE_GROUPS,
)
from src.pipeline   import run_grid_search, run_kfold_cv, run_final_model_image, run_final_model_meta
from src.importance import run_importance_analysis, grouped_permutation_importance_meta
from src.utils      import load_representation, save_rep_to_excel
import src.visualization as viz

np.random.seed(BASE_SEED)
tf.random.set_seed(BASE_SEED)


def ts():
    """Compact timestamp for SLURM log lines."""
    return time.strftime('[%H:%M:%S]')


print(f"{ts()} TF  : {tf.__version__}")
print(f"{ts()} GPU : {tf.config.list_physical_devices('GPU')}")


# =============================================================================
# 1 · Load data
# =============================================================================

meta_npz = np.load(METADATA_PATH, allow_pickle=True)
M_train  = meta_npz['X_meta_train']
M_test   = meta_npz['X_meta_test']

print(f"{ts()} Metadata   train: {M_train.shape}   test: {M_test.shape}")
print(f"{ts()} Image shape     : {IMG_SHAPE}")
print(f"{ts()} Representations : {[r[0] for r in REPRESENTATIONS]}")


# =============================================================================
# 2 · Image-only Baseline  (Phases 1 → 2 → 3)
# =============================================================================

SKIP_REPS_IO = []   # ['binary', 'normalized', 'log'] to skip after a crash

io_summary_rows     = []
all_cv_histories_io = {}

for rep_name, taxa_path, meta_save_path in REPRESENTATIONS:
    if rep_name in SKIP_REPS_IO:
        print(f"{ts()}   Skipping {rep_name} (in SKIP_REPS_IO)")
        continue

    # Mirror the metadata model save path: best_cnn_{rep}_meta.keras → _io.keras
    io_save_path = meta_save_path.replace('_meta.keras', '_io.keras')
    print(f"\n{ts()} {'#'*58}\n{ts()} #  {rep_name.upper()}  —  image only\n{ts()} {'#'*58}")

    X_train, X_test, y_train, y_test = load_representation(taxa_path)
    print(f"{ts()}   Train: {X_train.shape}  Test: {X_test.shape}  "
          f"IBD train: {int(y_train.sum())}/{len(y_train)} ({100*y_train.mean():.1f}%)")

    # Phase 1: grid search
    best_params, gs_df = run_grid_search(X_train, y_train, IMG_SHAPE, rep_name)

    # Phase 2: 5-fold CV
    cv_metrics, cv_histories_io = run_kfold_cv(
        X_train, y_train, IMG_SHAPE, best_params, rep_name)

    # Phase 3: final model
    test_metrics, test_probs_io = run_final_model_image(
        X_train, y_train, X_test, y_test, IMG_SHAPE, best_params, io_save_path, rep_name)

    # Incremental Excel save
    save_rep_to_excel(f'{rep_name}_io', gs_df, cv_metrics, test_metrics)

    # Per-representation figures
    # model_label='io' → filename _io.png; title displays 'image-only'
    viz.plot_roc_curve(test_probs_io, y_test, rep_name, VIZ_DIR, model_label='io')
    viz.plot_confusion_matrix(test_probs_io, test_metrics['threshold'], y_test,
                              rep_name, VIZ_DIR, model_label='io')
    all_cv_histories_io[rep_name] = cv_histories_io

    aucs = [m['auc']    for m in cv_metrics]
    f1s  = [m['f1']     for m in cv_metrics]
    recs = [m['recall'] for m in cv_metrics]
    io_summary_rows.append({
        'representation' : rep_name,
        'best_lr'        : best_params['lr'],
        'best_batch_size': best_params['batch_size'],
        'best_conv_drop' : best_params['conv_drop'],
        'cv_auc_mean'    : np.mean(aucs),  'cv_auc_std'    : np.std(aucs),
        'cv_f1_mean'     : np.mean(f1s),   'cv_f1_std'     : np.std(f1s),
        'cv_recall_mean' : np.mean(recs),  'cv_recall_std' : np.std(recs),
        'test_auc'       : test_metrics['auc'],
        'test_f1'        : test_metrics['f1'],
        'test_precision' : test_metrics['precision'],
        'test_recall'    : test_metrics['recall'],
        'test_tp'        : test_metrics['tp'],  'test_tn': test_metrics['tn'],
        'test_fp'        : test_metrics['fp'],  'test_fn': test_metrics['fn'],
    })

    tf.keras.backend.clear_session(); gc.collect()
    print(f"{ts()}   Memory cleared after {rep_name}")

io_summary_df = pd.DataFrame(io_summary_rows)
mode   = 'a' if os.path.exists(RESULTS_XLSX) else 'w'
kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode=mode, **kwargs) as writer:
    io_summary_df.to_excel(writer, sheet_name='summary_io', index=False)
print(f"\n{ts()} Image-only complete.  Results -> {RESULTS_XLSX}")


# =============================================================================
# 3 · Metadata Model  (Phases 1 → 2 → 3 → 4)
# =============================================================================

SKIP_REPS_META = []  # ['binary', 'normalized', 'log'] to skip after a crash

meta_summary_rows     = []
all_cv_histories_meta = {}
all_perm_imp_meta     = {}
all_grad_imp_meta     = {}
all_attr_meta         = {}
all_y_test_meta       = {}
all_grouped_perm_meta = {}

for rep_name, taxa_path, meta_save_path in REPRESENTATIONS:
    if rep_name in SKIP_REPS_META:
        print(f"{ts()}   Skipping {rep_name} (in SKIP_REPS_META)")
        continue

    print(f"\n{ts()} {'#'*58}\n{ts()} #  {rep_name.upper()}  —  metadata\n{ts()} {'#'*58}")

    X_train, X_test, y_train, y_test = load_representation(taxa_path)
    print(f"{ts()}   Train: {X_train.shape}  Test: {X_test.shape}  "
          f"IBD train: {int(y_train.sum())}/{len(y_train)} ({100*y_train.mean():.1f}%)")

    # Phase 1: grid search
    best_params, gs_df = run_grid_search(
        X_train, y_train, IMG_SHAPE, rep_name, M_train=M_train)

    # Phase 2: 5-fold CV
    cv_metrics, cv_histories_meta = run_kfold_cv(
        X_train, y_train, IMG_SHAPE, best_params, rep_name, M_train=M_train)
    all_cv_histories_meta[rep_name] = cv_histories_meta

    # Phase 3: final model
    test_metrics, final_model_meta, final_scaler_meta, final_meta_scaler, test_probs_meta = \
        run_final_model_meta(
            X_train, M_train, y_train, X_test, M_test, y_test,
            IMG_SHAPE, best_params, meta_save_path, rep_name)

    # Phase 4a: individual feature importance (permutation + gradient)
    # Scale BOTH image and metadata exactly as the model saw them in training
    X_test_s_meta = final_scaler_meta.transform(
        X_test.reshape(len(X_test), -1)).reshape(X_test.shape)
    M_test_s_meta = final_meta_scaler.transform(M_test)
    perm_df, mean_attr, per_sample_attr = run_importance_analysis(
        final_model_meta, X_test_s_meta, M_test_s_meta, y_test,
        test_probs_meta, test_metrics['threshold'], META_FEATURE_NAMES, rep_name, VIZ_DIR)
    all_perm_imp_meta[rep_name] = perm_df
    all_grad_imp_meta[rep_name] = mean_attr
    all_attr_meta[rep_name]     = per_sample_attr
    all_y_test_meta[rep_name]   = y_test

    # Phase 4b: grouped permutation importance
    grouped_perm_df, _ = grouped_permutation_importance_meta(
        final_model_meta, X_test_s_meta, M_test_s_meta, y_test,
        META_FEATURE_GROUPS, META_FEATURE_NAMES)
    all_grouped_perm_meta[rep_name] = grouped_perm_df

    # Phase 4c: per-rep attribution heatmap (auto-scale; crash-safe)
    # Section 4 (summary) will regenerate these with the shared vmax across all reps.
    viz.plot_attribution_heatmap(
        per_sample_attr, perm_df, META_FEATURE_NAMES, y_test, rep_name, VIZ_DIR)

    # ── Incremental Excel save (crash-safe) ───────────────────────────────────
    grad_df = (pd.DataFrame({'feature': META_FEATURE_NAMES, 'mean_attr': mean_attr})
               .sort_values('mean_attr', ascending=False))

    # 1) Consolidated human-readable sheet (grid search + CV + test + importance)
    save_rep_to_excel(f'{rep_name}_meta', gs_df, cv_metrics, test_metrics,
                      perm_df, grad_df, grouped_perm_df)

    # 2) Machine-readable per-rep sheets (used by summary section reload after crash)
    attr_df = pd.DataFrame(per_sample_attr, columns=META_FEATURE_NAMES)
    attr_df.insert(0, 'y_test', y_test.astype(int))
    _mode = 'a' if os.path.exists(RESULTS_XLSX) else 'w'
    _kw   = {'if_sheet_exists': 'replace'} if _mode == 'a' else {}
    with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode=_mode, **_kw) as writer:
        perm_df.to_excel(        writer, sheet_name=f'perm_{rep_name}_meta',  index=False)
        grad_df.to_excel(        writer, sheet_name=f'grad_{rep_name}_meta',  index=False)
        grouped_perm_df.to_excel(writer, sheet_name=f'gperm_{rep_name}_meta', index=False)
        attr_df.to_excel(        writer, sheet_name=f'attr_{rep_name}_meta',  index=False)
    print(f"{ts()}   Importance reload data saved for {rep_name}")

    aucs = [m['auc']    for m in cv_metrics]
    f1s  = [m['f1']     for m in cv_metrics]
    recs = [m['recall'] for m in cv_metrics]
    meta_summary_rows.append({
        'representation' : rep_name,
        'best_lr'        : best_params['lr'],
        'best_batch_size': best_params['batch_size'],
        'best_conv_drop' : best_params['conv_drop'],
        'cv_auc_mean'    : np.mean(aucs),  'cv_auc_std'    : np.std(aucs),
        'cv_f1_mean'     : np.mean(f1s),   'cv_f1_std'     : np.std(f1s),
        'cv_recall_mean' : np.mean(recs),  'cv_recall_std' : np.std(recs),
        'test_auc'       : test_metrics['auc'],
        'test_f1'        : test_metrics['f1'],
        'test_precision' : test_metrics['precision'],
        'test_recall'    : test_metrics['recall'],
        'test_tp'        : test_metrics['tp'],  'test_tn': test_metrics['tn'],
        'test_fp'        : test_metrics['fp'],  'test_fn': test_metrics['fn'],
    })

    tf.keras.backend.clear_session(); gc.collect()
    print(f"{ts()}   Memory cleared after {rep_name}")

meta_summary_df = pd.DataFrame(meta_summary_rows)
mode   = 'a' if os.path.exists(RESULTS_XLSX) else 'w'
kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode=mode, **kwargs) as writer:
    meta_summary_df.to_excel(writer, sheet_name='summary_meta', index=False)
print(f"\n{ts()} Metadata model complete.  Results -> {RESULTS_XLSX}")


# =============================================================================
# 4 · Summary visualisations
# =============================================================================

print(f"\n{ts()} Generating summary visualisations...")

# Combined performance comparison (image-only vs metadata — AUC and Recall)
viz.plot_combined_performance(meta_summary_df, io_summary_df, REPRESENTATIONS, VIZ_DIR)

# Per-model performance bars (CV vs test AUC/F1 within each model)
# model_label='io' / 'meta' → mirrored filenames _io.png / _meta.png and
# mirrored titles 'CNN image-only' / 'CNN metadata'.
viz.plot_performance_bars(io_summary_df,   VIZ_DIR, model_label='io')
viz.plot_performance_bars(meta_summary_df, VIZ_DIR, model_label='meta')

if all_perm_imp_meta:
    # Individual permutation importance (all reps, 20 features)
    perm_compare = viz.plot_perm_importance_all(all_perm_imp_meta, META_FEATURE_NAMES, VIZ_DIR)
    viz.plot_perm_importance_heatmap(all_perm_imp_meta, META_FEATURE_NAMES, VIZ_DIR)

    # Gradient importance (normalised to % of total — comparable across reps)
    grad_compare = viz.plot_grad_importance_all(all_grad_imp_meta, META_FEATURE_NAMES, VIZ_DIR)

    # ── Re-render per-rep importance bar charts with shared xlim ──────────────
    # Overwrites the in-loop auto-scaled versions with a shared x-axis so the
    # three per-rep figures are directly visually comparable in the thesis.
    perm_x_hi = max((df['mean_auc_drop'] + df['std_auc_drop']).max()
                    for df in all_perm_imp_meta.values())
    perm_x_lo = min(df['mean_auc_drop'].min() for df in all_perm_imp_meta.values())
    perm_xlim = (min(perm_x_lo, 0) - 0.005, perm_x_hi + 0.005)

    def _rel(arr):
        s = float(np.sum(arr))
        return (arr / s * 100) if s > 0 else arr
    grad_x_hi = max(_rel(v).max() for v in all_grad_imp_meta.values())
    grad_xlim = (0, grad_x_hi * 1.05)

    print(f"{ts()} Shared perm xlim = ({perm_xlim[0]:.3f}, {perm_xlim[1]:.3f})")
    print(f"{ts()} Shared grad xlim = (0, {grad_xlim[1]:.2f}%)")
    for rn in all_perm_imp_meta:
        viz.plot_perm_importance_single(all_perm_imp_meta[rn], rn, VIZ_DIR, xlim=perm_xlim)
    for rn in all_grad_imp_meta:
        viz.plot_grad_importance_single(all_grad_imp_meta[rn], META_FEATURE_NAMES,
                                         rn, VIZ_DIR, xlim=grad_xlim)

    # Grouped permutation importance (10 original categorical variables)
    if all_grouped_perm_meta:
        group_compare = viz.plot_grouped_perm_importance_all(all_grouped_perm_meta, VIZ_DIR)

    # Attribution heatmaps with shared colour scale across all representations
    if all_attr_meta:
        shared_vmax = max(
            max(np.percentile(np.abs(attr), 99), 1e-8)
            for attr in all_attr_meta.values()
        )
        print(f"{ts()} Shared attribution heatmap vmax = {shared_vmax:.4f}")
        for rn in all_attr_meta:
            viz.plot_attribution_heatmap(
                all_attr_meta[rn], all_perm_imp_meta[rn],
                META_FEATURE_NAMES, all_y_test_meta[rn],
                rn, VIZ_DIR, vmax=shared_vmax)

    mode   = 'a' if os.path.exists(RESULTS_XLSX) else 'w'
    kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
    with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode=mode, **kwargs) as writer:
        perm_compare.reset_index().to_excel(writer, sheet_name='perm_all_reps_meta',   index=False)
        grad_compare.reset_index().to_excel(writer, sheet_name='grad_all_reps_meta',   index=False)
        if all_grouped_perm_meta:
            group_compare.reset_index().to_excel(writer, sheet_name='perm_grouped_meta', index=False)
    print(f"{ts()} Cross-rep importance sheets -> {RESULTS_XLSX}")

viz.plot_metadata_contribution(meta_summary_df, io_summary_df, REPRESENTATIONS, VIZ_DIR)

print(f"\n{ts()} Pipeline complete.")
print(f"{ts()} Figures  -> {os.path.abspath(VIZ_DIR)}")
print(f"{ts()} Results  -> {os.path.abspath(RESULTS_XLSX)}")
