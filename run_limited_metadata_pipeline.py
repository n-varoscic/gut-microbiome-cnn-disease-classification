"""
run_limited_metadata_pipeline.py — Limited-metadata (leakage-controlled) ablation.

This is the metadata-only ablation (see run_only_metadata_pipeline.py) re-run on a
REDUCED metadata feature set, to control for label leakage / circularity.

Why this exists
---------------
The "healthy" label in this dataset is DEFINED by a rule of the form:
    healthy == TRUE  iff  20 ≤ age ≤ 69  AND  18.5 ≤ BMI ≤ 30
                          AND no antibiotic use in > 1 year
                          AND no IBD diagnosis  AND no diabetes diagnosis
Three of those defining criteria are present as predictors in the encoded
metadata: `age_corrected`, `bmi_cat`, and `antibiotic_history`. Because every
healthy sample is hard-constrained on these fields while IBD samples are not, the
models can recover part of the label deterministically from them — target leakage,
not biology. (Indeed, in the full metadata-only run these three are the top-3
features by both permutation and Gini importance.)

This script removes those three covariates and re-runs EVERY model that uses
metadata on the remaining 17 metadata dimensions, so we can see how much of the
metadata-based performance survives once the circular features are gone. It is
intended as a SENSITIVITY / ROBUSTNESS analysis reported ALONGSIDE the
full-metadata results, not as a replacement (age/BMI/antibiotic exposure are also
genuine IBD risk factors, so dropping them also discards some legitimate signal —
both views are informative and should be discussed together).

Models re-run here (all of them consume metadata):
  1. MLP            — metadata-only architecture-controlled model
  2. RandomForest   — metadata-only classical baseline (+ feature importance)
  3. LogReg         — metadata-only classical baseline
  4. Metadata CNN   — the dual-input late-fusion CNN (taxa image + metadata),
                      run once per taxa representation (binary / normalized / log),
                      with full Phase-1 grid search, Phase-2 5-fold CV, Phase-3
                      final model + test eval, and Phase-4 feature importance.
                      Re-used verbatim from src/ — only the metadata it is fed is
                      reduced. The image-only CNN is NOT re-run: it never uses
                      metadata, so the feature reduction cannot change it.

Models, protocol, metrics, threshold strategy, class weights, splits, grids and
importance methods are IDENTICAL to the full-metadata runs
(run_only_metadata_pipeline.py + run_pipeline.py) — only the metadata feature set
differs. This keeps the limited and full runs directly comparable.

NON-DESTRUCTIVE — this script only ever CREATES new files:
  results/results_lmo.xlsx
  results/visualizations_lmo/*.png
  saved_models/best_mlp_lmo.keras
  saved_models/rf_lmo.joblib
  saved_models/logreg_lmo.joblib
  saved_models/best_cnn_{rep}_meta_lmo.keras   (one per representation)
It never modifies, renames or deletes any existing file, notebook, data file,
model, figure or Excel workbook, and it does not touch the src/ package.

Usage (local or SLURM / Habrok):
    python run_limited_metadata_pipeline.py
"""

# ── Backend must be set before any other matplotlib/pyplot import ─────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import gc
import time
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, roc_curve,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Read-only imports from the existing project (nothing is added to config/src).
from config import (
    BASE_SEED, METADATA_PATH, RESULTS_DIR, MODELS_DIR, IMG_SHAPE,
    PARAM_GRID, GS_FOLDS, CV_FOLDS, EPOCHS, LOSS_FN,
    REPRESENTATIONS,
    META_FEATURE_NAMES, META_FEATURE_GROUPS, IMPORTANCE_N_REPEATS,
)
from src.utils import load_representation, save_rep_to_excel

# The dual-input metadata CNN is re-used directly from src/ (its builders and
# training take the metadata width + feature names as arguments, so they work
# unchanged on the reduced 17-dim metadata). src/ itself is NOT modified.
# Imported under cnn_* aliases so they don't shadow this file's own
# run_grid_search / run_kfold_cv (which drive the lightweight metadata-only models).
from src.pipeline import (
    run_grid_search   as cnn_grid_search,
    run_kfold_cv      as cnn_kfold_cv,
    run_final_model_meta,
)
from src.importance import run_importance_analysis, grouped_permutation_importance_meta
import src.visualization as viz
# Re-use the CNN pipeline's helpers verbatim (these are byte-identical to the
# definitions that used to live here, so importing them removes the duplication
# while guaranteeing the metadata-only models use EXACTLY the same class weights,
# callbacks and threshold sweep as the CNN).
from src.training import _class_weights, _callbacks, _best_threshold

np.random.seed(BASE_SEED)
tf.random.set_seed(BASE_SEED)


# ── Label-defining covariates to EXCLUDE (the leakage-control change) ──────────
# These three encoded metadata columns are part of the "healthy" definition, so
# they let the models reconstruct the label rather than learn disease signal.
EXCLUDED_FEATURES = ['age_corrected', 'bmi_cat', 'antibiotic_history']


# ── New (non-destructive) output locations ────────────────────────────────────
RESULTS_XLSX_LMO = os.path.join(str(RESULTS_DIR), 'results_lmo.xlsx')
VIZ_LMO_DIR      = os.path.join(str(RESULTS_DIR), 'visualizations_lmo')
os.makedirs(VIZ_LMO_DIR, exist_ok=True)

MLP_SAVE_PATH    = os.path.join(str(MODELS_DIR), 'best_mlp_lmo.keras')
RF_SAVE_PATH     = os.path.join(str(MODELS_DIR), 'rf_lmo.joblib')
LOGREG_SAVE_PATH = os.path.join(str(MODELS_DIR), 'logreg_lmo.joblib')


def ts():
    """Compact timestamp for SLURM log lines."""
    return time.strftime('[%H:%M:%S]')


# =============================================================================
# Shared helpers
# =============================================================================

def _metrics(y_true, probs, threshold, epochs_run=None):
    """Build a metrics dict — same fields as src, plus accuracy."""
    preds          = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    m = {
        'auc'      : roc_auc_score(y_true, probs),
        'accuracy' : accuracy_score(y_true, preds),
        'f1'       : f1_score(y_true, preds, zero_division=0),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall'   : recall_score(y_true, preds, zero_division=0),
        'threshold': threshold,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    if epochs_run is not None:
        m['epochs_run'] = epochs_run
    return m


# =============================================================================
# Model 1 — MLP (metadata branch of the CNN, image branch removed)
# =============================================================================

def build_mlp(meta_dim, lr=1e-4, dense_drop=0.5):
    """Metadata branch of build_metadata_model() + the same classification head,
    with the image branch and fusion removed. meta_dim is taken from the (reduced)
    feature set at call time, so the architecture adapts automatically to the 17
    remaining features."""
    meta_input = keras.Input(shape=(meta_dim,), name='metadata')
    m = layers.Dense(128, activation='relu')(meta_input)
    m = layers.Dropout(0.3)(m)
    m = layers.Dense(64, activation='relu')(m)
    m = layers.Dropout(0.3)(m)
    m = layers.Dense(32, activation='relu')(m)

    out = layers.Dense(64, activation='relu')(m)
    out = layers.Dropout(dense_drop)(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)

    model = Model(inputs=meta_input, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=LOSS_FN,
        metrics=['accuracy', keras.metrics.AUC(name='auc')],
    )
    return model


def train_eval_mlp(params, M_tr, y_tr, M_val, y_val, seed):
    """Train + evaluate the MLP on one train/val split (mirrors
    train_and_evaluate_meta, metadata-only)."""
    scaler  = StandardScaler()
    M_tr_s  = scaler.fit_transform(M_tr)
    M_val_s = scaler.transform(M_val)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = build_mlp(M_tr.shape[1], lr=params['lr'], dense_drop=params['dense_drop'])

    history = model.fit(
        {'metadata': M_tr_s}, y_tr,
        validation_data=({'metadata': M_val_s}, y_val),
        epochs=EPOCHS,
        batch_size=params['batch_size'],
        class_weight=_class_weights(y_tr),
        callbacks=_callbacks(),
        verbose=0,
    )
    val_probs = model.predict({'metadata': M_val_s}, verbose=0).ravel()
    thr       = _best_threshold(y_val, val_probs)
    return _metrics(y_val, val_probs, thr, len(history.history['loss'])), model, scaler


def final_mlp(best_params, M_train, y_train, M_test, y_test, save_path):
    """Phase 3 for the MLP — mirrors run_final_model_meta (90/10 holdout, threshold
    on the held-out 10%, evaluate on test). Scaler fit on full training set."""
    print(f"\n{'='*62}")
    print(f"  Phase 3 — Final MLP → test set")
    print(f"{'='*62}")

    scaler   = StandardScaler()
    M_tr_s   = scaler.fit_transform(M_train)
    M_test_s = scaler.transform(M_test)

    n       = len(y_train)
    rng     = np.random.default_rng(BASE_SEED + 999)
    val_idx = rng.choice(n, size=max(1, int(0.1 * n)), replace=False)
    tr_idx  = np.setdiff1d(np.arange(n), val_idx)

    tf.random.set_seed(BASE_SEED + 999)
    np.random.seed(BASE_SEED + 999)
    model = build_mlp(M_train.shape[1], lr=best_params['lr'],
                      dense_drop=best_params['dense_drop'])

    model.fit(
        {'metadata': M_tr_s[tr_idx]}, y_train[tr_idx],
        validation_data=({'metadata': M_tr_s[val_idx]}, y_train[val_idx]),
        epochs=EPOCHS,
        batch_size=best_params['batch_size'],
        class_weight=_class_weights(y_train),
        callbacks=_callbacks(),
        verbose=1,
    )

    val_probs  = model.predict({'metadata': M_tr_s[val_idx]}, verbose=0).ravel()
    thr        = _best_threshold(y_train[val_idx], val_probs)
    test_probs = model.predict({'metadata': M_test_s}, verbose=0).ravel()

    metrics = _metrics(y_test, test_probs, thr)
    model.save(save_path)
    print(f"  Saved -> {save_path}")
    return metrics, test_probs


# =============================================================================
# Models 2 & 3 — classical baselines (RF, LogReg)
# =============================================================================

def make_rf(params):
    return RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        class_weight='balanced',
        random_state=BASE_SEED,
        n_jobs=-1,
    )


def make_logreg(params):
    return LogisticRegression(
        C=params['C'],
        class_weight='balanced',
        max_iter=1000,
        random_state=BASE_SEED,
    )


def _sklearn_train_eval(make_fn):
    """Return a train_eval(params, M_tr, y_tr, M_val, y_val, seed) closure for a
    scikit-learn estimator. StandardScaler fit on the training split only; the
    same F-beta threshold sweep is applied so F1/precision/recall/accuracy are
    directly comparable to the CNN and MLP. (seed is accepted for a uniform
    signature but classical estimators carry their own random_state.)"""
    def train_eval(params, M_tr, y_tr, M_val, y_val, seed=None):
        scaler  = StandardScaler()
        M_tr_s  = scaler.fit_transform(M_tr)
        M_val_s = scaler.transform(M_val)

        clf = make_fn(params)
        clf.fit(M_tr_s, y_tr)
        val_probs = clf.predict_proba(M_val_s)[:, 1]
        thr       = _best_threshold(y_val, val_probs)
        return _metrics(y_val, val_probs, thr), clf, scaler
    return train_eval


def final_sklearn(make_fn, best_params, M_train, y_train, M_test, y_test,
                  save_path, name):
    """Phase 3 for a classical model — mirrors the CNN's 90/10 protocol so the
    effective training size and threshold-selection data match (train on 90%,
    tune threshold on the held-out 10%, evaluate on test)."""
    print(f"\n{'='*62}")
    print(f"  Phase 3 — Final {name} → test set")
    print(f"{'='*62}")

    scaler   = StandardScaler()
    M_tr_s   = scaler.fit_transform(M_train)
    M_test_s = scaler.transform(M_test)

    n       = len(y_train)
    rng     = np.random.default_rng(BASE_SEED + 999)
    val_idx = rng.choice(n, size=max(1, int(0.1 * n)), replace=False)
    tr_idx  = np.setdiff1d(np.arange(n), val_idx)

    clf = make_fn(best_params)
    clf.fit(M_tr_s[tr_idx], y_train[tr_idx])

    val_probs  = clf.predict_proba(M_tr_s[val_idx])[:, 1]
    thr        = _best_threshold(y_train[val_idx], val_probs)
    test_probs = clf.predict_proba(M_test_s)[:, 1]

    metrics = _metrics(y_test, test_probs, thr)
    joblib.dump({'model': clf, 'scaler': scaler}, save_path)
    print(f"  Saved -> {save_path}")
    return metrics, test_probs, clf, scaler


# =============================================================================
# Feature importance for the classical model (RF) — mirrors src/importance.py
# =============================================================================
# Same metric (mean AUC drop under permutation) and the SAME grouped scheme
# (one-hot columns of a categorical permuted together) as the metadata CNN's
# permutation_importance_meta / grouped_permutation_importance_meta. Here the
# feature_names / feature_groups passed in are the REDUCED set (the three
# label-defining covariates removed), so the indices line up with the reduced
# standardized test matrix. Native Gini importance is added as the RF analogue of
# the CNN's gradient (Input×Gradient) importance.

def permutation_importance_tabular(predict_fn, M_test_s, y_test, feature_names,
                                   n_repeats=IMPORTANCE_N_REPEATS, seed=BASE_SEED):
    """Per-feature permutation importance = mean drop in AUC when a single column
    is shuffled. Returns [feature, mean_auc_drop, std_auc_drop] sorted desc."""
    rng          = np.random.default_rng(seed)
    baseline_auc = roc_auc_score(y_test, predict_fn(M_test_s))
    rows = []
    for i, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            M_perm       = M_test_s.copy()
            M_perm[:, i] = M_perm[rng.permutation(len(M_perm)), i]
            drops.append(baseline_auc - roc_auc_score(y_test, predict_fn(M_perm)))
        rows.append({'feature': name,
                     'mean_auc_drop': np.mean(drops), 'std_auc_drop': np.std(drops)})
    df = pd.DataFrame(rows).sort_values('mean_auc_drop', ascending=False)
    return df, baseline_auc


def grouped_permutation_importance_tabular(predict_fn, M_test_s, y_test,
                                           feature_groups, feature_names,
                                           n_repeats=IMPORTANCE_N_REPEATS, seed=BASE_SEED):
    """Grouped permutation importance — all one-hot columns of a categorical are
    permuted with the SAME row order so every sample stays in exactly one
    category. Returns [group, n_features, mean_auc_drop, std_auc_drop]."""
    rng          = np.random.default_rng(seed)
    baseline_auc = roc_auc_score(y_test, predict_fn(M_test_s))
    rows = []
    for group_name, cols in feature_groups.items():
        col_idxs = [feature_names.index(c) for c in cols if c in feature_names]
        if not col_idxs:
            continue
        drops = []
        for _ in range(n_repeats):
            M_perm   = M_test_s.copy()
            perm_idx = rng.permutation(len(M_perm))
            M_perm[:, col_idxs] = M_perm[perm_idx][:, col_idxs]
            drops.append(baseline_auc - roc_auc_score(y_test, predict_fn(M_perm)))
        rows.append({'group': group_name, 'n_features': len(col_idxs),
                     'mean_auc_drop': np.mean(drops), 'std_auc_drop': np.std(drops)})
    df = pd.DataFrame(rows).sort_values('mean_auc_drop', ascending=False)
    return df, baseline_auc


# =============================================================================
# Generic three-phase driver (Phases 1 & 2 — shared across all models)
# =============================================================================

def run_grid_search(name, grid, train_eval, M_train, y_train):
    """Phase 1 — GS_FOLDS-fold CV over `grid`, select best mean val AUC.
    Same StratifiedKFold seed (BASE_SEED) as src/pipeline.run_grid_search."""
    print(f"\n{'='*62}")
    print(f"  Phase 1 — Grid Search  [{name}]")
    print(f"  {len(grid)} combos × {GS_FOLDS} folds = {len(grid)*GS_FOLDS} fits")
    print(f"{'='*62}")

    skf  = StratifiedKFold(n_splits=GS_FOLDS, shuffle=True, random_state=BASE_SEED)
    rows = []
    for ci, params in enumerate(grid):
        fold_aucs = []
        for fi, (tr_idx, val_idx) in enumerate(skf.split(M_train, y_train)):
            seed = BASE_SEED + ci * 100 + fi
            metrics, *_ = train_eval(params, M_train[tr_idx], y_train[tr_idx],
                                     M_train[val_idx], y_train[val_idx], seed)
            fold_aucs.append(metrics['auc'])
            print(f"  Combo {ci+1:2d}/{len(grid)} | Fold {fi+1}/{GS_FOLDS} | "
                  f"{params} -> AUC={metrics['auc']:.4f}")
        rows.append({'combo': ci + 1, **params,
                     'mean_auc': np.mean(fold_aucs), 'std_auc': np.std(fold_aucs)})

    gs_df       = pd.DataFrame(rows).sort_values('mean_auc', ascending=False).reset_index(drop=True)
    best_combo  = int(gs_df.iloc[0]['combo'])
    best_params = grid[best_combo - 1]
    print(f"\n  Best [{name}] -> {best_params}  "
          f"mean AUC={gs_df.iloc[0]['mean_auc']:.4f} ± {gs_df.iloc[0]['std_auc']:.4f}")
    return best_params, gs_df


def run_kfold_cv(name, best_params, train_eval, M_train, y_train, seed_offset):
    """Phase 2 — CV_FOLDS-fold CV with best_params for mean ± std reporting.
    Same StratifiedKFold seed (BASE_SEED + 1) as src/pipeline.run_kfold_cv."""
    print(f"\n{'='*62}")
    print(f"  Phase 2 — {CV_FOLDS}-fold CV  [{name}]   params: {best_params}")
    print(f"{'='*62}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=BASE_SEED + 1)
    cv_metrics = []
    for fi, (tr_idx, val_idx) in enumerate(skf.split(M_train, y_train)):
        seed = BASE_SEED + seed_offset + fi
        metrics, *_ = train_eval(best_params, M_train[tr_idx], y_train[tr_idx],
                                 M_train[val_idx], y_train[val_idx], seed)
        cv_metrics.append(metrics)
        print(f"  Fold {fi+1}/{CV_FOLDS} -> AUC={metrics['auc']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
              f"Rec={metrics['recall']:.4f}")

    aucs = [m['auc']      for m in cv_metrics]
    accs = [m['accuracy'] for m in cv_metrics]
    print(f"\n  CV AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  CV Acc : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    return cv_metrics


def summary_row(name, best_params, cv_metrics, test_metrics):
    """One row for the cross-model summary sheet."""
    def ms(key):
        vals = [m[key] for m in cv_metrics]
        return np.mean(vals), np.std(vals)
    auc_m, auc_s = ms('auc')
    acc_m, acc_s = ms('accuracy')
    f1_m,  f1_s  = ms('f1')
    rec_m, rec_s = ms('recall')
    return {
        'model'            : name,
        'best_params'      : str(best_params),
        'cv_auc_mean'      : auc_m, 'cv_auc_std'     : auc_s,
        'cv_accuracy_mean' : acc_m, 'cv_accuracy_std': acc_s,
        'cv_f1_mean'       : f1_m,  'cv_f1_std'      : f1_s,
        'cv_recall_mean'   : rec_m, 'cv_recall_std'  : rec_s,
        'test_auc'         : test_metrics['auc'],
        'test_accuracy'    : test_metrics['accuracy'],
        'test_f1'          : test_metrics['f1'],
        'test_precision'   : test_metrics['precision'],
        'test_recall'      : test_metrics['recall'],
        'test_tp': test_metrics['tp'], 'test_tn': test_metrics['tn'],
        'test_fp': test_metrics['fp'], 'test_fn': test_metrics['fn'],
    }


# =============================================================================
# Visualisations (saved to the new limited-metadata figure folder)
# =============================================================================

def plot_roc(test_probs, y_test, name, file_tag):
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    auc = roc_auc_score(y_test, test_probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title(f'ROC — {name} (limited metadata)')
    plt.legend(loc='lower right'); plt.tight_layout()
    out = os.path.join(VIZ_LMO_DIR, f'roc_{file_tag}.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure -> {out}")


def plot_confusion(test_metrics, name, file_tag):
    cm = np.array([[test_metrics['tn'], test_metrics['fp']],
                   [test_metrics['fn'], test_metrics['tp']]])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='black', fontsize=12)
    plt.xticks([0, 1], ['Healthy', 'IBD']); plt.yticks([0, 1], ['Healthy', 'IBD'])
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title(f'Confusion — {name}\n(thr={test_metrics["threshold"]:.2f}, limited metadata)')
    plt.tight_layout()
    out = os.path.join(VIZ_LMO_DIR, f'confusion_{file_tag}.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure -> {out}")


def plot_perm_importance(perm_df, name, file_tag):
    df = perm_df.sort_values('mean_auc_drop', ascending=True)
    plt.figure(figsize=(7, 6))
    plt.barh(df['feature'], df['mean_auc_drop'],
             xerr=df['std_auc_drop'], color='steelblue', error_kw={'elinewidth': 0.8})
    plt.axvline(0, color='grey', linewidth=0.8)
    plt.xlabel('Mean AUC drop when permuted')
    plt.title(f'Permutation importance — {name} (limited metadata)')
    plt.tight_layout()
    out = os.path.join(VIZ_LMO_DIR, f'perm_importance_{file_tag}.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure -> {out}")


def plot_grouped_perm_importance(gperm_df, name, file_tag):
    df = gperm_df.sort_values('mean_auc_drop', ascending=True)
    plt.figure(figsize=(7, 5))
    plt.barh(df['group'], df['mean_auc_drop'],
             xerr=df['std_auc_drop'], color='seagreen', error_kw={'elinewidth': 0.8})
    plt.axvline(0, color='grey', linewidth=0.8)
    plt.xlabel('Mean AUC drop when permuted (group)')
    plt.title(f'Grouped permutation importance — {name} (limited metadata)')
    plt.tight_layout()
    out = os.path.join(VIZ_LMO_DIR, f'perm_importance_grouped_{file_tag}.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure -> {out}")


def plot_gini_importance(gini_df, name, file_tag):
    df = gini_df.sort_values('gini_importance', ascending=True)
    plt.figure(figsize=(7, 6))
    plt.barh(df['feature'], df['gini_importance'], color='indianred')
    plt.xlabel('Gini (impurity-decrease) importance')
    plt.title(f'Native feature importance — {name} (limited metadata)')
    plt.tight_layout()
    out = os.path.join(VIZ_LMO_DIR, f'gini_importance_{file_tag}.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure -> {out}")


def plot_model_comparison(summary_df):
    metrics  = ['test_auc', 'test_accuracy', 'test_f1', 'test_recall']
    labels   = ['AUC', 'Accuracy', 'F1', 'Recall']
    models   = summary_df['model'].tolist()
    x        = np.arange(len(metrics))
    width    = 0.8 / len(models)
    plt.figure(figsize=(9, 5))
    for i, mdl in enumerate(models):
        row  = summary_df[summary_df['model'] == mdl].iloc[0]
        vals = [row[m] for m in metrics]
        plt.bar(x + i * width, vals, width, label=mdl)
    plt.xticks(x + width * (len(models) - 1) / 2, labels)
    plt.ylim(0, 1); plt.ylabel('Score (test set)')
    plt.title('Limited-metadata models — test-set performance')
    plt.legend(); plt.tight_layout()
    out = os.path.join(VIZ_LMO_DIR, 'performance_lmo.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure -> {out}")


# =============================================================================
# Model 4 — Metadata CNN (dual-input late-fusion, re-used from src/)
# =============================================================================
# This mirrors Section 3 + Section 4 (metadata parts) of run_pipeline.py exactly,
# but (a) feeds the reduced 17-dim metadata, (b) writes only new _lmo outputs, and
# (c) saves models to best_cnn_{rep}_meta_lmo.keras so nothing existing is touched.
# The image-only CNN is intentionally omitted: it uses no metadata and so is
# unaffected by the feature reduction.

# Representations already done can be listed here to resume after a crash.
SKIP_REPS_META_LMO = []


def run_metadata_cnn(M_train, M_test, feat_names, feat_groups):
    """Run the dual-input metadata CNN across all taxa representations on the
    reduced metadata, with the full 3-phase protocol + Phase-4 importance."""
    meta_summary_rows     = []
    all_perm_imp_meta     = {}
    all_grad_imp_meta     = {}
    all_attr_meta         = {}
    all_y_test_meta       = {}
    all_grouped_perm_meta = {}

    for rep_name, taxa_path, meta_save_path in REPRESENTATIONS:
        if rep_name in SKIP_REPS_META_LMO:
            print(f"{ts()}   Skipping {rep_name} (in SKIP_REPS_META_LMO)")
            continue

        # best_cnn_{rep}_meta.keras → best_cnn_{rep}_meta_lmo.keras (new file only)
        lmo_save_path = meta_save_path.replace('_meta.keras', '_meta_lmo.keras')
        print(f"\n{ts()} {'#'*58}\n{ts()} #  {rep_name.upper()}  —  metadata CNN (limited)\n{ts()} {'#'*58}")

        X_train, X_test, y_train, y_test = load_representation(taxa_path)
        print(f"{ts()}   Train: {X_train.shape}  Test: {X_test.shape}  "
              f"meta: {M_train.shape[1]}d  "
              f"IBD train: {int(y_train.sum())}/{len(y_train)} ({100*y_train.mean():.1f}%)")

        # Phase 1: grid search (12 combos × GS_FOLDS folds) — src code, reduced meta
        best_params, gs_df = cnn_grid_search(
            X_train, y_train, IMG_SHAPE, rep_name, M_train=M_train)

        # Phase 2: 5-fold CV
        cv_metrics, _ = cnn_kfold_cv(
            X_train, y_train, IMG_SHAPE, best_params, rep_name, M_train=M_train)

        # Phase 3: final model → test set
        test_metrics, final_model, final_scaler, final_meta_scaler, test_probs = \
            run_final_model_meta(
                X_train, M_train, y_train, X_test, M_test, y_test,
                IMG_SHAPE, best_params, lmo_save_path, rep_name)

        # Phase 4a: permutation + gradient importance (scale exactly as trained)
        X_test_s = final_scaler.transform(
            X_test.reshape(len(X_test), -1)).reshape(X_test.shape)
        M_test_s = final_meta_scaler.transform(M_test)
        perm_df, mean_attr, per_sample_attr = run_importance_analysis(
            final_model, X_test_s, M_test_s, y_test,
            test_probs, test_metrics['threshold'], feat_names, rep_name, VIZ_LMO_DIR)

        # Phase 4b: grouped permutation importance (reduced groups)
        grouped_perm_df, _ = grouped_permutation_importance_meta(
            final_model, X_test_s, M_test_s, y_test, feat_groups, feat_names)

        # Phase 4c: per-rep attribution heatmap (auto-scale; re-rendered later)
        viz.plot_attribution_heatmap(
            per_sample_attr, perm_df, feat_names, y_test, rep_name, VIZ_LMO_DIR)

        all_perm_imp_meta[rep_name]     = perm_df
        all_grad_imp_meta[rep_name]     = mean_attr
        all_attr_meta[rep_name]         = per_sample_attr
        all_y_test_meta[rep_name]       = y_test
        all_grouped_perm_meta[rep_name] = grouped_perm_df

        # ── Incremental Excel save (crash-safe) ────────────────────────────────
        grad_df = (pd.DataFrame({'feature': feat_names, 'mean_attr': mean_attr})
                   .sort_values('mean_attr', ascending=False))
        save_rep_to_excel(f'{rep_name}_meta_lmo', gs_df, cv_metrics, test_metrics,
                          perm_df, grad_df, grouped_perm_df,
                          results_path=RESULTS_XLSX_LMO)

        attr_df = pd.DataFrame(per_sample_attr, columns=feat_names)
        attr_df.insert(0, 'y_test', y_test.astype(int))
        _mode = 'a' if os.path.exists(RESULTS_XLSX_LMO) else 'w'
        _kw   = {'if_sheet_exists': 'replace'} if _mode == 'a' else {}
        with pd.ExcelWriter(RESULTS_XLSX_LMO, engine='openpyxl', mode=_mode, **_kw) as writer:
            perm_df.to_excel(        writer, sheet_name=f'perm_{rep_name}_meta_lmo',  index=False)
            grad_df.to_excel(        writer, sheet_name=f'grad_{rep_name}_meta_lmo',  index=False)
            grouped_perm_df.to_excel(writer, sheet_name=f'gperm_{rep_name}_meta_lmo', index=False)
            attr_df.to_excel(        writer, sheet_name=f'attr_{rep_name}_meta_lmo',  index=False)

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

    if not meta_summary_rows:
        print(f"{ts()}   No metadata-CNN representations run (all skipped).")
        return

    # ── Summary sheet ──────────────────────────────────────────────────────────
    meta_summary_df = pd.DataFrame(meta_summary_rows)
    mode   = 'a' if os.path.exists(RESULTS_XLSX_LMO) else 'w'
    kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
    with pd.ExcelWriter(RESULTS_XLSX_LMO, engine='openpyxl', mode=mode, **kwargs) as writer:
        meta_summary_df.to_excel(writer, sheet_name='summary_meta_lmo', index=False)

    # ── Cross-representation figures + sheets (metadata parts of run_pipeline §4)─
    viz.plot_performance_bars(meta_summary_df, VIZ_LMO_DIR, model_label='meta')

    if all_perm_imp_meta:
        perm_compare = viz.plot_perm_importance_all(all_perm_imp_meta, feat_names, VIZ_LMO_DIR)
        viz.plot_perm_importance_heatmap(all_perm_imp_meta, feat_names, VIZ_LMO_DIR)
        grad_compare = viz.plot_grad_importance_all(all_grad_imp_meta, feat_names, VIZ_LMO_DIR)

        # Re-render per-rep importance bars with a shared x-axis for comparability
        perm_x_hi = max((df['mean_auc_drop'] + df['std_auc_drop']).max()
                        for df in all_perm_imp_meta.values())
        perm_x_lo = min(df['mean_auc_drop'].min() for df in all_perm_imp_meta.values())
        perm_xlim = (min(perm_x_lo, 0) - 0.005, perm_x_hi + 0.005)

        def _rel(arr):
            s = float(np.sum(arr))
            return (arr / s * 100) if s > 0 else arr
        grad_x_hi = max(_rel(v).max() for v in all_grad_imp_meta.values())
        grad_xlim = (0, grad_x_hi * 1.05)

        for rn in all_perm_imp_meta:
            viz.plot_perm_importance_single(all_perm_imp_meta[rn], rn, VIZ_LMO_DIR, xlim=perm_xlim)
        for rn in all_grad_imp_meta:
            viz.plot_grad_importance_single(all_grad_imp_meta[rn], feat_names,
                                            rn, VIZ_LMO_DIR, xlim=grad_xlim)

        group_compare = None
        if all_grouped_perm_meta:
            group_compare = viz.plot_grouped_perm_importance_all(all_grouped_perm_meta, VIZ_LMO_DIR)

        if all_attr_meta:
            shared_vmax = max(
                max(np.percentile(np.abs(attr), 99), 1e-8)
                for attr in all_attr_meta.values()
            )
            for rn in all_attr_meta:
                viz.plot_attribution_heatmap(
                    all_attr_meta[rn], all_perm_imp_meta[rn],
                    feat_names, all_y_test_meta[rn], rn, VIZ_LMO_DIR, vmax=shared_vmax)

        mode   = 'a' if os.path.exists(RESULTS_XLSX_LMO) else 'w'
        kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
        with pd.ExcelWriter(RESULTS_XLSX_LMO, engine='openpyxl', mode=mode, **kwargs) as writer:
            perm_compare.reset_index().to_excel(writer, sheet_name='perm_all_reps_meta_lmo', index=False)
            grad_compare.reset_index().to_excel(writer, sheet_name='grad_all_reps_meta_lmo', index=False)
            if group_compare is not None:
                group_compare.reset_index().to_excel(writer, sheet_name='perm_grouped_meta_lmo', index=False)

    print(f"\n{ts()} {'='*58}")
    print(f"{ts()}  Metadata CNN (limited) test-set summary")
    print(f"{ts()} {'='*58}")
    for _, r in meta_summary_df.iterrows():
        print(f"{ts()}  {r['representation']:<12s}  AUC={r['test_auc']:.4f}  "
              f"F1={r['test_f1']:.4f}  Prec={r['test_precision']:.4f}  "
              f"Rec={r['test_recall']:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"{ts()} TF  : {tf.__version__}")
    print(f"{ts()} GPU : {tf.config.list_physical_devices('GPU')}")

    # ── Load data ─────────────────────────────────────────────────────────────
    meta_npz   = np.load(METADATA_PATH, allow_pickle=True)
    M_train_full = meta_npz['X_meta_train'].astype('float32')
    M_test_full  = meta_npz['X_meta_test'].astype('float32')

    # Labels are identical across taxa representations (same 80/20 split); load
    # them from the first representation's npz (images are discarded here).
    _, _, y_train, y_test = load_representation(REPRESENTATIONS[0][1])

    assert M_train_full.shape[1] == len(META_FEATURE_NAMES), (
        f"Metadata width {M_train_full.shape[1]} != {len(META_FEATURE_NAMES)} feature names")

    # ── Drop the label-defining covariates (the leakage-control step) ──────────
    missing = [f for f in EXCLUDED_FEATURES if f not in META_FEATURE_NAMES]
    assert not missing, f"EXCLUDED_FEATURES not found in META_FEATURE_NAMES: {missing}"

    keep_idx   = [i for i, n in enumerate(META_FEATURE_NAMES) if n not in EXCLUDED_FEATURES]
    feat_names = [META_FEATURE_NAMES[i] for i in keep_idx]
    M_train    = M_train_full[:, keep_idx]
    M_test     = M_test_full[:, keep_idx]

    # Reduced grouped map for the RF importance: drop groups whose columns were all
    # removed, and within surviving groups keep only columns that remain.
    feat_groups = {}
    for g, cols in META_FEATURE_GROUPS.items():
        kept = [c for c in cols if c in feat_names]
        if kept:
            feat_groups[g] = kept

    print(f"{ts()} EXCLUDED (label-defining): {EXCLUDED_FEATURES}")
    print(f"{ts()} Metadata  train: {M_train.shape}  test: {M_test.shape}  "
          f"(was {M_train_full.shape[1]} features → {M_train.shape[1]} kept)")
    print(f"{ts()} Kept features: {feat_names}")
    print(f"{ts()} Labels    train IBD: {int(y_train.sum())}/{len(y_train)} "
          f"({100*y_train.mean():.1f}%)   test IBD: {int(y_test.sum())}/{len(y_test)} "
          f"({100*y_test.mean():.1f}%)")

    # ── Hyperparameter grids ───────────────────────────────────────────────────
    # MLP: unique (lr, batch_size) pairs from the CNN's PARAM_GRID so the explored
    # lr/batch space matches; dense_drop fixed at 0.5 as in the CNN. conv_drop does
    # not apply (no convolutional layers).
    mlp_grid, seen = [], set()
    for p in PARAM_GRID:
        key = (p['lr'], p['batch_size'])
        if key in seen:
            continue
        seen.add(key)
        mlp_grid.append({'lr': p['lr'], 'batch_size': p['batch_size'], 'dense_drop': 0.5})

    rf_grid = [{'n_estimators': n, 'max_depth': d}
               for n in (200, 500) for d in (None, 10, 20)]
    logreg_grid = [{'C': c} for c in (0.01, 0.1, 1.0, 10.0)]

    summary_rows = []

    # ── Model 1: MLP ───────────────────────────────────────────────────────────
    print(f"\n{ts()} {'#'*58}\n{ts()} #  MLP  —  limited metadata\n{ts()} {'#'*58}")
    mlp_best, mlp_gs = run_grid_search('MLP', mlp_grid, train_eval_mlp, M_train, y_train)
    mlp_cv           = run_kfold_cv('MLP', mlp_best, train_eval_mlp, M_train, y_train, seed_offset=600)
    mlp_test, mlp_probs = final_mlp(mlp_best, M_train, y_train, M_test, y_test, MLP_SAVE_PATH)
    save_rep_to_excel('mlp_lmo', mlp_gs, mlp_cv, mlp_test, results_path=RESULTS_XLSX_LMO)
    plot_roc(mlp_probs, y_test, 'MLP', 'mlp_lmo')
    plot_confusion(mlp_test, 'MLP', 'mlp_lmo')
    summary_rows.append(summary_row('MLP', mlp_best, mlp_cv, mlp_test))
    tf.keras.backend.clear_session(); gc.collect()

    # ── Model 2: Random Forest ─────────────────────────────────────────────────
    print(f"\n{ts()} {'#'*58}\n{ts()} #  Random Forest  —  limited metadata\n{ts()} {'#'*58}")
    rf_eval        = _sklearn_train_eval(make_rf)
    rf_best, rf_gs = run_grid_search('RandomForest', rf_grid, rf_eval, M_train, y_train)
    rf_cv          = run_kfold_cv('RandomForest', rf_best, rf_eval, M_train, y_train, seed_offset=0)
    rf_test, rf_probs, rf_clf, rf_scaler = final_sklearn(
        make_rf, rf_best, M_train, y_train, M_test, y_test, RF_SAVE_PATH, 'RandomForest')

    # ── Feature importance (mirrors the metadata CNN's Phase 4) ─────────────────
    print(f"\n  [Feature importance — RandomForest]")
    M_test_s_rf  = rf_scaler.transform(M_test)
    rf_predict   = lambda M: rf_clf.predict_proba(M)[:, 1]
    rf_perm, rf_base = permutation_importance_tabular(
        rf_predict, M_test_s_rf, y_test, feat_names)
    print(f"    Permutation: baseline AUC={rf_base:.4f}, {len(feat_names)} features "
          f"× {IMPORTANCE_N_REPEATS} repeats")
    rf_gperm, _ = grouped_permutation_importance_tabular(
        rf_predict, M_test_s_rf, y_test, feat_groups, feat_names)
    rf_gini = (pd.DataFrame({'feature': feat_names,
                             'gini_importance': rf_clf.feature_importances_})
               .sort_values('gini_importance', ascending=False))
    print(f"    Top permutation feature: {rf_perm.iloc[0]['feature']} "
          f"(AUC drop {rf_perm.iloc[0]['mean_auc_drop']:.4f}) | "
          f"Top Gini feature: {rf_gini.iloc[0]['feature']}")

    # Human-readable sheet: grid + CV + test + PERMUTATION + GROUPED PERMUTATION
    # (correct section labels reused from save_rep_to_excel; Gini goes to its own
    # sheet since that section title is CNN-gradient-specific).
    save_rep_to_excel('rf_lmo', rf_gs, rf_cv, rf_test,
                      perm_df=rf_perm, grouped_perm_df=rf_gperm,
                      results_path=RESULTS_XLSX_LMO)
    # Machine-readable importance sheets, named to mirror the CNN's perm_/gperm_ sheets.
    _mode = 'a' if os.path.exists(RESULTS_XLSX_LMO) else 'w'
    _kw   = {'if_sheet_exists': 'replace'} if _mode == 'a' else {}
    with pd.ExcelWriter(RESULTS_XLSX_LMO, engine='openpyxl', mode=_mode, **_kw) as writer:
        rf_perm.to_excel( writer, sheet_name='perm_rf_lmo',  index=False)
        rf_gperm.to_excel(writer, sheet_name='gperm_rf_lmo', index=False)
        rf_gini.to_excel( writer, sheet_name='gini_rf_lmo',  index=False)

    plot_roc(rf_probs, y_test, 'Random Forest', 'rf_lmo')
    plot_confusion(rf_test, 'Random Forest', 'rf_lmo')
    plot_perm_importance(rf_perm, 'Random Forest', 'rf_lmo')
    plot_grouped_perm_importance(rf_gperm, 'Random Forest', 'rf_lmo')
    plot_gini_importance(rf_gini, 'Random Forest', 'rf_lmo')
    summary_rows.append(summary_row('RandomForest', rf_best, rf_cv, rf_test))

    # ── Model 3: Logistic Regression ───────────────────────────────────────────
    print(f"\n{ts()} {'#'*58}\n{ts()} #  Logistic Regression  —  limited metadata\n{ts()} {'#'*58}")
    lr_eval        = _sklearn_train_eval(make_logreg)
    lr_best, lr_gs = run_grid_search('LogisticRegression', logreg_grid, lr_eval, M_train, y_train)
    lr_cv          = run_kfold_cv('LogisticRegression', lr_best, lr_eval, M_train, y_train, seed_offset=0)
    lr_test, lr_probs, _, _ = final_sklearn(make_logreg, lr_best, M_train, y_train, M_test, y_test,
                                            LOGREG_SAVE_PATH, 'LogisticRegression')
    save_rep_to_excel('logreg_lmo', lr_gs, lr_cv, lr_test, results_path=RESULTS_XLSX_LMO)
    plot_roc(lr_probs, y_test, 'Logistic Regression', 'logreg_lmo')
    plot_confusion(lr_test, 'Logistic Regression', 'logreg_lmo')
    summary_rows.append(summary_row('LogisticRegression', lr_best, lr_cv, lr_test))

    # ── Summary sheet + comparison figure ──────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    mode   = 'a' if os.path.exists(RESULTS_XLSX_LMO) else 'w'
    kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
    with pd.ExcelWriter(RESULTS_XLSX_LMO, engine='openpyxl', mode=mode, **kwargs) as writer:
        summary_df.to_excel(writer, sheet_name='summary_lmo', index=False)

    plot_model_comparison(summary_df)

    print(f"\n{ts()} {'='*58}")
    print(f"{ts()}  Limited-metadata test-set summary "
          f"(excluded: {', '.join(EXCLUDED_FEATURES)})")
    print(f"{ts()} {'='*58}")
    for _, r in summary_df.iterrows():
        print(f"{ts()}  {r['model']:<20s}  AUC={r['test_auc']:.4f}  "
              f"Acc={r['test_accuracy']:.4f}  F1={r['test_f1']:.4f}  "
              f"Prec={r['test_precision']:.4f}  Rec={r['test_recall']:.4f}")

    # ── Model 4: Metadata CNN (dual-input) across all representations ───────────
    # The heavy model runs last so the fast metadata-only results are already
    # written to Excel and survive a mid-run crash.
    print(f"\n{ts()} {'#'*58}\n{ts()} #  Metadata CNN (dual-input)  —  limited metadata\n{ts()} {'#'*58}")
    run_metadata_cnn(M_train, M_test, feat_names, feat_groups)

    print(f"\n{ts()} Pipeline complete.")
    print(f"{ts()} Results -> {os.path.abspath(RESULTS_XLSX_LMO)}")
    print(f"{ts()} Figures -> {os.path.abspath(VIZ_LMO_DIR)}")


if __name__ == '__main__':
    main()
