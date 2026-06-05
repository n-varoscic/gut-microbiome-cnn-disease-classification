"""
run_only_metadata_pipeline.py — Metadata-only (host-metadata) ablation.

Trains classifiers on the host metadata ALONE (no taxa images), as requested by
the supervisor, to test the three-scenario question of whether microbial profiles
add predictive value on top of host metadata.

Three models, all run on EXACTLY the same data, splits, preprocessing, class
weights, threshold strategy and metrics as the CNN pipeline (run_pipeline.py),
so their numbers drop straight into the same comparison table:

  1. MLP    — the metadata branch of the dual-input CNN with the image branch
              removed (Dense 128→64→32 → Dense 64 → sigmoid). This is the
              architecture-controlled comparison vs the metadata CNN: same
              architecture family, only the inputs differ, so any gap is
              attributable to modality (metadata vs metadata+image) rather than
              to trees-vs-CNN.
  2. RF     — RandomForest, the classical tabular baseline the supervisor asked
              for ("trees").
  3. LogReg — LogisticRegression, the classical linear baseline ("regression").

Protocol mirrored from src/ exactly:
  • StandardScaler on metadata, fit on the training split ONLY (no leakage).
  • Inverse-frequency class weights  ==  sklearn class_weight='balanced'.
  • F-beta(beta=2) threshold sweep over 0.20–0.80 on the validation data.
  • Three phases: (1) grid search, GS_FOLDS-fold CV, select on mean val AUC;
                  (2) CV_FOLDS-fold CV for mean ± std reporting;
                  (3) final model trained on 90% of train, threshold tuned on the
                      held-out 10%, evaluated on the untouched test set.

Metadata is identical across the three taxa representations, so each model is
run ONCE (the single result is compared against each metadata-CNN representation).

Accuracy is reported alongside AUC/F1/precision/recall (the supervisor explicitly
asked to compare accuracies despite the class imbalance).

NON-DESTRUCTIVE — this script only ever CREATES new files:
  results/results_mo.xlsx
  results/visualizations_mo/*.png
  saved_models/best_mlp_mo.keras
  saved_models/rf_mo.joblib
  saved_models/logreg_mo.joblib
It never modifies, renames or deletes any existing notebook, data file, model,
figure or the existing results_all_representations.xlsx workbook.

Usage (local or SLURM / Habrok):
    python run_only_metadata_pipeline.py
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
    BASE_SEED, METADATA_PATH, RESULTS_DIR, MODELS_DIR,
    PARAM_GRID, GS_FOLDS, CV_FOLDS, EPOCHS, LOSS_FN,
    REPRESENTATIONS,
    META_FEATURE_NAMES, META_FEATURE_GROUPS, IMPORTANCE_N_REPEATS,
)
from src.utils import load_representation, save_rep_to_excel
# Re-use the CNN pipeline's helpers verbatim (these are byte-identical to the
# definitions that used to live here, so importing them removes the duplication
# while guaranteeing the metadata-only models use EXACTLY the same class weights,
# callbacks and threshold sweep as the CNN).
from src.training import _class_weights, _callbacks, _best_threshold

np.random.seed(BASE_SEED)
tf.random.set_seed(BASE_SEED)


# ── New (non-destructive) output locations ────────────────────────────────────
RESULTS_XLSX_MO = os.path.join(str(RESULTS_DIR), 'results_mo.xlsx')
VIZ_MO_DIR           = os.path.join(str(RESULTS_DIR), 'visualizations_mo')
os.makedirs(VIZ_MO_DIR, exist_ok=True)

MLP_SAVE_PATH    = os.path.join(str(MODELS_DIR), 'best_mlp_mo.keras')
RF_SAVE_PATH     = os.path.join(str(MODELS_DIR), 'rf_mo.joblib')
LOGREG_SAVE_PATH = os.path.join(str(MODELS_DIR), 'logreg_mo.joblib')


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
    with the image branch and fusion removed. Keeping this byte-for-byte aligned
    with the CNN's metadata branch is what makes the comparison architecture-
    controlled."""
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
# permutation_importance_meta / grouped_permutation_importance_meta, so the RF
# importance table can be placed directly next to the CNN's. Permutation runs on
# the standardized test metadata the model was trained on (matches the CNN, which
# permutes its standardized M_test). Native Gini importance is added as the RF
# analogue of the CNN's gradient (Input×Gradient) importance.

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
# Visualisations (saved to the new metadata-only figure folder)
# =============================================================================

def plot_roc(test_probs, y_test, name, file_tag):
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    auc = roc_auc_score(y_test, test_probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title(f'ROC — {name} (metadata only)')
    plt.legend(loc='lower right'); plt.tight_layout()
    out = os.path.join(VIZ_MO_DIR, f'roc_{file_tag}.png')
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
    plt.title(f'Confusion — {name}\n(thr={test_metrics["threshold"]:.2f}, metadata only)')
    plt.tight_layout()
    out = os.path.join(VIZ_MO_DIR, f'confusion_{file_tag}.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure -> {out}")


def plot_perm_importance(perm_df, name, file_tag):
    df = perm_df.sort_values('mean_auc_drop', ascending=True)
    plt.figure(figsize=(7, 6))
    plt.barh(df['feature'], df['mean_auc_drop'],
             xerr=df['std_auc_drop'], color='steelblue', error_kw={'elinewidth': 0.8})
    plt.axvline(0, color='grey', linewidth=0.8)
    plt.xlabel('Mean AUC drop when permuted')
    plt.title(f'Permutation importance — {name} (metadata only)')
    plt.tight_layout()
    out = os.path.join(VIZ_MO_DIR, f'perm_importance_{file_tag}.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure -> {out}")


def plot_grouped_perm_importance(gperm_df, name, file_tag):
    df = gperm_df.sort_values('mean_auc_drop', ascending=True)
    plt.figure(figsize=(7, 5))
    plt.barh(df['group'], df['mean_auc_drop'],
             xerr=df['std_auc_drop'], color='seagreen', error_kw={'elinewidth': 0.8})
    plt.axvline(0, color='grey', linewidth=0.8)
    plt.xlabel('Mean AUC drop when permuted (group)')
    plt.title(f'Grouped permutation importance — {name} (metadata only)')
    plt.tight_layout()
    out = os.path.join(VIZ_MO_DIR, f'perm_importance_grouped_{file_tag}.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure -> {out}")


def plot_gini_importance(gini_df, name, file_tag):
    df = gini_df.sort_values('gini_importance', ascending=True)
    plt.figure(figsize=(7, 6))
    plt.barh(df['feature'], df['gini_importance'], color='indianred')
    plt.xlabel('Gini (impurity-decrease) importance')
    plt.title(f'Native feature importance — {name} (metadata only)')
    plt.tight_layout()
    out = os.path.join(VIZ_MO_DIR, f'gini_importance_{file_tag}.png')
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
    plt.title('Metadata-only models — test-set performance')
    plt.legend(); plt.tight_layout()
    out = os.path.join(VIZ_MO_DIR, 'performance_mo.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure -> {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"{ts()} TF  : {tf.__version__}")
    print(f"{ts()} GPU : {tf.config.list_physical_devices('GPU')}")

    # ── Load data ─────────────────────────────────────────────────────────────
    meta_npz = np.load(METADATA_PATH, allow_pickle=True)
    M_train  = meta_npz['X_meta_train'].astype('float32')
    M_test   = meta_npz['X_meta_test'].astype('float32')

    # Labels are identical across taxa representations (same 80/20 split); load
    # them from the first representation's npz (images are discarded here).
    _, _, y_train, y_test = load_representation(REPRESENTATIONS[0][1])

    assert M_train.shape[1] == len(META_FEATURE_NAMES), (
        f"Metadata width {M_train.shape[1]} != {len(META_FEATURE_NAMES)} feature names")

    print(f"{ts()} Metadata  train: {M_train.shape}  test: {M_test.shape}")
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
    print(f"\n{ts()} {'#'*58}\n{ts()} #  MLP  —  metadata only\n{ts()} {'#'*58}")
    mlp_best, mlp_gs = run_grid_search('MLP', mlp_grid, train_eval_mlp, M_train, y_train)
    mlp_cv           = run_kfold_cv('MLP', mlp_best, train_eval_mlp, M_train, y_train, seed_offset=600)
    mlp_test, mlp_probs = final_mlp(mlp_best, M_train, y_train, M_test, y_test, MLP_SAVE_PATH)
    save_rep_to_excel('mlp_mo', mlp_gs, mlp_cv, mlp_test, results_path=RESULTS_XLSX_MO)
    plot_roc(mlp_probs, y_test, 'MLP', 'mlp_mo')
    plot_confusion(mlp_test, 'MLP', 'mlp_mo')
    summary_rows.append(summary_row('MLP', mlp_best, mlp_cv, mlp_test))
    tf.keras.backend.clear_session(); gc.collect()

    # ── Model 2: Random Forest ─────────────────────────────────────────────────
    print(f"\n{ts()} {'#'*58}\n{ts()} #  Random Forest  —  metadata only\n{ts()} {'#'*58}")
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
        rf_predict, M_test_s_rf, y_test, META_FEATURE_NAMES)
    print(f"    Permutation: baseline AUC={rf_base:.4f}, {len(META_FEATURE_NAMES)} features "
          f"× {IMPORTANCE_N_REPEATS} repeats")
    rf_gperm, _ = grouped_permutation_importance_tabular(
        rf_predict, M_test_s_rf, y_test, META_FEATURE_GROUPS, META_FEATURE_NAMES)
    rf_gini = (pd.DataFrame({'feature': META_FEATURE_NAMES,
                             'gini_importance': rf_clf.feature_importances_})
               .sort_values('gini_importance', ascending=False))
    print(f"    Top permutation feature: {rf_perm.iloc[0]['feature']} "
          f"(AUC drop {rf_perm.iloc[0]['mean_auc_drop']:.4f}) | "
          f"Top Gini feature: {rf_gini.iloc[0]['feature']}")

    # Human-readable sheet: grid + CV + test + PERMUTATION + GROUPED PERMUTATION
    # (correct section labels reused from save_rep_to_excel; Gini goes to its own
    # sheet since that section title is CNN-gradient-specific).
    save_rep_to_excel('rf_mo', rf_gs, rf_cv, rf_test,
                      perm_df=rf_perm, grouped_perm_df=rf_gperm,
                      results_path=RESULTS_XLSX_MO)
    # Machine-readable importance sheets, named to mirror the CNN's perm_/gperm_ sheets.
    _mode = 'a' if os.path.exists(RESULTS_XLSX_MO) else 'w'
    _kw   = {'if_sheet_exists': 'replace'} if _mode == 'a' else {}
    with pd.ExcelWriter(RESULTS_XLSX_MO, engine='openpyxl', mode=_mode, **_kw) as writer:
        rf_perm.to_excel( writer, sheet_name='perm_rf_mo',  index=False)
        rf_gperm.to_excel(writer, sheet_name='gperm_rf_mo', index=False)
        rf_gini.to_excel( writer, sheet_name='gini_rf_mo',  index=False)

    plot_roc(rf_probs, y_test, 'Random Forest', 'rf_mo')
    plot_confusion(rf_test, 'Random Forest', 'rf_mo')
    plot_perm_importance(rf_perm, 'Random Forest', 'rf_mo')
    plot_grouped_perm_importance(rf_gperm, 'Random Forest', 'rf_mo')
    plot_gini_importance(rf_gini, 'Random Forest', 'rf_mo')
    summary_rows.append(summary_row('RandomForest', rf_best, rf_cv, rf_test))

    # ── Model 3: Logistic Regression ───────────────────────────────────────────
    print(f"\n{ts()} {'#'*58}\n{ts()} #  Logistic Regression  —  metadata only\n{ts()} {'#'*58}")
    lr_eval        = _sklearn_train_eval(make_logreg)
    lr_best, lr_gs = run_grid_search('LogisticRegression', logreg_grid, lr_eval, M_train, y_train)
    lr_cv          = run_kfold_cv('LogisticRegression', lr_best, lr_eval, M_train, y_train, seed_offset=0)
    lr_test, lr_probs, _, _ = final_sklearn(make_logreg, lr_best, M_train, y_train, M_test, y_test,
                                            LOGREG_SAVE_PATH, 'LogisticRegression')
    save_rep_to_excel('logreg_mo', lr_gs, lr_cv, lr_test, results_path=RESULTS_XLSX_MO)
    plot_roc(lr_probs, y_test, 'Logistic Regression', 'logreg_mo')
    plot_confusion(lr_test, 'Logistic Regression', 'logreg_mo')
    summary_rows.append(summary_row('LogisticRegression', lr_best, lr_cv, lr_test))

    # ── Summary sheet + comparison figure ──────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    mode   = 'a' if os.path.exists(RESULTS_XLSX_MO) else 'w'
    kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
    with pd.ExcelWriter(RESULTS_XLSX_MO, engine='openpyxl', mode=mode, **kwargs) as writer:
        summary_df.to_excel(writer, sheet_name='summary_mo', index=False)

    plot_model_comparison(summary_df)

    print(f"\n{ts()} {'='*58}")
    print(f"{ts()}  Metadata-only test-set summary")
    print(f"{ts()} {'='*58}")
    for _, r in summary_df.iterrows():
        print(f"{ts()}  {r['model']:<20s}  AUC={r['test_auc']:.4f}  "
              f"Acc={r['test_accuracy']:.4f}  F1={r['test_f1']:.4f}  "
              f"Prec={r['test_precision']:.4f}  Rec={r['test_recall']:.4f}")

    print(f"\n{ts()} Pipeline complete.")
    print(f"{ts()} Results -> {os.path.abspath(RESULTS_XLSX_MO)}")
    print(f"{ts()} Figures -> {os.path.abspath(VIZ_MO_DIR)}")


if __name__ == '__main__':
    main()
