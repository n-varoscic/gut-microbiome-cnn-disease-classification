"""
Three-phase training pipeline.

Phase 1  run_grid_search                                — selects best hyperparameters via k-fold CV
Phase 2  run_kfold_cv                                   — runs 5-fold CV with best hyperparameters
Phase 3  run_final_model_image / run_final_model_meta   - train on full training data → evaluate on held-out test set

run_grid_search and run_kfold_cv are shared by both models:
  pass M_train for the metadata model; omit (M_train=None) for image-only.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, fbeta_score,
    precision_score, recall_score, confusion_matrix,
)
import tensorflow as tf
from tensorflow import keras

from config import BASE_SEED, PARAM_GRID, GS_FOLDS, CV_FOLDS, EPOCHS, PATIENCE, THRESHOLD_BETA
from .models import build_image_only_model, build_metadata_model
from .training import train_and_evaluate_image, train_and_evaluate_meta


# ── Phase 1: Grid search ──────────────────────────────────────────────────────

def run_grid_search(X_train, y_train, img_shape, rep_name, M_train=None):
    """Phase 1: grid search over PARAM_GRID using GS_FOLDS-fold stratified CV.

    Selects the combination with the highest mean validation AUC.

    Args:
        X_train  : (N, H, W, C) training images
        y_train  : (N,) binary labels
        img_shape: tuple — passed to model builders
        rep_name : str   — representation name for logging
        M_train  : array or None — metadata; None = image-only mode

    Returns:
        best_params : dict  — lr, batch_size, conv_drop, dense_drop
        gs_df       : DataFrame — all grid search results sorted by mean_auc (desc)
    """
    mode = 'metadata' if M_train is not None else 'image only'
    print(f"\n{'='*62}")
    print(f"  Phase 1 — Grid Search  [{rep_name}  |  {mode}]")
    print(f"  {len(PARAM_GRID)} combos × {GS_FOLDS} folds = {len(PARAM_GRID)*GS_FOLDS} trains")
    print(f"{'='*62}")

    skf     = StratifiedKFold(n_splits=GS_FOLDS, shuffle=True, random_state=BASE_SEED)
    gs_rows = []

    for ci, params in enumerate(PARAM_GRID):
        fold_aucs = []
        for fi, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            seed = BASE_SEED + ci * 100 + fi
            if M_train is not None:
                metrics, *_ = train_and_evaluate_meta(
                    X_train[tr_idx], M_train[tr_idx], y_train[tr_idx],
                    X_train[val_idx], M_train[val_idx], y_train[val_idx],
                    params, seed, img_shape)
            else:
                metrics, *_ = train_and_evaluate_image(
                    X_train[tr_idx], y_train[tr_idx],
                    X_train[val_idx], y_train[val_idx],
                    params, seed, img_shape)
            fold_aucs.append(metrics['auc'])
            print(f"  Combo {ci+1:2d}/{len(PARAM_GRID)} | Fold {fi+1}/{GS_FOLDS} | "
                  f"lr={params['lr']:.0e}  bs={params['batch_size']:2d}  "
                  f"cd={params['conv_drop']} -> AUC={metrics['auc']:.4f}")

        gs_rows.append({
            'combo'     : ci + 1,
            'lr'        : params['lr'],
            'batch_size': params['batch_size'],
            'conv_drop' : params['conv_drop'],
            'dense_drop': params['dense_drop'],
            'mean_auc'  : np.mean(fold_aucs),
            'std_auc'   : np.std(fold_aucs),
        })

    gs_df    = pd.DataFrame(gs_rows).sort_values('mean_auc', ascending=False)
    best_row = gs_df.iloc[0]
    best_params = {
        'lr'        : best_row['lr'],
        'batch_size': int(best_row['batch_size']),
        'conv_drop' : best_row['conv_drop'],
        'dense_drop': best_row['dense_drop'],
    }
    print(f"\n  Best -> lr={best_params['lr']:.0e}  batch={best_params['batch_size']}  "
          f"conv_drop={best_params['conv_drop']}  "
          f"mean AUC={best_row['mean_auc']:.4f} ± {best_row['std_auc']:.4f}")
    return best_params, gs_df


# ── Phase 2: K-fold CV ────────────────────────────────────────────────────────

def run_kfold_cv(X_train, y_train, img_shape, best_params, rep_name, M_train=None):
    """Phase 2: CV_FOLDS-fold stratified CV with best_params for reporting.

    Uses independent seed offsets so image-only and metadata models are evaluated
    on independently sampled folds:
      metadata   seed offset: +200
      image-only seed offset: +400

    Args:
        X_train    : training images
        y_train    : binary labels
        img_shape  : tuple
        best_params: dict from run_grid_search
        rep_name   : str — for logging
        M_train    : array or None — metadata; None = image-only mode

    Returns:
        cv_metrics   : list[dict]    — one metrics dict per fold
        cv_histories : list[History] — one Keras History per fold
    """
    mode        = 'metadata' if M_train is not None else 'image only'
    seed_offset = 200 if M_train is not None else 400
    print(f"\n{'='*62}")
    print(f"  Phase 2 — {CV_FOLDS}-fold CV  [{rep_name}  |  {mode}]")
    print(f"  Params: {best_params}")
    print(f"{'='*62}")

    skf          = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                   random_state=BASE_SEED + 1)
    cv_metrics   = []
    cv_histories = []

    for fi, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        seed = BASE_SEED + seed_offset + fi
        if M_train is not None:
            metrics, _, _, history = train_and_evaluate_meta(
                X_train[tr_idx], M_train[tr_idx], y_train[tr_idx],
                X_train[val_idx], M_train[val_idx], y_train[val_idx],
                best_params, seed, img_shape, verbose=0)
        else:
            metrics, _, _, history = train_and_evaluate_image(
                X_train[tr_idx], y_train[tr_idx],
                X_train[val_idx], y_train[val_idx],
                best_params, seed, img_shape, verbose=0)
        cv_metrics.append(metrics)
        cv_histories.append(history)
        print(f"  Fold {fi+1}/{CV_FOLDS} -> AUC={metrics['auc']:.4f}  "
              f"F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}  "
              f"Rec={metrics['recall']:.4f}  (ep={metrics['epochs_run']})")

    aucs = [m['auc'] for m in cv_metrics]
    f1s  = [m['f1']  for m in cv_metrics]
    print(f"\n  CV AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  CV F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    return cv_metrics, cv_histories


# ── Phase 3: Final model ──────────────────────────────────────────────────────

def run_final_model_image(X_train, y_train, X_test, y_test,
                          img_shape, best_params, model_save_path, rep_name):
    """Phase 3 (image-only): train on full training data, evaluate on test set.

    10% of training data is held out for EarlyStopping. The F-beta threshold is
    optimised on this held-out split and then applied to the test set — the test
    set is never used during threshold selection.

    Args:
        X_train, y_train : full training set
        X_test,  y_test  : held-out test set
        img_shape        : tuple
        best_params      : dict from run_grid_search
        model_save_path  : str — path to save the trained .keras model
        rep_name         : str — for logging

    Returns:
        test_metrics : dict       — AUC, F1, precision, recall, threshold, TP/TN/FP/FN
        test_probs   : (N_test,)  — raw probabilities (for ROC / confusion matrix plots)
    """
    print(f"\n{'='*62}")
    print(f"  Phase 3 — Final model → test set  [{rep_name}  |  image only]")
    print(f"{'='*62}")

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(X_test.shape)

    n       = len(y_train)
    n_ibd   = int(y_train.sum())
    n_hel   = n - n_ibd
    cw      = {0: n / (2 * n_hel), 1: n / (2 * n_ibd)}

    # Hold out 10% of training data for EarlyStopping validation
    rng     = np.random.default_rng(BASE_SEED + 998)
    val_idx = rng.choice(n, size=max(1, int(0.1 * n)), replace=False)
    tr_idx  = np.setdiff1d(np.arange(n), val_idx)

    tf.random.set_seed(BASE_SEED + 998)
    np.random.seed(BASE_SEED + 998)
    model = build_image_only_model(img_shape,
                                   lr=best_params['lr'],
                                   conv_drop=best_params['conv_drop'],
                                   dense_drop=best_params['dense_drop'])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=PATIENCE, mode='max',
            restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=0),
    ]

    model.fit(
        {'taxa_image': X_tr_s[tr_idx]}, y_train[tr_idx],
        validation_data=({'taxa_image': X_tr_s[val_idx]}, y_train[val_idx]),
        epochs=EPOCHS,
        batch_size=best_params['batch_size'],
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    # Find F-beta optimal threshold on held-out split; apply to test set
    val_probs    = model.predict({'taxa_image': X_tr_s[val_idx]}, verbose=0).ravel()
    best_thr, best_score = 0.5, 0.0
    for thr in np.arange(0.20, 0.81, 0.01):
        score = fbeta_score(y_train[val_idx], (val_probs >= thr).astype(int),
                            beta=THRESHOLD_BETA, zero_division=0)
        if score > best_score:
            best_score, best_thr = score, thr

    test_probs     = model.predict({'taxa_image': X_test_s}, verbose=0).ravel()
    preds          = (test_probs >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    test_metrics = {
        'auc'      : roc_auc_score(y_test, test_probs),
        'f1'       : f1_score(y_test, preds, zero_division=0),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall'   : recall_score(y_test, preds, zero_division=0),
        'threshold': best_thr,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    print(f"  Test  AUC={test_metrics['auc']:.4f}  F1={test_metrics['f1']:.4f}  "
          f"Prec={test_metrics['precision']:.4f}  Rec={test_metrics['recall']:.4f}")
    print(f"  Conf  TP={tp}  TN={tn}  FP={fp}  FN={fn}  (thr={best_thr:.2f})")
    model.save(model_save_path)
    print(f"  Saved -> {model_save_path}")
    return test_metrics, test_probs


def run_final_model_meta(X_train, M_train, y_train,
                         X_test,  M_test,  y_test,
                         img_shape, best_params, model_save_path, rep_name):
    """Phase 3 (metadata): train on full training data, evaluate on test set.

    Same structure as run_final_model_image but uses both image and metadata inputs.
    Returns the model, scaler, and raw test probabilities so Phase 4
    (feature importance) can use them directly.

    Args:
        X_train, M_train, y_train : full training set
        X_test,  M_test,  y_test  : held-out test set
        img_shape                 : tuple
        best_params               : dict from run_grid_search
        model_save_path           : str
        rep_name                  : str — for logging

    Returns:
        test_metrics : dict
        model        : trained Keras model    (needed for Phase 4)
        scaler       : fitted image StandardScaler    (needed for Phase 4)
        meta_scaler  : fitted metadata StandardScaler (needed for Phase 4)
        test_probs   : (N_test,) raw probabilities    (needed for Phase 4)
    """
    print(f"\n{'='*62}")
    print(f"  Phase 3 — Final model → test set  [{rep_name}  |  metadata]")
    print(f"{'='*62}")

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(X_test.shape)

    # Standardize metadata — fit on full training set only (test never seen).
    # Must be returned so Phase 4 can scale M_test identically before attribution.
    meta_scaler = StandardScaler()
    M_train_s   = meta_scaler.fit_transform(M_train)
    M_test_s    = meta_scaler.transform(M_test)

    n       = len(y_train)
    n_ibd   = int(y_train.sum())
    n_hel   = n - n_ibd
    cw      = {0: n / (2 * n_hel), 1: n / (2 * n_ibd)}

    rng     = np.random.default_rng(BASE_SEED + 999)
    val_idx = rng.choice(n, size=max(1, int(0.1 * n)), replace=False)
    tr_idx  = np.setdiff1d(np.arange(n), val_idx)

    tf.random.set_seed(BASE_SEED + 999)
    np.random.seed(BASE_SEED + 999)
    model = build_metadata_model(img_shape, M_train.shape[1],
                                 lr=best_params['lr'],
                                 conv_drop=best_params['conv_drop'],
                                 dense_drop=best_params['dense_drop'])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=PATIENCE, mode='max',
            restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=0),
    ]

    model.fit(
        {'taxa_image': X_tr_s[tr_idx], 'metadata': M_train_s[tr_idx]},
        y_train[tr_idx],
        validation_data=(
            {'taxa_image': X_tr_s[val_idx], 'metadata': M_train_s[val_idx]},
            y_train[val_idx]),
        epochs=EPOCHS,
        batch_size=best_params['batch_size'],
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    val_probs    = model.predict(
        {'taxa_image': X_tr_s[val_idx], 'metadata': M_train_s[val_idx]},
        verbose=0).ravel()
    best_thr, best_score = 0.5, 0.0
    for thr in np.arange(0.20, 0.81, 0.01):
        score = fbeta_score(y_train[val_idx], (val_probs >= thr).astype(int),
                            beta=THRESHOLD_BETA, zero_division=0)
        if score > best_score:
            best_score, best_thr = score, thr

    test_probs     = model.predict(
        {'taxa_image': X_test_s, 'metadata': M_test_s}, verbose=0).ravel()
    preds          = (test_probs >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    test_metrics = {
        'auc'      : roc_auc_score(y_test, test_probs),
        'f1'       : f1_score(y_test, preds, zero_division=0),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall'   : recall_score(y_test, preds, zero_division=0),
        'threshold': best_thr,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    print(f"  Test  AUC={test_metrics['auc']:.4f}  F1={test_metrics['f1']:.4f}  "
          f"Prec={test_metrics['precision']:.4f}  Rec={test_metrics['recall']:.4f}")
    print(f"  Conf  TP={tp}  TN={tn}  FP={fp}  FN={fn}  (thr={best_thr:.2f})")
    model.save(model_save_path)
    print(f"  Saved -> {model_save_path}")
    return test_metrics, model, scaler, meta_scaler, test_probs
