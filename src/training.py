""" 
Training and evaluation functions for the CNN models. 

Both training functions follow the same pattern:
  1. Fit StandardScaler on the training split only (prevents cross-fold leakage).
     The metadata model additionally standardizes its metadata inputs the same
     way (separate scaler, also fit on the training split only).
  2. Compute inverse-frequency class weights (IBD is ~23% of samples).
  3. Build model and fit with EarlyStopping + ReduceLROnPlateau.
  4. Sweep thresholds 0.20–0.80 to find the one maximising F-beta on the
     validation set. THRESHOLD_BETA=2 weights recall 2× more than precision,
     prioritising sensitivity for disease detection.
  5. Return metrics dict, trained model, fitted scaler, and training history.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, fbeta_score,
    precision_score, recall_score, confusion_matrix,
)
import tensorflow as tf
from tensorflow import keras

from config import EPOCHS, PATIENCE, THRESHOLD_BETA
from .models import build_image_only_model, build_metadata_model


# ── Helpers ───────────────────────────────────────────────────────────

def _class_weights(y):
    """Inverse-frequency class weights for binary labels."""
    n     = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    return {0: n / (2 * n_neg), 1: n / (2 * n_pos)}


def _callbacks():
    """Return EarlyStopping + ReduceLROnPlateau callbacks."""
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=PATIENCE, mode='max',
            restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=0),
    ]


def _best_threshold(y_true, probs):
    """Sweep thresholds 0.20–0.80 and return the one maximising F-beta."""
    best_thr, best_score = 0.5, 0.0
    for thr in np.arange(0.20, 0.81, 0.01):
        score = fbeta_score(y_true, (probs >= thr).astype(int),
                            beta=THRESHOLD_BETA, zero_division=0)
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr


def _metrics(y_true, probs, threshold, history):
    """Build a metrics dict from raw predictions, threshold, and history."""
    preds          = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        'auc'       : roc_auc_score(y_true, probs),
        'f1'        : f1_score(y_true, preds, zero_division=0),
        'precision' : precision_score(y_true, preds, zero_division=0),
        'recall'    : recall_score(y_true, preds, zero_division=0),
        'threshold' : threshold,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'epochs_run': len(history.history['loss']),
    }


# ── Training functions ─────────────────────────────────────────────────

def train_and_evaluate_image(X_tr, y_tr, X_val, y_val,
                              params, seed, img_shape, verbose=0):
    """Train and evaluate the image-only CNN on one train/val split.

    Args:
        X_tr, y_tr   : training images and labels
        X_val, y_val : validation images and labels
        params       : dict — lr, batch_size, conv_drop, dense_drop
        seed         : int  — set before model build for reproducibility
        img_shape    : tuple — (H, W, C)
        verbose      : Keras fit verbosity (0 = silent)

    Returns:
        metrics : dict — AUC, F1, precision, recall, threshold, TP/TN/FP/FN,
                         epochs_run
        model   : trained Keras model
        scaler  : fitted StandardScaler
        history : Keras History object
    """
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr.reshape(len(X_tr), -1)).reshape(X_tr.shape)
    X_val_s = scaler.transform(X_val.reshape(len(X_val), -1)).reshape(X_val.shape)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = build_image_only_model(img_shape,
                                   lr=params['lr'],
                                   conv_drop=params['conv_drop'],
                                   dense_drop=params['dense_drop'])

    history = model.fit(
        {'taxa_image': X_tr_s}, y_tr,
        validation_data=({'taxa_image': X_val_s}, y_val),
        epochs=EPOCHS,
        batch_size=params['batch_size'],
        class_weight=_class_weights(y_tr),
        callbacks=_callbacks(),
        verbose=verbose,
    )

    val_probs = model.predict({'taxa_image': X_val_s}, verbose=0).ravel()
    threshold = _best_threshold(y_val, val_probs)
    return _metrics(y_val, val_probs, threshold, history), model, scaler, history


def train_and_evaluate_meta(X_tr, M_tr, y_tr, X_val, M_val, y_val,
                             params, seed, img_shape, verbose=0):
    """Train and evaluate the metadata-fusion CNN on one train/val split.

    Args:
        X_tr, M_tr, y_tr   : training images, metadata, and labels
        X_val, M_val, y_val : validation images, metadata, and labels
        params              : dict — lr, batch_size, conv_drop, dense_drop
        seed                : int  — for reproducibility
        img_shape           : tuple — (H, W, C)
        verbose             : Keras fit verbosity

    Returns:
        metrics : dict — AUC, F1, precision, recall, threshold, TP/TN/FP/FN,
                         epochs_run
        model   : trained Keras model
        scaler  : fitted StandardScaler
        history : Keras History object
    """
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr.reshape(len(X_tr), -1)).reshape(X_tr.shape)
    X_val_s = scaler.transform(X_val.reshape(len(X_val), -1)).reshape(X_val.shape)

    # Standardize metadata too — fit on THIS fold's training split only (no leakage).
    # Puts ordinals, one-hots and the (already-standardized) age on a common scale.
    meta_scaler = StandardScaler()
    M_tr_s      = meta_scaler.fit_transform(M_tr)
    M_val_s     = meta_scaler.transform(M_val)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = build_metadata_model(img_shape, M_tr.shape[1],
                                 lr=params['lr'],
                                 conv_drop=params['conv_drop'],
                                 dense_drop=params['dense_drop'])

    history = model.fit(
        {'taxa_image': X_tr_s, 'metadata': M_tr_s}, y_tr,
        validation_data=({'taxa_image': X_val_s, 'metadata': M_val_s}, y_val),
        epochs=EPOCHS,
        batch_size=params['batch_size'],
        class_weight=_class_weights(y_tr),
        callbacks=_callbacks(),
        verbose=verbose,
    )

    val_probs = model.predict(
        {'taxa_image': X_val_s, 'metadata': M_val_s}, verbose=0).ravel()
    threshold = _best_threshold(y_val, val_probs)
    return _metrics(y_val, val_probs, threshold, history), model, scaler, history
