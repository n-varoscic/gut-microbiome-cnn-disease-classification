"""
Phase 4: Feature importance analysis for the metadata CNN.

Two complementary methods are applied to the final trained model on the test set:

  1. Permutation importance
     Shuffle each metadata feature independently n_repeats times and measure
     the mean drop in AUC-ROC. Model-agnostic; measures real signal loss.
     Positive drop = feature is useful; negative = adding noise is slightly
     harmful. Cost: n_features × n_repeats forward passes.

  2. Input × Gradient (saliency)
     attribution_i = (∂ŷ/∂m_i) × m_i per test sample.
     Single backward pass per batch. Enables per-sample attribution heatmaps.

Figures saved per representation (to viz_dir):
  perm_importance_{rep}_meta.png      — permutation AUC drop with error bars
  grad_importance_{rep}_meta.png      — mean |Input × Gradient| per feature
  attribution_heatmap_{rep}_meta.png  — per-sample signed attributions (IBD vs healthy)
  roc_{rep}_meta.png                  — ROC curve with AUC
  confusion_{rep}_meta.png            — confusion matrix (counts + row-normalised)
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import tensorflow as tf

from config import BASE_SEED, IMPORTANCE_N_REPEATS
from src.visualization import (
    plot_roc_curve, plot_confusion_matrix,
    plot_perm_importance_single, plot_grad_importance_single,
)


# ── Method 1: Permutation importance ─────────────────────────────────────────

def permutation_importance_meta(model, X_test_s, M_test, y_test,
                                feature_names, n_repeats=IMPORTANCE_N_REPEATS,
                                seed=BASE_SEED):
    """Compute permutation importance for each metadata feature.

    Args:
        model        : trained Keras model
        X_test_s     : scaled test images
        M_test       : test metadata, STANDARDIZED with the model's metadata
                       scaler (must match what the model saw during training)
        y_test       : test labels
        feature_names: list[str] — metadata column names
        n_repeats    : int — shuffles per feature
        seed         : int — RNG seed

    Returns:
        df           : DataFrame with [feature, mean_auc_drop, std_auc_drop],
                       sorted descending by mean_auc_drop
        baseline_auc : float — AUC on unshuffled data
    """
    rng          = np.random.default_rng(seed)
    base_probs   = model.predict(
        {'taxa_image': X_test_s, 'metadata': M_test}, verbose=0).ravel()
    baseline_auc = roc_auc_score(y_test, base_probs)

    rows = []
    for i, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            M_perm       = M_test.copy()
            M_perm[:, i] = M_perm[rng.permutation(len(M_perm)), i]
            probs        = model.predict(
                {'taxa_image': X_test_s, 'metadata': M_perm}, verbose=0).ravel()
            drops.append(baseline_auc - roc_auc_score(y_test, probs))
        rows.append({
            'feature'      : name,
            'mean_auc_drop': np.mean(drops),
            'std_auc_drop' : np.std(drops),
        })

    df = pd.DataFrame(rows).sort_values('mean_auc_drop', ascending=False)
    return df, baseline_auc


# ── Method 2: Input × Gradient attributions ──────────────────────────────────

def gradient_importance_meta(model, X_test_s, M_test, batch_size=256):
    """Compute Input × Gradient attributions for each metadata feature.

    Attribution for feature i: a_i = (∂ŷ/∂m_i) × m_i.
    Batched for memory efficiency; one backward pass per batch.

    Args:
        model      : trained Keras model
        X_test_s   : scaled test images
        M_test     : test metadata
        batch_size : int — samples per batch

    Returns:
        mean_attr  : (n_features,)         — mean |attribution| across test samples
        per_sample : (n_test, n_features)  — signed attributions per sample
    """
    all_attrs = []
    n = len(M_test)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_b = tf.constant(X_test_s[start:end].astype('float32'))
        M_b = tf.constant(M_test[start:end].astype('float32'))
        with tf.GradientTape() as tape:
            tape.watch(M_b)
            preds = model({'taxa_image': X_b, 'metadata': M_b}, training=False)
        grads = tape.gradient(preds, M_b).numpy()
        all_attrs.append(grads * M_test[start:end])  # element-wise Input × Grad

    per_sample = np.concatenate(all_attrs, axis=0)  # (n_test, n_features)
    mean_attr  = np.abs(per_sample).mean(axis=0)    # (n_features,)
    return mean_attr, per_sample


# ── Full Phase 4 runner ───────────────────────────────────────────────────────

def run_importance_analysis(model, X_test_s, M_test, y_test,
                            test_probs, test_threshold, feature_names,
                            rep_name, viz_dir):
    """Run full Phase 4 analysis and save all five figures.

    Args:
        model          : trained Keras model
        X_test_s       : scaled test images
        M_test         : test metadata
        y_test         : test labels
        test_probs     : (N_test,) raw probabilities from Phase 3
        test_threshold : float — F-beta-optimal threshold from Phase 3
        feature_names  : list[str]
        rep_name       : str — used in figure titles and file names
        viz_dir        : str — directory to save figures

    Returns:
        perm_df         : DataFrame — permutation importance results
        mean_attr       : (n_features,) — gradient importance
        per_sample_attr : (n_test, n_features) — per-sample attributions
    """
    os.makedirs(viz_dir, exist_ok=True)
    n_feat = len(feature_names)

    print(f"\n  [Phase 4 — Feature Importance: {rep_name}]")
    print(f"  Permutation: {n_feat} features × {IMPORTANCE_N_REPEATS} repeats "
          f"= {n_feat * IMPORTANCE_N_REPEATS} forward passes")

    # ── Figure 1: Permutation importance (auto-scale; Section 5 re-renders
    #             this with a shared xlim across all reps for comparability)
    perm_df, baseline_auc = permutation_importance_meta(
        model, X_test_s, M_test, y_test, feature_names)
    print(f"    Baseline AUC = {baseline_auc:.4f} | n_repeats = {IMPORTANCE_N_REPEATS}")
    plot_perm_importance_single(perm_df, rep_name, viz_dir)

    # ── Figure 2: Input × Gradient importance (auto-scale; Section 5 re-renders
    #             with shared xlim).  Values are normalised inside the viz to
    #             '% of total' so reps with different input scales are comparable.
    print(f"  Gradient: single backward pass over {len(M_test)} test samples (batched)")
    mean_attr, per_sample_attr = gradient_importance_meta(model, X_test_s, M_test)
    plot_grad_importance_single(mean_attr, feature_names, rep_name, viz_dir)

    # Note: attribution heatmap (Figure 3) is generated in Section 5 via
    # viz.plot_attribution_heatmap so a shared colour scale can be applied
    # across all three representations for appendix comparison.

    # ── Figures 3 & 4: ROC curve + confusion matrix ──────────────────────────
    # Delegated to src.visualization.* so the metadata model uses the EXACT
    # same plotting code (and therefore the exact same mirrored title format)
    # as the image-only model.  Filenames will be roc_{rep}_meta.png and
    # confusion_{rep}_meta.png; titles will say '| metadata' to mirror
    # the image-only side's '| image-only'.
    plot_roc_curve(test_probs, y_test, rep_name, viz_dir, model_label='meta')
    plot_confusion_matrix(test_probs, test_threshold, y_test,
                          rep_name, viz_dir, model_label='meta')

    return perm_df, mean_attr, per_sample_attr


# ── Grouped permutation importance ───────────────────────────────────────────

def grouped_permutation_importance_meta(model, X_test_s, M_test, y_test,
                                        feature_groups, feature_names,
                                        n_repeats=IMPORTANCE_N_REPEATS,
                                        seed=BASE_SEED):
    """Grouped permutation importance: permute all one-hot columns of a
    categorical variable simultaneously.

    Shuffling the whole group together keeps samples internally consistent — every sample
    still belongs to exactly one category, just a different one — and correctly
    attributes the combined importance of the original categorical variable.

    Args:
        model          : trained Keras model
        X_test_s       : scaled test images
        M_test         : test metadata
        y_test         : test labels
        feature_groups : dict {group_name: [col_names]}  (e.g. from META_FEATURE_GROUPS)
        feature_names  : list[str] — ordered feature names (index lookup)
        n_repeats      : int — shuffles per group
        seed           : int

    Returns:
        df           : DataFrame [group, n_features, mean_auc_drop, std_auc_drop],
                       sorted descending by mean_auc_drop
        baseline_auc : float
    """
    rng = np.random.default_rng(seed)
    base_probs   = model.predict(
        {'taxa_image': X_test_s, 'metadata': M_test}, verbose=0).ravel()
    baseline_auc = roc_auc_score(y_test, base_probs)

    rows = []
    for group_name, cols in feature_groups.items():
        col_idxs = [feature_names.index(c) for c in cols if c in feature_names]
        if not col_idxs:
            continue
        drops = []
        for _ in range(n_repeats):
            M_perm = M_test.copy()
            perm_idx = rng.permutation(len(M_perm))
            M_perm[:, col_idxs] = M_perm[perm_idx][:, col_idxs]
            probs = model.predict(
                {'taxa_image': X_test_s, 'metadata': M_perm}, verbose=0).ravel()
            drops.append(baseline_auc - roc_auc_score(y_test, probs))
        rows.append({
            'group'        : group_name,
            'n_features'   : len(col_idxs),
            'mean_auc_drop': np.mean(drops),
            'std_auc_drop' : np.std(drops),
        })

    df = pd.DataFrame(rows).sort_values('mean_auc_drop', ascending=False)
    print(f"    Grouped permutation done: {len(rows)} groups  "
          f"(baseline AUC = {baseline_auc:.4f})")
    return df, baseline_auc
