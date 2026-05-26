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
  perm_importance_{rep}.png      — permutation AUC drop with error bars
  grad_importance_{rep}.png      — mean |Input × Gradient| per feature
  attribution_heatmap_{rep}.png  — per-sample signed attributions (IBD vs healthy)
  roc_{rep}.png                  — ROC curve with AUC
  confusion_{rep}.png            — confusion matrix (counts + row-normalised)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import tensorflow as tf

from config import BASE_SEED, IMPORTANCE_N_REPEATS


# ── Method 1: Permutation importance ─────────────────────────────────────────

def permutation_importance_meta(model, X_test_s, M_test, y_test,
                                feature_names, n_repeats=IMPORTANCE_N_REPEATS,
                                seed=BASE_SEED):
    """Compute permutation importance for each metadata feature.

    Args:
        model        : trained Keras model
        X_test_s     : scaled test images
        M_test       : test metadata (unscaled)
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
    bar_h  = max(5, n_feat * 0.42)

    print(f"\n  [Phase 4 — Feature Importance: {rep_name}]")
    print(f"  Permutation: {n_feat} features × {IMPORTANCE_N_REPEATS} repeats "
          f"= {n_feat * IMPORTANCE_N_REPEATS} forward passes")

    # ── Figure 1: Permutation importance ─────────────────────────────────────
    perm_df, baseline_auc = permutation_importance_meta(
        model, X_test_s, M_test, y_test, feature_names)

    fig, ax = plt.subplots(figsize=(9, bar_h))
    colors  = ['#d62728' if v > 0 else '#aec7e8' for v in perm_df['mean_auc_drop']]
    ax.barh(perm_df['feature'], perm_df['mean_auc_drop'],
            xerr=perm_df['std_auc_drop'], color=colors,
            capsize=4, alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Mean AUC drop after permutation  (higher = more important)')
    ax.set_title(f'Metadata Permutation Importance  [{rep_name}]\n'
                 f'Baseline AUC = {baseline_auc:.4f} | n_repeats = {IMPORTANCE_N_REPEATS}',
                 fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    p = os.path.join(viz_dir, f'perm_importance_{rep_name}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
    print(f"    Saved -> {p}")

    # ── Figure 2: Input × Gradient importance ────────────────────────────────
    print(f"  Gradient: single backward pass over {len(M_test)} test samples (batched)")
    mean_attr, per_sample_attr = gradient_importance_meta(model, X_test_s, M_test)

    grad_df = (pd.DataFrame({'feature': feature_names, 'mean_attr': mean_attr})
               .sort_values('mean_attr', ascending=False))
    fig, ax = plt.subplots(figsize=(9, bar_h))
    ax.barh(grad_df['feature'], grad_df['mean_attr'],
            color='steelblue', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Mean |Input × Gradient|  (higher = more important)')
    ax.set_title(f'Gradient-based Metadata Importance  [{rep_name}]', fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    p = os.path.join(viz_dir, f'grad_importance_{rep_name}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
    print(f"    Saved -> {p}")

    # ── Figure 3: Per-sample attribution heatmap ──────────────────────────────
    # Subsample to ≤200 (100 IBD + 100 healthy) for readability
    ibd_idx  = np.where(y_test == 1)[0]
    heal_idx = np.where(y_test == 0)[0]
    n_ibd_s  = min(len(ibd_idx),  100)
    n_hel_s  = min(len(heal_idx), 100)
    idx_show = np.concatenate([ibd_idx[:n_ibd_s], heal_idx[:n_hel_s]])

    # Order features by permutation importance (most important at top)
    feat_order = perm_df['feature'].tolist()
    feat_idx   = [feature_names.index(f) for f in feat_order if f in feature_names]
    hm_data    = per_sample_attr[idx_show][:, feat_idx]
    vmax       = max(np.percentile(np.abs(hm_data), 99), 1e-8)

    fig, ax = plt.subplots(figsize=(13, max(4, n_feat * 0.38)))
    im = ax.imshow(hm_data.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Input × Gradient', shrink=0.7, pad=0.02)
    ax.set_yticks(range(len(feat_order)))
    ax.set_yticklabels(feat_order, fontsize=8)
    ax.set_xlabel(
        f'Test samples  [0–{n_ibd_s-1}: IBD  |  {n_ibd_s}–{n_ibd_s+n_hel_s-1}: Healthy]',
        fontsize=9)
    ax.set_title(f'Per-sample Metadata Attributions  [{rep_name}]\n'
                 f'(features ranked top-to-bottom by permutation importance)', fontsize=11)
    ax.axvline(n_ibd_s - 0.5, color='gold', linewidth=1.5, linestyle='--',
               label='IBD | Healthy boundary')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    p = os.path.join(viz_dir, f'attribution_heatmap_{rep_name}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
    print(f"    Saved -> {p}")

    # ── Figure 4: ROC curve ───────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    auc_val     = roc_auc_score(y_test, test_probs)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color='steelblue', lw=2.5, label=f'AUC = {auc_val:.4f}')
    ax.fill_between(fpr, tpr, alpha=0.10, color='steelblue')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve  [{rep_name}]  —  Final Model', fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(viz_dir, f'roc_{rep_name}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
    print(f"    Saved -> {p}")

    # ── Figure 5: Confusion matrix (counts + row-normalised side-by-side) ────
    preds   = (test_probs >= test_threshold).astype(int)
    cm      = confusion_matrix(y_test.astype(int), preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = ['Healthy (0)', 'IBD (1)']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, fmt, title in zip(
            axes, [cm, cm_norm], ['d', '.2f'], ['Counts', 'Row-normalised']):
        im = ax.imshow(data, cmap='Blues', vmin=0, vmax=data.max() * 1.1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for r in range(2):
            for c in range(2):
                val   = format(data[r, c], fmt)
                color = 'white' if data[r, c] > data.max() * 0.6 else 'black'
                ax.text(c, r, val, ha='center', va='center',
                        color=color, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 1]); ax.set_xticklabels([f'Pred {l}' for l in labels])
        ax.set_yticks([0, 1]); ax.set_yticklabels([f'True {l}' for l in labels])
        ax.set_title(f'Confusion Matrix ({title})', fontsize=10)
    fig.suptitle(f'{rep_name}  —  threshold = {test_threshold:.2f}', fontsize=12)
    plt.tight_layout()
    p = os.path.join(viz_dir, f'confusion_{rep_name}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
    print(f"    Saved -> {p}")

    return perm_df, mean_attr, per_sample_attr
