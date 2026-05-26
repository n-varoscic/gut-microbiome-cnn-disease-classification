"""
Called after all representations have been trained. Each function saves one
figure to viz_dir and prints the path.

Functions
---------
plot_performance_bars        — CV vs test AUC and F1 per representation
plot_cv_learning_curves      — val loss/AUC curves per fold per representation
plot_perm_importance_all     — grouped permutation importance (all representations)
plot_perm_importance_heatmap — features × representations heatmap
plot_grad_importance_all     — grouped gradient importance (all representations)
plot_metadata_contribution   — metadata model vs image-only baseline comparison
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from config import META_FEATURE_NAMES, RESULTS_XLSX


# ── Helpers ───────────────────────────────────────────────────────────────────

def _perm_matrix(all_perm_imp, feature_names):
    """Build a feature × representation DataFrame sorted by mean importance."""
    rep_names    = list(all_perm_imp.keys())
    perm_compare = pd.DataFrame({'feature': feature_names})
    for rn in rep_names:
        df = all_perm_imp[rn].set_index('feature')
        perm_compare[rn] = perm_compare['feature'].map(df['mean_auc_drop'])
    perm_compare = perm_compare.set_index('feature')
    perm_compare['_mean'] = perm_compare.mean(axis=1)
    return perm_compare.sort_values('_mean', ascending=False).drop(columns='_mean')


def _grad_matrix(all_grad_imp, feature_names):
    """Build a feature × representation DataFrame sorted by mean importance."""
    rep_names    = list(all_grad_imp.keys())
    grad_compare = pd.DataFrame({'feature': feature_names})
    for rn in rep_names:
        grad_compare[rn] = all_grad_imp[rn]
    grad_compare = grad_compare.set_index('feature')
    grad_compare['_mean'] = grad_compare.mean(axis=1)
    return grad_compare.sort_values('_mean', ascending=False).drop(columns='_mean')


_COLORS = ['steelblue', 'mediumpurple', 'crimson', 'slategray']


# ── Cross-representation charts (one figure aggregates all representations) ───

def plot_performance_bars(summary_df, viz_dir, model_label=''):
    """Grouped bar chart: CV AUC (± std) and CV F1 (± std) per representation.

    Args:
        summary_df   : DataFrame with columns representation, cv_auc_mean,
                       cv_auc_std, cv_f1_mean, cv_f1_std, test_auc, test_f1
        viz_dir      : str
        model_label  : str — optional suffix for title and filename,
                             e.g. 'image_only' → performance_comparison_image_only.png

    Saves:  performance_comparison.png  /  performance_comparison_{label}.png
    """
    os.makedirs(viz_dir, exist_ok=True)
    reps = summary_df['representation'].tolist()
    x, w = np.arange(len(reps)), 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.bar(x,     summary_df['cv_auc_mean'],  w, label='CV AUC (mean ± std)',
           yerr=summary_df['cv_auc_std'],  capsize=5, color='steelblue', alpha=0.85)
    ax.bar(x + w, summary_df['test_auc'],     w, label='Test AUC',
           color='darkorange', alpha=0.85)
    ax.set_xticks(x + w / 2); ax.set_xticklabels(reps)
    ax.set_ylabel('AUC'); ax.set_title('AUC by Representation')
    ax.set_ylim(0.5, 1.0); ax.legend(); ax.grid(axis='y', alpha=0.4)

    ax = axes[1]
    ax.bar(x,     summary_df['cv_f1_mean'],   w, label='CV F1 (mean ± std)',
           yerr=summary_df['cv_f1_std'],   capsize=5, color='steelblue', alpha=0.85)
    ax.bar(x + w, summary_df['test_f1'],      w, label='Test F1',
           color='darkorange', alpha=0.85)
    ax.set_xticks(x + w / 2); ax.set_xticklabels(reps)
    ax.set_ylabel('F1 Score'); ax.set_title('F1 by Representation')
    ax.set_ylim(0, 1.0); ax.legend(); ax.grid(axis='y', alpha=0.4)

    title_str = model_label.replace('_', ' ').title() if model_label else 'CNN + Metadata'
    plt.suptitle(f'{title_str} — Performance by Taxa Representation',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    suffix = f'_{model_label}' if model_label else ''
    p = os.path.join(viz_dir, f'performance_comparison{suffix}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Saved -> {p}")


def plot_cv_learning_curves(all_cv_histories, viz_dir, model_label=''):
    """Validation loss and AUC curves for each CV fold, per representation.

    Args:
        all_cv_histories : dict  — {rep_name: [History, ...]}
        viz_dir          : str
        model_label      : str   — optional suffix appended to filename,
                                   e.g. 'image_only' → cv_curves_{rep}_image_only.png

    Saves:  cv_curves_{rep}.png            (model_label='')
            cv_curves_{rep}_{label}.png    (model_label provided)
    """
    os.makedirs(viz_dir, exist_ok=True)
    suffix = f'_{model_label}' if model_label else ''
    for rep_name, histories in all_cv_histories.items():
        label_str = f'{rep_name}  |  {model_label}' if model_label else rep_name
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f'CV Learning Curves  [{label_str}]', fontsize=12)
        for fi, hist in enumerate(histories):
            ep = range(1, len(hist.history['loss']) + 1)
            axes[0].plot(ep, hist.history['val_loss'], alpha=0.75, label=f'Fold {fi+1}')
            axes[1].plot(ep, hist.history['val_auc'],  alpha=0.75, label=f'Fold {fi+1}')
        for ax, ylabel, title in zip(
                axes, ['Loss', 'AUC'], ['Validation Loss', 'Validation AUC']):
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
            ax.legend(fontsize=8); ax.grid(alpha=0.35)
        plt.tight_layout()
        p = os.path.join(viz_dir, f'cv_curves_{rep_name}{suffix}.png')
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show()
        print(f"Saved -> {p}")


def plot_perm_importance_all(all_perm_imp, feature_names, viz_dir):
    """Grouped horizontal bars: permutation importance for all representations.

    Features sorted by their mean importance across representations.
    Saves:  perm_importance_all_reps.png

    Returns:
        perm_compare : DataFrame — feature × representation importance matrix
    """
    os.makedirs(viz_dir, exist_ok=True)
    perm_compare = _perm_matrix(all_perm_imp, feature_names)
    rep_names    = list(perm_compare.columns)
    n_feat, n_rep = len(feature_names), len(rep_names)
    w_bar         = 0.8 / n_rep

    fig, ax = plt.subplots(figsize=(10, max(6, n_feat * 0.55)))
    yi = np.arange(len(perm_compare))
    for ri, rn in enumerate(rep_names):
        offset = (ri - n_rep / 2 + 0.5) * w_bar
        ax.barh(yi + offset, perm_compare[rn], w_bar,
                label=rn, color=_COLORS[ri % len(_COLORS)], alpha=0.85,
                edgecolor='white', linewidth=0.4)
    ax.set_yticks(yi); ax.set_yticklabels(perm_compare.index)
    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xlabel('Mean AUC drop after permutation  (higher = more important)')
    ax.set_title('Permutation Importance — All Representations\n'
                 '(features sorted by mean importance across representations)',
                 fontsize=11)
    ax.invert_yaxis(); ax.legend(loc='lower right'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(viz_dir, 'perm_importance_all_reps.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Saved -> {p}")
    return perm_compare


def plot_perm_importance_heatmap(all_perm_imp, feature_names, viz_dir):
    """Heatmap: features × representations for permutation importance.

    Red = feature helps, blue = noisy/harmful. Sorted by mean importance.
    Saves:  perm_importance_heatmap.png
    """
    os.makedirs(viz_dir, exist_ok=True)
    perm_compare = _perm_matrix(all_perm_imp, feature_names)
    rep_names    = list(perm_compare.columns)
    n_feat, n_rep = len(perm_compare), len(rep_names)

    fig, ax = plt.subplots(figsize=(max(5, n_rep * 1.8), max(6, n_feat * 0.55)))
    hm   = perm_compare.values.astype(float)
    vmax = max(np.abs(hm).max(), 1e-8)
    im   = ax.imshow(hm, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Mean AUC drop', shrink=0.6)
    ax.set_xticks(range(n_rep)); ax.set_xticklabels(rep_names, fontsize=10)
    ax.set_yticks(range(n_feat)); ax.set_yticklabels(perm_compare.index)
    for r in range(n_feat):
        for c in range(n_rep):
            val = hm[r, c]
            ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                    fontsize=7, color='black' if abs(val) < vmax * 0.6 else 'white')
    ax.set_title('Permutation Importance Heatmap\n'
                 '(red = helps; blue = hurts/noise; sorted by mean importance)',
                 fontsize=11)
    plt.tight_layout()
    p = os.path.join(viz_dir, 'perm_importance_heatmap.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Saved -> {p}")


def plot_grad_importance_all(all_grad_imp, feature_names, viz_dir):
    """Grouped horizontal bars: gradient importance for all representations.

    Features sorted by mean importance across representations.
    Saves:  grad_importance_all_reps.png

    Returns:
        grad_compare : DataFrame — feature × representation importance matrix
    """
    os.makedirs(viz_dir, exist_ok=True)
    grad_compare = _grad_matrix(all_grad_imp, feature_names)
    rep_names    = list(grad_compare.columns)
    n_feat, n_rep = len(feature_names), len(rep_names)
    w_bar         = 0.8 / n_rep

    fig, ax = plt.subplots(figsize=(10, max(6, n_feat * 0.55)))
    yi = np.arange(len(grad_compare))
    for ri, rn in enumerate(rep_names):
        offset = (ri - n_rep / 2 + 0.5) * w_bar
        ax.barh(yi + offset, grad_compare[rn], w_bar,
                label=rn, color=_COLORS[ri % len(_COLORS)], alpha=0.85,
                edgecolor='white', linewidth=0.4)
    ax.set_yticks(yi); ax.set_yticklabels(grad_compare.index)
    ax.set_xlabel('Mean |Input × Gradient|  (higher = more important)')
    ax.set_title('Gradient-based Importance — All Representations\n'
                 '(features sorted by mean importance across representations)',
                 fontsize=11)
    ax.invert_yaxis(); ax.legend(loc='lower right'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(viz_dir, 'grad_importance_all_reps.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Saved -> {p}")
    return grad_compare


def plot_metadata_contribution(summary_df, io_summary_df, representations, viz_dir):
    """Side-by-side comparison: metadata model vs image-only baseline.

    Left:  CV AUC (mean ± std) — discriminative power
    Right: Test recall         — sensitivity for IBD detection

    Delta annotations show how much metadata adds over the image-only baseline.
    Positive delta = metadata helps.

    Saves:  metadata_contribution.png
    """
    os.makedirs(viz_dir, exist_ok=True)
    reps = [r[0] for r in representations]
    x, w = np.arange(len(reps)), 0.35

    meta_auc = summary_df.set_index('representation')['cv_auc_mean']
    meta_std = summary_df.set_index('representation')['cv_auc_std']
    io_auc   = io_summary_df.set_index('representation')['cv_auc_mean']
    io_std   = io_summary_df.set_index('representation')['cv_auc_std']
    meta_rec = summary_df.set_index('representation')['test_recall']
    io_rec   = io_summary_df.set_index('representation')['test_recall']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # AUC comparison
    ax = axes[0]
    ax.bar(x - w/2, [meta_auc[r] for r in reps], w,
           yerr=[meta_std[r] for r in reps], capsize=5,
           label='CNN + metadata', color='steelblue', alpha=0.85)
    ax.bar(x + w/2, [io_auc[r]   for r in reps], w,
           yerr=[io_std[r]   for r in reps], capsize=5,
           label='CNN image only', color='mediumpurple', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(reps)
    ax.set_ylabel('CV AUC (mean ± std)')
    ax.set_title('Metadata Contribution — CV AUC')
    ax.set_ylim(0.5, 1.0); ax.legend(); ax.grid(axis='y', alpha=0.4)
    for i, r in enumerate(reps):
        delta = meta_auc[r] - io_auc[r]
        ax.text(i, max(meta_auc[r], io_auc[r]) + 0.012,
                f'{"+" if delta >= 0 else ""}{delta:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='steelblue' if delta >= 0 else 'crimson')

    # Recall comparison
    ax = axes[1]
    ax.bar(x - w/2, [meta_rec[r] for r in reps], w,
           label='CNN + metadata', color='steelblue', alpha=0.85)
    ax.bar(x + w/2, [io_rec[r]   for r in reps], w,
           label='CNN image only', color='mediumpurple', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(reps)
    ax.set_ylabel('Test Recall (sensitivity)')
    ax.set_title('Metadata Contribution — Test Recall')
    ax.set_ylim(0, 1.0); ax.legend(); ax.grid(axis='y', alpha=0.4)
    for i, r in enumerate(reps):
        delta = meta_rec[r] - io_rec[r]
        ax.text(i, max(meta_rec[r], io_rec[r]) + 0.015,
                f'{"+" if delta >= 0 else ""}{delta:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='steelblue' if delta >= 0 else 'crimson')

    plt.suptitle('CNN + Metadata  vs  CNN Image Only\n'
                 '(delta annotated above bars; positive = metadata helps)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    p = os.path.join(viz_dir, 'metadata_contribution.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Saved -> {p}")


# ── Per-representation charts (one figure per representation, per model call) ─

def plot_roc_curve(test_probs, y_test, rep_name, viz_dir, model_label=''):
    """ROC curve with AUC fill for any trained model.

    Args:
        test_probs   : (N_test,) raw predicted probabilities
        y_test       : (N_test,) binary ground-truth labels
        rep_name     : str — representation name (used in title + filename)
        viz_dir      : str — output directory
        model_label  : str — optional suffix, e.g. 'image_only'
                             → roc_{rep}_{label}.png  (or roc_{rep}.png if '')

    Saves:  roc_{rep_name}.png  /  roc_{rep_name}_{model_label}.png
    """
    os.makedirs(viz_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    auc_val      = roc_auc_score(y_test, test_probs)

    suffix    = f'_{model_label}' if model_label else ''
    label_str = f'{rep_name}  |  {model_label}' if model_label else rep_name

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color='steelblue', lw=2.5, label=f'AUC = {auc_val:.4f}')
    ax.fill_between(fpr, tpr, alpha=0.10, color='steelblue')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve  [{label_str}]', fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(viz_dir, f'roc_{rep_name}{suffix}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
    print(f"Saved -> {p}")


def plot_confusion_matrix(test_probs, threshold, y_test, rep_name, viz_dir,
                          model_label=''):
    """Confusion matrix (raw counts + row-normalised) for any trained model.

    Args:
        test_probs  : (N_test,) raw predicted probabilities
        threshold   : float — classification threshold (from F-beta sweep)
        y_test      : (N_test,) binary ground-truth labels
        rep_name    : str
        viz_dir     : str
        model_label : str — optional suffix, e.g. 'image_only'

    Saves:  confusion_{rep_name}.png  /  confusion_{rep_name}_{model_label}.png
    """
    os.makedirs(viz_dir, exist_ok=True)
    preds   = (test_probs >= threshold).astype(int)
    cm      = confusion_matrix(y_test.astype(int), preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = ['Healthy (0)', 'IBD (1)']

    suffix    = f'_{model_label}' if model_label else ''
    label_str = f'{rep_name}  |  {model_label}' if model_label else rep_name

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
    fig.suptitle(f'{label_str}  —  threshold = {threshold:.2f}', fontsize=12)
    plt.tight_layout()
    p = os.path.join(viz_dir, f'confusion_{rep_name}{suffix}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
    print(f"Saved -> {p}")
