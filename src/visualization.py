"""
Called after all representations have been trained. Each function saves one
figure to viz_dir and prints the path.

Functions
---------
plot_performance_bars           — CV vs test AUC and F1 per representation
plot_cv_learning_curves         — val loss/AUC curves per fold per representation
plot_perm_importance_all        — grouped permutation importance (all representations)
plot_perm_importance_heatmap    — features × representations heatmap
plot_grad_importance_all        — grouped gradient importance (all representations)
plot_metadata_contribution      — metadata model vs image-only baseline comparison
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

# ── Naming convention ────────────────────────────────────────────────────────

_LABEL_DISPLAY = {'io': 'image-only', 'meta': 'metadata'}

def _display(label):
    """Map a code-side model label to its display form for plot titles."""
    if not label:
        return ''
    return _LABEL_DISPLAY.get(label, label.replace('_', ' '))


# ── Cross-representation charts (one figure aggregates all representations) ───

def plot_performance_bars(summary_df, viz_dir, model_label=''):
    """Grouped bar chart: CV AUC (± std) and CV F1 (± std) per representation.

    Args:
        summary_df   : DataFrame with columns representation, cv_auc_mean,
                       cv_auc_std, cv_f1_mean, cv_f1_std, test_auc, test_f1
        viz_dir      : str
        model_label  : str — 'io' or 'meta'.  Drives both the title
                             ('CNN image-only' / 'CNN metadata') and the
                             filename (performance_comparison_{label}.png).

    Saves:  performance_comparison_{label}.png   (label='io' or 'meta')
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

    title_str = f'CNN {_display(model_label)}' if model_label else 'CNN'
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
        model_label      : str   — 'io' or 'meta'.  Drives title ('image-only' /
                                   'metadata') and filename suffix.

    Saves:  cv_curves_{rep}.png            (model_label='')
            cv_curves_{rep}_{label}.png    (label='io' or 'meta')
    """
    os.makedirs(viz_dir, exist_ok=True)
    suffix = f'_{model_label}' if model_label else ''
    for rep_name, histories in all_cv_histories.items():
        label_str = f'{rep_name}  |  {_display(model_label)}' if model_label else rep_name
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
    p = os.path.join(viz_dir, 'perm_importance_all_reps_meta.png')
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
    p = os.path.join(viz_dir, 'perm_importance_heatmap_meta.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Saved -> {p}")


def plot_grad_importance_all(all_grad_imp, feature_names, viz_dir):
    """Grouped horizontal bars: gradient importance for all representations.

    Raw |Input × Gradient| values are normalised to relative importance
    (% of total per representation) before plotting so that representations
    with different input scales remain directly comparable.
    Features sorted by mean relative importance across representations.

    Saves:  grad_importance_all_reps_meta.png

    Returns:
        grad_compare : DataFrame — feature × representation relative importance (%)
    """
    os.makedirs(viz_dir, exist_ok=True)

    # Normalise each representation to % of total before building the matrix
    all_grad_rel = {}
    for rn, values in all_grad_imp.items():
        total = np.sum(values)
        all_grad_rel[rn] = (values / total * 100) if total > 0 else values

    grad_compare = _grad_matrix(all_grad_rel, feature_names)
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
    ax.set_xlabel('Relative importance  (% of total |Input × Gradient|  per representation)')
    ax.set_title('Gradient-based Importance — All Representations\n'
                 '(normalised within each representation; sorted by mean relative importance)',
                 fontsize=11)
    ax.invert_yaxis(); ax.legend(loc='lower right'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(viz_dir, 'grad_importance_all_reps_meta.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Saved -> {p}")
    return grad_compare


def plot_combined_performance(meta_summary_df, io_summary_df, representations, viz_dir):
    """Primary performance comparison: image-only vs metadata for all representations.

    Left subplot  — Test AUC  (error bars = CV AUC std from 5-fold CV)
    Right subplot — Test Recall (IBD sensitivity — most clinically relevant metric)

    Delta annotations show metadata gain over image-only baseline.
    Positive delta = metadata helps. Both subplots share y-axis bounds
    so AUC [0.5–1.0] and Recall [0.0–1.0] are on interpretable scales.

    Saves:  performance_combined.png

    Args:
        meta_summary_df : DataFrame with columns representation, cv_auc_std, test_auc, test_recall
        io_summary_df   : same structure for image-only baseline
        representations : list of (name, ...) tuples (from REPRESENTATIONS config)
        viz_dir         : str
    """
    os.makedirs(viz_dir, exist_ok=True)
    reps = [r[0] for r in representations]
    x, w = np.arange(len(reps)), 0.35

    meta = meta_summary_df.set_index('representation')
    io   = io_summary_df.set_index('representation')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Colours: image-only = blue, metadata = purple (image-only listed first)
    C_IO, C_META = 'steelblue', 'mediumpurple'

    # ── Left: Test AUC ───────────────────────────────────────────────────────
    ax = axes[0]
    io_auc   = [io['test_auc'][r]     for r in reps]
    meta_auc = [meta['test_auc'][r]   for r in reps]
    io_err   = [io['cv_auc_std'][r]   for r in reps]
    meta_err = [meta['cv_auc_std'][r] for r in reps]
    ax.bar(x - w/2, io_auc,   w, yerr=io_err,   capsize=5,
           label='CNN image-only', color=C_IO,   alpha=0.85)
    ax.bar(x + w/2, meta_auc, w, yerr=meta_err, capsize=5,
           label='CNN metadata',   color=C_META, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(reps)
    ax.set_ylabel('Test AUC'); ax.set_title('AUC by Representation')
    # Both subplots share ylim (0, 1.2): identical scale + headroom for labels
    ax.set_ylim(0, 1.2); ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.4)
    # Value labels (bar height) above each bar, clearing the error bar
    for xi, v, e in zip(x - w/2, io_auc,   io_err):
        ax.text(xi, v + e + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    for xi, v, e in zip(x + w/2, meta_auc, meta_err):
        ax.text(xi, v + e + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # ── Right: Test Recall ────────────────────────────────────────────────────
    ax = axes[1]
    io_rec   = [io['test_recall'][r]   for r in reps]
    meta_rec = [meta['test_recall'][r] for r in reps]
    ax.bar(x - w/2, io_rec,   w, label='CNN image-only', color=C_IO,   alpha=0.85)
    ax.bar(x + w/2, meta_rec, w, label='CNN metadata',   color=C_META, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(reps)
    ax.set_ylabel('Test Recall (IBD sensitivity)')
    ax.set_title('Recall by Representation')
    # Same scale as the AUC subplot for direct visual comparison
    ax.set_ylim(0, 1.2); ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.4)
    for xi, v in zip(x - w/2, io_rec):
        ax.text(xi, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    for xi, v in zip(x + w/2, meta_rec):
        ax.text(xi, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('CNN image-only  vs  CNN metadata — Test Performance\n'
                 '(bar labels = metric value;  error bars on AUC = CV AUC std)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    p = os.path.join(viz_dir, 'performance_combined.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Saved -> {p}")


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

    # Colours: image-only = blue, metadata = purple (image-only listed first)
    C_IO, C_META = 'steelblue', 'mediumpurple'

    # AUC comparison (image-only left, metadata right)
    ax = axes[0]
    io_a   = [io_auc[r]   for r in reps]
    meta_a = [meta_auc[r] for r in reps]
    io_e   = [io_std[r]   for r in reps]
    meta_e = [meta_std[r] for r in reps]
    ax.bar(x - w/2, io_a,   w, yerr=io_e,   capsize=5,
           label='CNN image-only', color=C_IO,   alpha=0.85)
    ax.bar(x + w/2, meta_a, w, yerr=meta_e, capsize=5,
           label='CNN metadata',   color=C_META, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(reps)
    ax.set_ylabel('CV AUC (mean ± std)')
    ax.set_title('Metadata Contribution — CV AUC')
    # Both subplots share ylim (0, 1.2): identical scale + headroom for labels
    ax.set_ylim(0, 1.2); ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.4)
    for xi, v, e in zip(x - w/2, io_a,   io_e):
        ax.text(xi, v + e + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    for xi, v, e in zip(x + w/2, meta_a, meta_e):
        ax.text(xi, v + e + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # Recall comparison (image-only left, metadata right)
    ax = axes[1]
    io_r   = [io_rec[r]   for r in reps]
    meta_r = [meta_rec[r] for r in reps]
    ax.bar(x - w/2, io_r,   w, label='CNN image-only', color=C_IO,   alpha=0.85)
    ax.bar(x + w/2, meta_r, w, label='CNN metadata',   color=C_META, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(reps)
    ax.set_ylabel('Test Recall (sensitivity)')
    ax.set_title('Metadata Contribution — Test Recall')
    # Same scale as the AUC subplot for direct visual comparison
    ax.set_ylim(0, 1.2); ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.4)
    for xi, v in zip(x - w/2, io_r):
        ax.text(xi, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    for xi, v in zip(x + w/2, meta_r):
        ax.text(xi, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('CNN image-only  vs  CNN metadata\n'
                 '(bar labels = metric value;  error bars on CV AUC = std across folds)',
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
        model_label  : str — 'io' or 'meta'.  Title shows 'image-only' / 'metadata';
                             filename gets suffix _io / _meta.

    Saves:  roc_{rep_name}.png  /  roc_{rep_name}_{label}.png   (label='io' or 'meta')
    """
    os.makedirs(viz_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    auc_val      = roc_auc_score(y_test, test_probs)

    suffix    = f'_{model_label}' if model_label else ''
    label_str = f'{rep_name}  |  {_display(model_label)}' if model_label else rep_name

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
        model_label : str — 'io' or 'meta'.  Title shows 'image-only' / 'metadata';
                            filename gets suffix _io / _meta.

    Saves:  confusion_{rep_name}.png  /  confusion_{rep_name}_{label}.png
    """
    os.makedirs(viz_dir, exist_ok=True)
    preds   = (test_probs >= threshold).astype(int)
    cm      = confusion_matrix(y_test.astype(int), preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = ['Healthy (0)', 'IBD (1)']

    suffix    = f'_{model_label}' if model_label else ''
    label_str = f'{rep_name}  |  {_display(model_label)}' if model_label else rep_name

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


def plot_perm_importance_single(perm_df, rep_name, viz_dir, xlim=None):
    """Single-representation permutation-importance bar chart.

    Called twice for each rep:
      1. Inside the metadata loop via run_importance_analysis (xlim=None → auto-scale,
         crash-safe so a per-rep figure exists even if Section 5 never runs).
      2. From Section 5 with a shared xlim derived from all reps so the three
         figures become visually comparable side-by-side in the thesis.

    Args:
        perm_df   : DataFrame [feature, mean_auc_drop, std_auc_drop],
                    sorted descending by mean_auc_drop
        rep_name  : str
        viz_dir   : str
        xlim      : (lo, hi) tuple or None — when set, applied as ax.set_xlim()

    Saves:  perm_importance_{rep_name}_meta.png
    """
    os.makedirs(viz_dir, exist_ok=True)
    n_feat = len(perm_df)
    bar_h  = max(5, n_feat * 0.42)

    fig, ax = plt.subplots(figsize=(9, bar_h))
    colors  = ['#d62728' if v > 0 else '#aec7e8' for v in perm_df['mean_auc_drop']]
    ax.barh(perm_df['feature'], perm_df['mean_auc_drop'],
            xerr=perm_df['std_auc_drop'], color=colors,
            capsize=4, alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Mean AUC drop after permutation  (higher = more important)')
    ax.set_title(f'Metadata Permutation Importance  [{rep_name}]', fontsize=11)
    ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.tight_layout()
    p = os.path.join(viz_dir, f'perm_importance_{rep_name}_meta.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
    print(f"Saved -> {p}")


def plot_grad_importance_single(mean_attr, feature_names, rep_name, viz_dir,
                                 xlim=None):
    """Single-representation gradient-importance bar chart (relative %).

    Normalises mean_attr to % of total within the rep so reps with different
    input magnitude scales (binary vs log vs normalized) can be directly
    compared.  Pass a shared xlim from Section 5 to lock the three per-rep
    figures onto the same axis.

    Args:
        mean_attr     : (n_features,) numpy array of mean |Input × Gradient|
        feature_names : list[str] — column order matching mean_attr
        rep_name      : str
        viz_dir       : str
        xlim          : (0, hi) tuple or None — when set, applied as ax.set_xlim()

    Saves:  grad_importance_{rep_name}_meta.png
    """
    os.makedirs(viz_dir, exist_ok=True)
    n_feat = len(feature_names)
    bar_h  = max(5, n_feat * 0.42)

    total    = float(np.sum(mean_attr))
    rel_attr = (mean_attr / total * 100) if total > 0 else mean_attr
    grad_df  = (pd.DataFrame({'feature': feature_names, 'rel_importance': rel_attr})
                .sort_values('rel_importance', ascending=False))

    fig, ax = plt.subplots(figsize=(9, bar_h))
    ax.barh(grad_df['feature'], grad_df['rel_importance'],
            color='steelblue', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Relative importance  (% of total |Input × Gradient|)')
    ax.set_title(f'Gradient-based Metadata Importance  [{rep_name}]', fontsize=11)
    ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.tight_layout()
    p = os.path.join(viz_dir, f'grad_importance_{rep_name}_meta.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
    print(f"Saved -> {p}")


def plot_attribution_heatmap(per_sample_attr, perm_df, feature_names, y_test,
                              rep_name, viz_dir, vmax=None):
    """Per-sample Input×Gradient attribution heatmap.

    Features are ordered top-to-bottom by permutation importance.
    Samples are arranged IBD (left) | Healthy (right) with a dashed boundary.

    Args:
        per_sample_attr : (n_test, n_features) — signed attributions
        perm_df         : DataFrame from permutation_importance_meta — sets feature order
        feature_names   : list[str]
        y_test          : (n_test,) binary labels
        rep_name        : str
        viz_dir         : str
        vmax            : float or None — shared colour scale cap across representations
                          (99th percentile of this rep if None → auto-scale)

    Saves:  attribution_heatmap_{rep_name}_meta.png
    """
    os.makedirs(viz_dir, exist_ok=True)
    n_feat = len(feature_names)

    # Subsample to ≤200 (100 IBD + 100 healthy) for readability
    ibd_idx  = np.where(np.array(y_test) == 1)[0]
    heal_idx = np.where(np.array(y_test) == 0)[0]
    n_ibd_s  = min(len(ibd_idx),  100)
    n_hel_s  = min(len(heal_idx), 100)
    idx_show = np.concatenate([ibd_idx[:n_ibd_s], heal_idx[:n_hel_s]])

    # Order features by permutation importance (most important at top)
    feat_order = perm_df['feature'].tolist()
    feat_idx   = [feature_names.index(f) for f in feat_order if f in feature_names]
    hm_data    = per_sample_attr[idx_show][:, feat_idx]

    if vmax is None:
        vmax = max(np.percentile(np.abs(hm_data), 99), 1e-8)

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
    p = os.path.join(viz_dir, f'attribution_heatmap_{rep_name}_meta.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
    print(f"Saved -> {p}")


def plot_grouped_perm_importance_all(all_grouped_perm, viz_dir):
    """Grouped bar chart: grouped permutation importance for all representations.

    Categorical variables (sex, race, geographic_location) are treated as units —
    their one-hot columns are permuted together — giving the true importance of
    the original variable rather than individual dummy columns.

    Features sorted by mean AUC drop across representations.
    Saves:  perm_importance_grouped_all_reps_meta.png

    Returns:
        group_compare : DataFrame — group × representation importance matrix
    """
    os.makedirs(viz_dir, exist_ok=True)
    rep_names = list(all_grouped_perm.keys())

    # Build group × representation DataFrame aligned on group name
    all_dfs = {rn: df.set_index('group') for rn, df in all_grouped_perm.items()}
    all_groups = list(all_dfs[rep_names[0]].index)
    group_compare = pd.DataFrame(index=all_groups)
    for rn in rep_names:
        group_compare[rn] = all_dfs[rn]['mean_auc_drop']
    group_compare['_mean'] = group_compare.mean(axis=1)
    group_compare = group_compare.sort_values('_mean', ascending=False).drop(columns='_mean')

    n_groups = len(group_compare)
    n_rep    = len(rep_names)
    w_bar    = 0.8 / n_rep

    fig, ax = plt.subplots(figsize=(9, max(5, n_groups * 0.55)))
    yi = np.arange(n_groups)
    for ri, rn in enumerate(rep_names):
        offset = (ri - n_rep / 2 + 0.5) * w_bar
        ax.barh(yi + offset, group_compare[rn], w_bar,
                label=rn, color=_COLORS[ri % len(_COLORS)], alpha=0.85,
                edgecolor='white', linewidth=0.4)
    ax.set_yticks(yi); ax.set_yticklabels(group_compare.index)
    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xlabel('Mean AUC drop after grouped permutation  (higher = more important)')
    ax.set_title('Grouped Permutation Importance — All Representations\n'
                 '(one-hot dummies permuted together per original categorical variable)',
                 fontsize=11)
    ax.invert_yaxis(); ax.legend(loc='lower right'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(viz_dir, 'perm_importance_grouped_all_reps_meta.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Saved -> {p}")
    return group_compare
