"""
config.py — Central configuration for the Gut Microbiome CNN — IBD Classification project.

All paths, hyperparameters, and experiment constants live here.
Every other module imports from this file so there is a single place to edit.
"""
import os
from pathlib import Path

# ── Project root (resolved relative to this file so paths work from anywhere) ─
ROOT = Path(__file__).parent

# ── Output directories ────────────────────────────────────────────────────────
MODELS_DIR  = ROOT / 'saved_models'               # trained .keras files
RESULTS_DIR = ROOT / 'results'                    # Excel results
VIZ_DIR     = str(ROOT / 'results' / 'visualizations')  # saved figures

# ── Input paths ───────────────────────────────────────────────────────────────
DATA_DIR      = ROOT / 'data' / 'preprocessed'
METADATA_PATH = str(DATA_DIR / 'encoded_metadata.npz')
RESULTS_XLSX  = str(RESULTS_DIR / 'results_all_representations.xlsx')

# Create output directories at import time
os.makedirs(str(MODELS_DIR), exist_ok=True)
os.makedirs(str(RESULTS_DIR), exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
BASE_SEED = 42

# ── Model training ────────────────────────────────────────────────────────────
EPOCHS         = 40    # max epochs; EarlyStopping usually stops earlier
PATIENCE       = 8     # EarlyStopping patience (monitored on val_auc)
LOSS_FN        = 'binary_crossentropy'
THRESHOLD_BETA = 2.0   # F-beta used for threshold search:
                       #   beta=2 weights recall 2× more than precision,
                       #   appropriate for disease detection (FN more costly than FP)

# ── Experiment design ─────────────────────────────────────────────────────────
GS_FOLDS = 3   # folds for Phase 1 grid search  (12 combos × 3 = 36 trains per rep)
CV_FOLDS = 5   # folds for Phase 2 evaluation   (stable mean ± std for reporting)

# ── Image shape ───────────────────────────────────────────────────────────────
IMG_SIDE  = 78                       # taxa image side length (78×78 = 6,084 ≥ 6,007 taxa)
IMG_SHAPE = (IMG_SIDE, IMG_SIDE, 1)  # (height, width, channels)

# ── Metadata ──────────────────────────────────────────────────────────────────
META_DIM = 20  # number of encoded metadata dimensions

# Encoded metadata feature names — must match column order in encoded_metadata.npz.
# Derived from df_encoded_meta.columns in preprocessing.ipynb.
META_FEATURE_NAMES = [
    'age_corrected',                        # continuous, standardised
    'bmi_cat',                              # ordinal
    'alcohol_frequency',                    # ordinal
    'sleep_duration',                       # ordinal
    'exercise_frequency',                   # ordinal
    'smoking_frequency',                    # ordinal
    'antibiotic_history',                   # binary
    'sex_female',                           # one-hot
    'sex_male',                             # one-hot
    'race_African American',                # one-hot
    'race_Asian or Pacific Islander',       # one-hot
    'race_Caucasian',                       # one-hot
    'race_Hispanic',                        # one-hot
    'race_Other',                           # one-hot
    'geographic location_Australia',        # one-hot
    'geographic location_Canada',           # one-hot
    'geographic location_Germany',          # one-hot
    'geographic location_Other',            # one-hot
    'geographic location_USA',              # one-hot
    'geographic location_United Kingdom',   # one-hot
]
assert len(META_FEATURE_NAMES) == META_DIM, (
    f"Expected {META_DIM} feature names, got {len(META_FEATURE_NAMES)}"
)

# ── Feature importance ────────────────────────────────────────────────────────
IMPORTANCE_N_REPEATS = 10  # shuffles per feature in permutation importance

# ── Taxa representations ──────────────────────────────────────────────────────
# Each entry: (name, taxa_npz_path, saved_model_path)
REPRESENTATIONS = [
    ('binary',
     str(DATA_DIR / 'preprocessed_binary.npz'),
     str(MODELS_DIR / 'best_cnn_binary.keras')),
    ('normalized',
     str(DATA_DIR / 'preprocessed_normalized.npz'),
     str(MODELS_DIR / 'best_cnn_normalized.keras')),
    ('log',
     str(DATA_DIR / 'preprocessed_log.npz'),
     str(MODELS_DIR / 'best_cnn_log.keras')),
]

# ── Hyperparameter grid ───────────────────────────────────────────────────────
# 3 lr × 2 batch_size × 2 conv_drop = 12 combinations
# dense_drop is fixed at 0.5 (standard for wide penultimate dense layer)
PARAM_GRID = [
    {'lr': 1e-3, 'batch_size': 32, 'conv_drop': 0.25, 'dense_drop': 0.5},
    {'lr': 1e-3, 'batch_size': 64, 'conv_drop': 0.25, 'dense_drop': 0.5},
    {'lr': 1e-4, 'batch_size': 32, 'conv_drop': 0.25, 'dense_drop': 0.5},
    {'lr': 1e-4, 'batch_size': 64, 'conv_drop': 0.25, 'dense_drop': 0.5},
    {'lr': 1e-5, 'batch_size': 32, 'conv_drop': 0.25, 'dense_drop': 0.5},
    {'lr': 1e-5, 'batch_size': 64, 'conv_drop': 0.25, 'dense_drop': 0.5},
    {'lr': 1e-3, 'batch_size': 32, 'conv_drop': 0.35, 'dense_drop': 0.5},
    {'lr': 1e-3, 'batch_size': 64, 'conv_drop': 0.35, 'dense_drop': 0.5},
    {'lr': 1e-4, 'batch_size': 32, 'conv_drop': 0.35, 'dense_drop': 0.5},
    {'lr': 1e-4, 'batch_size': 64, 'conv_drop': 0.35, 'dense_drop': 0.5},
    {'lr': 1e-5, 'batch_size': 32, 'conv_drop': 0.35, 'dense_drop': 0.5},
    {'lr': 1e-5, 'batch_size': 64, 'conv_drop': 0.35, 'dense_drop': 0.5},
]
