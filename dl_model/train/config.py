"""
Training Configuration and Hyperparameters.
Defines training settings, loss functions, learning rates, checkpoint destinations,
and dataset split ratios for traffic prediction and classification models.
"""
import os

# Root directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DL_MODEL_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(DL_MODEL_DIR)

# Data paths
DATA_DIR = os.path.join(DL_MODEL_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Checkpoints directory
CHECKPOINTS_DIR = os.path.join(DL_MODEL_DIR, 'checkpoints')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Dataset Split Ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Global Reproducibility Seed
RANDOM_SEED = 42

# ──────────────── LSTM Training Hyperparameters ────────────────
LSTM_INPUT_SIZE = 5
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_SEQ_LENGTH = 12
LSTM_FORECAST_HORIZON = 4
LSTM_DROPOUT = 0.2

LSTM_LEARNING_RATE = 0.001
LSTM_WEIGHT_DECAY = 1e-5
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 30
LSTM_EARLY_STOPPING_PATIENCE = 5
LSTM_CHECKPOINT_PT = os.path.join(CHECKPOINTS_DIR, 'best_lstm.pt')
LSTM_CHECKPOINT_NPZ = os.path.join(CHECKPOINTS_DIR, 'best_lstm.npz')

# ──────────────── CNN Training Hyperparameters ────────────────
CNN_INPUT_FEATURES = 20
CNN_NUM_CLASSES = 4
CNN_CLASSES = ['low', 'medium', 'high', 'critical']
# Class weights to counteract dataset imbalance (more medium traffic than critical)
CNN_CLASS_WEIGHTS = [1.2, 0.8, 1.0, 1.5]

CNN_LEARNING_RATE = 0.001
CNN_WEIGHT_DECAY = 1e-4
CNN_BATCH_SIZE = 64
CNN_EPOCHS = 25
CNN_EARLY_STOPPING_PATIENCE = 5
CNN_CHECKPOINT_PT = os.path.join(CHECKPOINTS_DIR, 'best_cnn.pt')
CNN_CHECKPOINT_NPZ = os.path.join(CHECKPOINTS_DIR, 'best_cnn.npz')

# ──────────────── Device Configuration ────────────────
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'


if __name__ == '__main__':
    print("=" * 60)
    print("Autonomous Smart Traffic DL Training Configuration")
    print("=" * 60)
    print(f"Project Root:         {PROJECT_ROOT}")
    print(f"Data Processed Dir:   {PROCESSED_DATA_DIR}")
    print(f"Checkpoints Dir:      {CHECKPOINTS_DIR}")
    print(f"Data Splits:          Train={TRAIN_SPLIT:.0%}, Val={VAL_SPLIT:.0%}, Test={TEST_SPLIT:.0%}")
    print(f"LSTM Config:          hidden={LSTM_HIDDEN_SIZE}, layers={LSTM_NUM_LAYERS}, seq_len={LSTM_SEQ_LENGTH}, horizon={LSTM_FORECAST_HORIZON}")
    print(f"CNN Config:           features={CNN_INPUT_FEATURES}, classes={CNN_NUM_CLASSES}, class_weights={CNN_CLASS_WEIGHTS}")
    print(f"Compute Device:       {DEVICE}")
    print("[SUCCESS] Training configuration loaded successfully.")
