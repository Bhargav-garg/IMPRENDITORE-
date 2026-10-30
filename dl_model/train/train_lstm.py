"""
Training Script for CongestionLSTM Model.
Trains a 2-layer LSTM on multi-step traffic congestion time-series sequences.
Handles missing PyTorch gracefully with clear exit instructions.
"""
import os
import sys
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from dl_model.models.congestion_lstm import CongestionLSTM, classify_congestion
from dl_model.train.config import (
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
    LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    LSTM_SEQ_LENGTH, LSTM_FORECAST_HORIZON, LSTM_DROPOUT,
    LSTM_LEARNING_RATE, LSTM_WEIGHT_DECAY, LSTM_BATCH_SIZE,
    LSTM_EPOCHS, LSTM_EARLY_STOPPING_PATIENCE,
    LSTM_CHECKPOINT_PT, LSTM_CHECKPOINT_NPZ, PROCESSED_DATA_DIR, DEVICE
)


def generate_synthetic_lstm_data(num_samples=1200, seq_len=12, horizon=4, input_size=5):
    """Generate synthetic sequential traffic dynamics for training."""
    np.random.seed(RANDOM_SEED)
    X = np.zeros((num_samples, seq_len, input_size), dtype=np.float32)
    y = np.zeros((num_samples, horizon), dtype=np.float32)

    for i in range(num_samples):
        # Base daily rhythm frequency
        base_intensity = np.random.uniform(0.1, 0.9)
        trend = np.random.choice([-1, 0, 1]) * np.random.uniform(0.01, 0.04)

        current_val = base_intensity
        history = []
        for t in range(seq_len):
            current_val = np.clip(current_val + trend + np.random.normal(0, 0.03), 0.05, 0.98)
            # Feature 0: vehicle count norm [0, 1]
            # Feature 1: speed norm [0, 1] (inversely correlated with count)
            # Feature 2: queue length norm [0, 1]
            # Feature 3: wait time norm [0, 1]
            # Feature 4: congestion index [0, 1]
            feat = [
                current_val,
                np.clip(1.0 - current_val + np.random.normal(0, 0.02), 0.05, 0.98),
                np.clip(current_val * 1.1 + np.random.normal(0, 0.03), 0.0, 1.0),
                np.clip(current_val * 0.9 + np.random.normal(0, 0.04), 0.0, 1.0),
                current_val
            ]
            X[i, t] = feat[:input_size]
            history.append(current_val)

        # Future horizon target
        for h in range(horizon):
            future_step = np.clip(current_val + trend * (h + 1) + np.random.normal(0, 0.03), 0.05, 0.98)
            y[i, h] = future_step

    return X, y


def evaluate_metrics(preds, targets):
    """Calculate MAE, RMSE, and class accuracy across forecast steps."""
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    pred_classes = [classify_congestion(p) for p in preds.flatten()]
    true_classes = [classify_congestion(t) for t in targets.flatten()]
    correct = sum(1 for p, t in zip(pred_classes, true_classes) if p == t)
    acc = correct / len(true_classes) if true_classes else 0.0

    return mae, rmse, acc


def train_lstm():
    """Main training routine for CongestionLSTM."""
    if not TORCH_AVAILABLE:
        print("=" * 60)
        print("[INFO] PyTorch is not installed in this environment.")
        print("[INFO] CongestionLSTM training requires PyTorch (pip install torch).")
        print("[INFO] The system operates seamlessly via the NumPy fallback model.")
        print("[INFO] Standalone execution completed gracefully.")
        print("=" * 60)
        sys.exit(0)

    print("=" * 60)
    print("Training CongestionLSTM (PyTorch)")
    print(f"Device: {DEVICE} | Hidden: {LSTM_HIDDEN_SIZE} | Layers: {LSTM_NUM_LAYERS}")
    print("=" * 60)

    # Check for preprocessed data or generate
    data_file = os.path.join(PROCESSED_DATA_DIR, 'lstm_data.npz')
    if os.path.exists(data_file):
        print(f"[Data] Loading preprocessed data from {data_file}")
        data = np.load(data_file)
        X, y = data['X'], data['y']
    else:
        print("[Data] Generating synthetic sequential traffic data for training...")
        X, y = generate_synthetic_lstm_data(num_samples=1500, seq_len=LSTM_SEQ_LENGTH, horizon=LSTM_FORECAST_HORIZON)

    # Train / Val / Test Splits
    n_samples = len(X)
    n_train = int(n_samples * TRAIN_SPLIT)
    n_val = int(n_samples * VAL_SPLIT)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    print(f"[Dataset] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)

    model = CongestionLSTM(
        input_size=LSTM_INPUT_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        forecast_horizon=LSTM_FORECAST_HORIZON,
        dropout=LSTM_DROPOUT
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE, weight_decay=LSTM_WEIGHT_DECAY)

    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting Training Loop...")
    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_list, val_targets_list = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(batch_y.cpu().numpy())

        val_loss /= len(val_dataset)
        val_preds_all = np.vstack(val_preds_list)
        val_targets_all = np.vstack(val_targets_list)
        val_mae, val_rmse, val_acc = evaluate_metrics(val_preds_all, val_targets_all)

        print(f"Epoch [{epoch:02d}/{LSTM_EPOCHS:02d}] - Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | Val MAE: {val_mae:.4f} | Class Acc: {val_acc:.1%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_weights(LSTM_CHECKPOINT_PT)
        else:
            patience_counter += 1
            if patience_counter >= LSTM_EARLY_STOPPING_PATIENCE:
                print(f"[Early Stopping] Patience reached at epoch {epoch}.")
                break

    # Final evaluation on Test Set
    print("\nEvaluating on Test Set...")
    model.load_weights(LSTM_CHECKPOINT_PT)
    test_preds_list, test_targets_list = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            preds = model(batch_x)
            test_preds_list.append(preds.cpu().numpy())
            test_targets_list.append(batch_y.numpy())

    test_preds_all = np.vstack(test_preds_list)
    test_targets_all = np.vstack(test_targets_list)
    test_mae, test_rmse, test_acc = evaluate_metrics(test_preds_all, test_targets_all)

    print("-" * 50)
    print(f"Test MAE:       {test_mae:.4f}")
    print(f"Test RMSE:      {test_rmse:.4f}")
    print(f"Test Class Acc: {test_acc:.2%}")
    print("-" * 50)
    print(f"[SUCCESS] Best model weights preserved at {LSTM_CHECKPOINT_PT}")


if __name__ == '__main__':
    train_lstm()
