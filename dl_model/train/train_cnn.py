"""
Training Script for TrafficCNN Model.
Trains a 1D Convolutional Neural Network for multi-class traffic density classification.
Incorporates class-weighted Cross-Entropy loss to mitigate distribution skew.
Handles missing PyTorch gracefully with clear exit instructions.
"""
import os
import sys
import numpy as np

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

from dl_model.models.vehicle_cnn import TrafficCNN, CLASSES
from dl_model.train.config import (
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
    CNN_INPUT_FEATURES, CNN_NUM_CLASSES, CNN_CLASS_WEIGHTS,
    CNN_LEARNING_RATE, CNN_WEIGHT_DECAY, CNN_BATCH_SIZE,
    CNN_EPOCHS, CNN_EARLY_STOPPING_PATIENCE,
    CNN_CHECKPOINT_PT, PROCESSED_DATA_DIR, DEVICE
)


def generate_synthetic_cnn_data(num_samples=1600):
    """
    Generate synthetic intersection feature vectors and ground truth class labels.
    Reflects realistic class imbalance: more medium/low traffic than extreme critical rush.
    """
    np.random.seed(RANDOM_SEED)
    X = np.zeros((num_samples, CNN_INPUT_FEATURES), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

    # Class probabilities: 25% low, 40% medium, 20% high, 15% critical
    class_assignments = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.25, 0.40, 0.20, 0.15])

    for i, c_idx in enumerate(class_assignments):
        y[i] = c_idx

        # Generate correlated features according to class index
        if c_idx == 0:  # low
            cars = np.random.randint(1, 8)
            bikes = np.random.randint(1, 6)
            autos = np.random.randint(0, 3)
            buses = np.random.randint(0, 2)
            trucks = np.random.randint(0, 1)
            speed = np.random.uniform(40.0, 55.0)
            queue = np.random.uniform(0.0, 15.0)
            wait = np.random.uniform(0.0, 15.0)
            intensity = np.random.uniform(0.1, 0.3)
        elif c_idx == 1:  # medium
            cars = np.random.randint(8, 20)
            bikes = np.random.randint(5, 15)
            autos = np.random.randint(3, 8)
            buses = np.random.randint(1, 3)
            trucks = np.random.randint(0, 2)
            speed = np.random.uniform(25.0, 42.0)
            queue = np.random.uniform(15.0, 40.0)
            wait = np.random.uniform(15.0, 35.0)
            intensity = np.random.uniform(0.35, 0.65)
        elif c_idx == 2:  # high
            cars = np.random.randint(20, 35)
            bikes = np.random.randint(12, 25)
            autos = np.random.randint(6, 14)
            buses = np.random.randint(2, 5)
            trucks = np.random.randint(1, 4)
            speed = np.random.uniform(12.0, 26.0)
            queue = np.random.uniform(40.0, 80.0)
            wait = np.random.uniform(35.0, 65.0)
            intensity = np.random.uniform(0.65, 0.85)
        else:  # critical
            cars = np.random.randint(32, 55)
            bikes = np.random.randint(20, 40)
            autos = np.random.randint(12, 22)
            buses = np.random.randint(3, 7)
            trucks = np.random.randint(2, 6)
            speed = np.random.uniform(3.0, 15.0)
            queue = np.random.uniform(75.0, 140.0)
            wait = np.random.uniform(60.0, 110.0)
            intensity = np.random.uniform(0.85, 1.0)

        total_v = cars + bikes + autos + buses + trucks
        weighted_u = cars * 1.0 + bikes * 0.5 + autos * 0.8 + buses * 3.0 + trucks * 3.5

        X[i, 0] = float(cars)
        X[i, 1] = float(bikes)
        X[i, 2] = float(autos)
        X[i, 3] = float(buses)
        X[i, 4] = float(trucks)
        X[i, 5] = float(total_v)
        X[i, 6] = float(weighted_u)
        X[i, 7] = float(speed)
        X[i, 8] = float(np.random.uniform(2.0, 8.0))   # speed variance
        X[i, 9] = float(queue)                         # queue length
        X[i, 10] = float(queue * np.random.uniform(0.3, 0.6))  # max approach queue
        X[i, 11] = float(wait)                         # wait time
        X[i, 12] = float(total_v * 0.8)                # inflow rate
        X[i, 13] = float(total_v * 0.7)                # outflow rate
        X[i, 14] = float(intensity)                    # time slot intensity
        X[i, 15] = float(np.random.uniform(0.0, 1.0))  # hour norm
        X[i, 16] = float(np.random.uniform(0.0, 1.0))  # day norm
        X[i, 17] = float(np.random.choice([0.0, 1.0], p=[0.7, 0.3]))  # weekend
        X[i, 18] = float(np.random.choice([0.0, 1.0], p=[0.95, 0.05])) # emergency flag
        X[i, 19] = float(np.random.choice([0.0, 0.3, 0.7], p=[0.8, 0.15, 0.05])) # weather

    return X, y


def train_cnn():
    """Main training routine for TrafficCNN."""
    if not TORCH_AVAILABLE:
        print("=" * 60)
        print("[INFO] PyTorch is not installed in this environment.")
        print("[INFO] TrafficCNN training requires PyTorch (pip install torch).")
        print("[INFO] The system operates seamlessly via the NumPy fallback model.")
        print("[INFO] Standalone execution completed gracefully.")
        print("=" * 60)
        sys.exit(0)

    print("=" * 60)
    print("Training TrafficCNN (PyTorch)")
    print(f"Device: {DEVICE} | Input Features: {CNN_INPUT_FEATURES} | Classes: {CNN_NUM_CLASSES}")
    print("=" * 60)

    data_file = os.path.join(PROCESSED_DATA_DIR, 'cnn_data.npz')
    if os.path.exists(data_file):
        print(f"[Data] Loading preprocessed data from {data_file}")
        data = np.load(data_file)
        X, y = data['X'], data['y']
    else:
        print("[Data] Generating synthetic intersection feature vectors...")
        X, y = generate_synthetic_cnn_data(num_samples=2000)

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

    train_loader = DataLoader(train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False)

    model = TrafficCNN(input_features=CNN_INPUT_FEATURES, num_classes=CNN_NUM_CLASSES).to(DEVICE)

    weights_tensor = torch.tensor(CNN_CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY)

    best_val_acc = 0.0
    patience_counter = 0

    print("\nStarting Training Loop...")
    for epoch in range(1, CNN_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)

        train_loss /= total_train
        train_acc = train_correct / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)

        val_loss /= total_val
        val_acc = val_correct / total_val

        print(f"Epoch [{epoch:02d}/{CNN_EPOCHS:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model.save_weights(CNN_CHECKPOINT_PT)
        else:
            patience_counter += 1
            if patience_counter >= CNN_EARLY_STOPPING_PATIENCE:
                print(f"[Early Stopping] Patience reached at epoch {epoch}.")
                break

    # Final evaluation on test set
    print("\nEvaluating on Test Set...")
    model.load_weights(CNN_CHECKPOINT_PT)
    test_preds, test_actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_actuals.extend(batch_y.numpy())

    test_preds = np.array(test_preds)
    test_actuals = np.array(test_actuals)
    total_acc = (test_preds == test_actuals).mean()

    print("-" * 50)
    print(f"Test Accuracy: {total_acc:.2%}")
    for c_id, c_name in enumerate(CLASSES):
        mask = (test_actuals == c_id)
        if np.sum(mask) > 0:
            c_acc = (test_preds[mask] == test_actuals[mask]).mean()
            print(f"  Class '{c_name}': {c_acc:.2%} ({np.sum(mask)} samples)")
    print("-" * 50)
    print(f"[SUCCESS] Best model weights preserved at {CNN_CHECKPOINT_PT}")


if __name__ == '__main__':
    train_cnn()
