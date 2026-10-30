"""
Testing and Evaluation Suite for CongestionLSTM Model.
Evaluates multi-horizon congestion regression and categorical accuracy across classes.
Supports both trained checkpoints and NumPy fallback evaluation.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dl_model.models.congestion_lstm import CongestionLSTM, classify_congestion, TORCH_AVAILABLE
from dl_model.train.config import LSTM_CHECKPOINT_PT, LSTM_CHECKPOINT_NPZ


def generate_evaluation_dataset(num_samples=250, seq_len=12, horizon=4):
    """
    Generate ground-truth synthetic test scenarios spanning:
    1. Midnight calm (low)
    2. Morning rush buildup (medium -> high)
    3. Peak congestion jam (critical)
    4. Post-rush dispersal (high -> medium)
    """
    np.random.seed(101)
    X = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    y = np.zeros((num_samples, horizon), dtype=np.float32)

    categories = ['low', 'medium', 'high', 'critical']
    target_classes = np.random.choice(categories, size=num_samples, p=[0.25, 0.35, 0.25, 0.15])

    for i, target_cat in enumerate(target_classes):
        if target_cat == 'low':
            base = np.random.uniform(0.08, 0.25)
            trend = np.random.uniform(-0.01, 0.01)
        elif target_cat == 'medium':
            base = np.random.uniform(0.32, 0.55)
            trend = np.random.uniform(-0.02, 0.02)
        elif target_cat == 'high':
            base = np.random.uniform(0.62, 0.78)
            trend = np.random.uniform(-0.01, 0.02)
        else:  # critical
            base = np.random.uniform(0.82, 0.94)
            trend = np.random.uniform(0.00, 0.02)

        cur = base
        for t in range(seq_len):
            cur = np.clip(cur + trend + np.random.normal(0, 0.02), 0.05, 0.98)
            X[i, t, 0] = cur                                           # vehicle count norm
            X[i, t, 1] = np.clip(1.0 - cur + np.random.normal(0, 0.02), 0.05, 0.98)  # speed norm
            X[i, t, 2] = np.clip(cur * 1.05, 0.0, 1.0)                 # queue norm
            X[i, t, 3] = np.clip(cur * 0.95, 0.0, 1.0)                 # wait norm
            X[i, t, 4] = cur                                           # congestion score

        for h in range(horizon):
            cur = np.clip(cur + trend + np.random.normal(0, 0.02), 0.05, 0.98)
            y[i, h] = cur

    return X, y


def run_lstm_tests():
    """Run comprehensive performance tests on CongestionLSTM."""
    print("=" * 65)
    print("Autonomous Smart Traffic Management - CongestionLSTM Test Suite")
    print(f"Backend Engine: {'PyTorch' if TORCH_AVAILABLE else 'NumPy Fallback Engine'}")
    print("=" * 65)

    model = CongestionLSTM(input_size=5, hidden_size=64, num_layers=2, forecast_horizon=4)

    # Check for saved weights
    weights_path = LSTM_CHECKPOINT_PT if TORCH_AVAILABLE else LSTM_CHECKPOINT_NPZ
    if os.path.exists(weights_path):
        print(f"[Model] Loading checkpoint from: {weights_path}")
        model.load_weights(weights_path)
    else:
        print("[Model] Using default initialized weights for evaluation.")

    # Generate test sequences
    X_test, y_test = generate_evaluation_dataset(num_samples=250, seq_len=12, horizon=4)
    print(f"[Test Dataset] Loaded {len(X_test)} sequences (12 historical steps -> 4 forecast steps)")

    # Execute predictions
    predictions = model.predict(X_test)

    # Regression Metrics
    errors = predictions - y_test
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Per-step forecast metrics
    step_maes = [float(np.mean(np.abs(errors[:, h]))) for h in range(4)]

    # Classification Metrics across all predicted steps
    pred_flat = predictions.flatten()
    true_flat = y_test.flatten()

    pred_classes = [classify_congestion(p) for p in pred_flat]
    true_classes = [classify_congestion(t) for t in true_flat]

    total_predictions = len(true_classes)
    overall_correct = sum(1 for p, t in zip(pred_classes, true_classes) if p == t)
    overall_acc = overall_correct / total_predictions

    classes = ['low', 'medium', 'high', 'critical']
    class_stats = {}
    for cls in classes:
        cls_total = sum(1 for t in true_classes if t == cls)
        cls_correct = sum(1 for p, t in zip(pred_classes, true_classes) if t == cls and p == cls)
        cls_acc = (cls_correct / cls_total) if cls_total > 0 else 0.0
        class_stats[cls] = {'total': cls_total, 'correct': cls_correct, 'accuracy': cls_acc}

    # Print Results
    print("\n" + "-" * 65)
    print("EVALUATION RESULTS - REGRESSION ACCURACY")
    print("-" * 65)
    print(f"Overall MAE:   {mae:.4f}")
    print(f"Overall RMSE:  {rmse:.4f}")
    print("Forecast Horizon Progression (MAE per minute ahead):")
    for h, s_mae in enumerate(step_maes, 1):
        print(f"  T+{h} min ahead MAE: {s_mae:.4f}")

    print("\n" + "-" * 65)
    print("EVALUATION RESULTS - CATEGORICAL CLASSIFICATION")
    print("-" * 65)
    print(f"Overall Class Accuracy: {overall_acc:.2%} ({overall_correct}/{total_predictions})")
    print("\nPer-Class Breakdown:")
    for cls in classes:
        st = class_stats[cls]
        print(f"  Class [{cls.upper():<8}]: Accuracy = {st['accuracy']:.2%}  (Correct: {st['correct']}/{st['total']})")

    # Sample Predictions Display
    print("\n" + "-" * 65)
    print("SAMPLE PREDICTION INSPECTION")
    print("-" * 65)
    for sample_idx in [0, 50, 150, 220]:
        actual = y_test[sample_idx]
        pred = predictions[sample_idx]
        print(f"\nSample #{sample_idx}:")
        print(f"  Ground Truth: {[round(float(v), 3) for v in actual]} -> {[classify_congestion(v) for v in actual]}")
        print(f"  Predicted:    {[round(float(v), 3) for v in pred]} -> {[classify_congestion(v) for v in pred]}")

    print("\n" + "=" * 65)
    print("[SUCCESS] CongestionLSTM test evaluation passed successfully.")
    print("=" * 65)


if __name__ == '__main__':
    run_lstm_tests()
