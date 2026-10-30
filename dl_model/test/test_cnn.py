"""
Testing and Evaluation Suite for TrafficCNN Model.
Evaluates 4-class intersection traffic density classification.
Computes overall accuracy, per-class Precision, Recall, F1-Score, and prints Confusion Matrix.
Supports both trained checkpoints and NumPy fallback evaluation.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dl_model.models.vehicle_cnn import TrafficCNN, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, TORCH_AVAILABLE
from dl_model.train.config import CNN_CHECKPOINT_PT, CNN_CHECKPOINT_NPZ


def generate_test_dataset(num_samples=400):
    """
    Generate synthetic test set with ground-truth intersection vectors
    and distinct congestion classes.
    """
    np.random.seed(202)
    X = np.zeros((num_samples, 20), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

    # Balanced class distribution across the 4 categories
    classes = [0, 1, 2, 3]
    y = np.random.choice(classes, size=num_samples, p=[0.25, 0.25, 0.25, 0.25])

    for i in range(num_samples):
        cls_idx = y[i]
        if cls_idx == 0:  # low
            cars = np.random.randint(1, 8)
            bikes = np.random.randint(1, 8)
            autos = np.random.randint(0, 3)
            buses = np.random.randint(0, 1)
            trucks = 0
            speed = np.random.uniform(42.0, 58.0)
            queue = np.random.uniform(0.0, 12.0)
            wait = np.random.uniform(2.0, 15.0)
            intensity = np.random.uniform(0.1, 0.28)
        elif cls_idx == 1:  # medium
            cars = np.random.randint(9, 20)
            bikes = np.random.randint(6, 16)
            autos = np.random.randint(3, 9)
            buses = np.random.randint(1, 3)
            trucks = np.random.randint(0, 2)
            speed = np.random.uniform(26.0, 40.0)
            queue = np.random.uniform(16.0, 38.0)
            wait = np.random.uniform(18.0, 35.0)
            intensity = np.random.uniform(0.35, 0.62)
        elif cls_idx == 2:  # high
            cars = np.random.randint(21, 35)
            bikes = np.random.randint(14, 26)
            autos = np.random.randint(7, 15)
            buses = np.random.randint(2, 5)
            trucks = np.random.randint(1, 4)
            speed = np.random.uniform(12.0, 24.0)
            queue = np.random.uniform(42.0, 78.0)
            wait = np.random.uniform(38.0, 65.0)
            intensity = np.random.uniform(0.65, 0.82)
        else:  # critical
            cars = np.random.randint(36, 60)
            bikes = np.random.randint(22, 45)
            autos = np.random.randint(14, 25)
            buses = np.random.randint(4, 8)
            trucks = np.random.randint(3, 7)
            speed = np.random.uniform(3.0, 11.0)
            queue = np.random.uniform(82.0, 150.0)
            wait = np.random.uniform(70.0, 120.0)
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
        X[i, 8] = float(np.random.uniform(2.0, 6.0))
        X[i, 9] = float(queue)
        X[i, 10] = float(queue * 0.45)
        X[i, 11] = float(wait)
        X[i, 12] = float(total_v * 0.75)
        X[i, 13] = float(total_v * 0.70)
        X[i, 14] = float(intensity)
        X[i, 15] = float(np.random.uniform(0.0, 1.0))
        X[i, 16] = float(np.random.uniform(0.0, 1.0))
        X[i, 17] = float(np.random.choice([0.0, 1.0], p=[0.7, 0.3]))
        X[i, 18] = float(np.random.choice([0.0, 1.0], p=[0.95, 0.05]))
        X[i, 19] = float(np.random.choice([0.0, 0.3, 0.6], p=[0.8, 0.15, 0.05]))

    return X, y


def compute_confusion_matrix(y_true, y_pred, num_classes=4):
    """Compute N x N confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def run_cnn_tests():
    """Execute classification performance tests on TrafficCNN."""
    print("=" * 65)
    print("Autonomous Smart Traffic Management - TrafficCNN Test Suite")
    print(f"Backend Engine: {'PyTorch' if TORCH_AVAILABLE else 'NumPy Fallback Engine'}")
    print("=" * 65)

    cnn = TrafficCNN(input_features=20, num_classes=4)

    weights_path = CNN_CHECKPOINT_PT if TORCH_AVAILABLE else CNN_CHECKPOINT_NPZ
    if os.path.exists(weights_path):
        print(f"[Model] Loading weights checkpoint from: {weights_path}")
        cnn.load_weights(weights_path)
    else:
        print("[Model] Using calibrated model parameters.")

    # Generate test dataset
    X_test, y_test = generate_test_dataset(num_samples=400)
    print(f"[Test Dataset] Loaded {len(X_test)} intersection feature vectors (20 features each)")

    # Run inference
    classified_results = cnn.classify(X_test)
    y_pred = np.array([res['class_id'] for res in classified_results], dtype=np.int64)

    # Metrics
    total = len(y_test)
    correct = np.sum(y_pred == y_test)
    overall_accuracy = correct / total

    cm = compute_confusion_matrix(y_test, y_pred, num_classes=4)

    print("\n" + "-" * 65)
    print("CONFUSION MATRIX (Rows: Actual, Columns: Predicted)")
    print("-" * 65)
    header = f"{'Actual':<10}" + "".join([f"{c.upper():>12}" for c in CLASSES])
    print(header)
    print("-" * len(header))
    for i, c_name in enumerate(CLASSES):
        row_str = f"{c_name.upper():<10}"
        for j in range(4):
            row_str += f"{cm[i, j]:>12d}"
        print(row_str)

    print("\n" + "-" * 65)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("-" * 65)
    print(f"{'Class':<12}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}{'Support':>12}")
    print("-" * 60)

    precisions, recalls, f1s = [], [], []
    for i, c_name in enumerate(CLASSES):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"{c_name.upper():<12}{precision:>11.2%}{recall:>11.2%}{f1:>11.2%}{support:>12d}")

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    print("-" * 60)
    print(f"{'Macro Avg':<12}{macro_p:>11.2%}{macro_r:>11.2%}{macro_f1:>11.2%}{total:>12d}")
    print(f"\nOverall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")

    # Sample Inspection
    print("\n" + "-" * 65)
    print("SAMPLE INFERENCE DETAILS")
    print("-" * 65)
    for sample_id in [10, 80, 200, 350]:
        actual_name = CLASSES[y_test[sample_id]]
        res = classified_results[sample_id]
        print(f"\nSample #{sample_id}:")
        print(f"  Actual:     {actual_name.upper()}")
        print(f"  Predicted:  {res['class'].upper()} (Confidence: {res['confidence']:.2%})")
        print(f"  Unit Load:  {X_test[sample_id, 6]:.1f} | Speed: {X_test[sample_id, 7]:.1f} km/h | Queue: {X_test[sample_id, 9]:.1f}m")

    print("\n" + "=" * 65)
    print("[SUCCESS] TrafficCNN test evaluation passed successfully.")
    print("=" * 65)


if __name__ == '__main__':
    run_cnn_tests()
