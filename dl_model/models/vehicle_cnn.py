"""
TrafficCNN: 1D Convolutional Neural Network for Traffic Density Classification.
Classifies intersection feature vectors (dimension=20) into 4 congestion classes:
'low', 'medium', 'high', 'critical'.
Provides full PyTorch implementation with an intelligent NumPy fallback.
"""
import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    F = None

CLASSES = ['low', 'medium', 'high', 'critical']
CLASS_TO_IDX = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
IDX_TO_CLASS = {0: 'low', 1: 'medium', 2: 'high', 3: 'critical'}


if TORCH_AVAILABLE:
    class TrafficCNN(nn.Module):
        """
        PyTorch Implementation of TrafficCNN.
        Architecture: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d -> Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d -> FC.
        """
        def __init__(self, input_features=20, num_classes=4):
            super().__init__()
            self.input_features = input_features
            self.num_classes = num_classes
            self.is_fallback = False

            self.conv_net = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),  # 20 -> 10

                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),  # 10 -> 5
            )
            self.classifier = nn.Sequential(
                nn.Linear(32 * 5, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            """
            Forward pass.
            x shape: (batch_size, 20) or (batch_size, 1, 20)
            Returns: logits of shape (batch_size, num_classes)
            """
            if x.ndim == 2:
                x = x.unsqueeze(1)
            elif x.ndim == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            feat = self.conv_net(x)
            flat = feat.view(feat.size(0), -1)
            logits = self.classifier(flat)
            return logits

        def classify(self, features):
            """
            Classify feature vector(s) into congestion categories with confidence scores.
            features: (20,) or (batch_size, 20)
            Returns: dict (single input) or list of dicts (batch input)
            """
            self.eval()
            with torch.no_grad():
                arr = np.asarray(features, dtype=np.float32)
                single = (arr.ndim == 1)
                if single:
                    arr = np.expand_dims(arr, axis=0)

                tensor_x = torch.from_numpy(arr)
                logits = self.forward(tensor_x)
                probs = F.softmax(logits, dim=1).cpu().numpy()

                results = []
                for p in probs:
                    top_idx = int(np.argmax(p))
                    cls_name = IDX_TO_CLASS[top_idx]
                    prob_dict = {CLASSES[i]: float(p[i]) for i in range(self.num_classes)}
                    results.append({
                        'class': cls_name,
                        'class_id': top_idx,
                        'confidence': float(p[top_idx]),
                        'probabilities': prob_dict
                    })

                if single:
                    return results[0]
                return results

        def save_weights(self, path):
            """Save PyTorch weights."""
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(self.state_dict(), path)
            print(f"[TrafficCNN] Weights saved to: {path}")

        def load_weights(self, path):
            """Load PyTorch weights."""
            if os.path.exists(path):
                self.load_state_dict(torch.load(path, map_location='cpu'))
                self.eval()
                print(f"[TrafficCNN] Weights loaded from: {path}")
            else:
                print(f"[TrafficCNN] Warning: Weights file not found at {path}")

else:
    class TrafficCNN:
        """
        NumPy Fallback for TrafficCNN.
        Implements weighted density estimation and statistical Bayesian classification.
        """
        def __init__(self, input_features=20, num_classes=4):
            self.input_features = input_features
            self.num_classes = num_classes
            self.is_fallback = True
            # Class center prototypes in normalized congestion space [0, 1]
            self.centers = np.array([0.15, 0.45, 0.70, 0.90], dtype=np.float32)
            self.spread = 0.14

        def _compute_score(self, feat):
            """Compute continuous congestion index from 20-dim feature vector."""
            feat = np.asarray(feat, dtype=np.float32)
            # Standard feature layout:
            # 0: car, 1: bike, 2: auto, 3: bus, 4: truck, 5: total, 6: weighted_units
            # 7: avg_speed, 8: speed_var, 9: queue_len, 10: max_queue, 11: wait_time
            # 12: inflow, 13: outflow, 14: slot_intensity, 15: hour_norm, 16: day_norm, 17: weekend
            # 18: emergency_flag, 19: weather_factor
            if len(feat) >= 7 and feat[6] > 0:
                weighted_units = feat[6]
            elif len(feat) >= 5:
                weighted_units = (feat[0] * 1.0 + feat[1] * 0.5 + feat[2] * 0.8 +
                                  feat[3] * 3.0 + feat[4] * 3.5)
            else:
                weighted_units = np.mean(feat) * 30.0

            # Normalization scale: ~50 weighted units is high congestion
            density_score = np.clip(weighted_units / 55.0, 0.0, 1.0)

            speed_factor = 0.5
            if len(feat) > 7:
                # speed in km/h (0 to 60+), lower speed -> higher congestion
                speed = feat[7]
                speed_score = np.clip(1.0 - (speed / 50.0), 0.0, 1.0)
                speed_factor = speed_score

            queue_factor = 0.5
            if len(feat) > 9:
                queue_m = feat[9]
                queue_factor = np.clip(queue_m / 120.0, 0.0, 1.0)

            composite = 0.50 * density_score + 0.30 * speed_factor + 0.20 * queue_factor
            return float(np.clip(composite, 0.02, 0.98))

        def forward(self, x):
            """Returns pseudo-logits for compatibility with training/eval loops."""
            arr = np.asarray(x, dtype=np.float32)
            single = (arr.ndim == 1)
            if single:
                arr = np.expand_dims(arr, axis=0)
            logits = np.zeros((arr.shape[0], self.num_classes), dtype=np.float32)
            for i, row in enumerate(arr):
                score = self._compute_score(row)
                dists = -((score - self.centers) ** 2) / (2 * (self.spread ** 2))
                logits[i] = dists
            if single:
                return logits[0]
            return logits

        def classify(self, features):
            """
            Classify feature vector(s) into congestion categories with confidence scores.
            features: (20,) or (batch_size, 20)
            """
            arr = np.asarray(features, dtype=np.float32)
            single = (arr.ndim == 1)
            if single:
                arr = np.expand_dims(arr, axis=0)

            results = []
            for feat in arr:
                score = self._compute_score(feat)
                # Compute Gaussian kernel distances to class centers
                diffs = -((score - self.centers) ** 2) / (2 * (self.spread ** 2))
                exp_diffs = np.exp(diffs - np.max(diffs))
                probs = exp_diffs / np.sum(exp_diffs)

                top_idx = int(np.argmax(probs))
                cls_name = IDX_TO_CLASS[top_idx]
                prob_dict = {CLASSES[i]: float(probs[i]) for i in range(self.num_classes)}
                results.append({
                    'class': cls_name,
                    'class_id': top_idx,
                    'confidence': float(probs[top_idx]),
                    'score': score,
                    'probabilities': prob_dict
                })

            if single:
                return results[0]
            return results

        def save_weights(self, path):
            """Save fallback parameters."""
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            np.savez(path, centers=self.centers, spread=np.array([self.spread]))
            print(f"[TrafficCNN-NumPy] Weights saved to: {path}")

        def load_weights(self, path):
            """Load fallback parameters."""
            if os.path.exists(path):
                data = np.load(path)
                if 'centers' in data:
                    self.centers = data['centers']
                if 'spread' in data:
                    self.spread = float(data['spread'][0])
                print(f"[TrafficCNN-NumPy] Weights loaded from: {path}")
            else:
                print(f"[TrafficCNN-NumPy] Warning: File not found at {path}")


if __name__ == '__main__':
    print("=" * 60)
    print("Autonomous Smart Traffic Management - TrafficCNN Demo")
    print(f"Backend: {'PyTorch' if TORCH_AVAILABLE else 'NumPy Fallback'}")
    print("=" * 60)

    cnn = TrafficCNN(input_features=20, num_classes=4)

    # Test sample 1: Heavy Congestion
    # 25 cars, 10 bikes, 8 autos, 4 buses, 3 trucks -> heavy units, slow speed (8 km/h), long queue (105m)
    heavy_features = np.zeros(20, dtype=np.float32)
    heavy_features[0] = 25.0    # cars
    heavy_features[1] = 10.0    # bikes
    heavy_features[2] = 8.0     # autos
    heavy_features[3] = 4.0     # buses
    heavy_features[4] = 3.0     # trucks
    heavy_features[5] = 50.0    # total vehicles
    heavy_features[6] = 25.0*1.0 + 10.0*0.5 + 8.0*0.8 + 4.0*3.0 + 3.0*3.5  # 58.9 weighted units
    heavy_features[7] = 8.5     # avg speed km/h
    heavy_features[8] = 3.2     # speed var
    heavy_features[9] = 110.0   # queue length m
    heavy_features[10] = 55.0   # max approach queue
    heavy_features[11] = 68.0   # waiting time s
    heavy_features[14] = 0.95   # evening rush intensity

    res_heavy = cnn.classify(heavy_features)
    print("\nTest Case 1: Rush Hour Intersection (50 vehicles, 8.5 km/h, 110m queue)")
    print(f"  Predicted Class: {res_heavy['class'].upper()}")
    print(f"  Confidence:     {res_heavy['confidence']:.2%}")
    print(f"  Probabilities:  {', '.join(f'{k}: {v:.3f}' for k, v in res_heavy['probabilities'].items())}")

    # Test sample 2: Low Traffic
    # 3 cars, 2 bikes, 0 buses, 0 trucks -> speed 45 km/h, queue 5m
    light_features = np.zeros(20, dtype=np.float32)
    light_features[0] = 3.0
    light_features[1] = 2.0
    light_features[5] = 5.0
    light_features[6] = 4.0
    light_features[7] = 45.0
    light_features[9] = 5.0
    light_features[14] = 0.20

    res_light = cnn.classify(light_features)
    print("\nTest Case 2: Midnight Quiet Intersection (5 vehicles, 45 km/h, 5m queue)")
    print(f"  Predicted Class: {res_light['class'].upper()}")
    print(f"  Confidence:     {res_light['confidence']:.2%}")
    print(f"  Probabilities:  {', '.join(f'{k}: {v:.3f}' for k, v in res_light['probabilities'].items())}")

    # Test weight saving / loading
    tmp_path = os.path.join(os.path.dirname(__file__), "cnn_test_weights.npz" if not TORCH_AVAILABLE else "cnn_test_weights.pth")
    cnn.save_weights(tmp_path)
    cnn.load_weights(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print("\n[SUCCESS] TrafficCNN verification complete.")
