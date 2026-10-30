"""
CongestionLSTM: LSTM Model for Time-Series Traffic Congestion Prediction.
Supports PyTorch (2-layer LSTM + FC) with a full NumPy fallback (weighted moving average).
"""
import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

CONGESTION_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8,
    'critical': 1.0,
}


def classify_congestion(value):
    """
    Classify a continuous congestion score [0.0, 1.0] into a categorical level.
    - low: < 0.3
    - medium: 0.3 - 0.6
    - high: 0.6 - 0.8
    - critical: >= 0.8
    """
    if isinstance(value, (list, np.ndarray)):
        return [classify_congestion(v) for v in value]
    val = float(value)
    if val < 0.3:
        return 'low'
    elif val < 0.6:
        return 'medium'
    elif val < 0.8:
        return 'high'
    else:
        return 'critical'


if TORCH_AVAILABLE:
    class CongestionLSTM(nn.Module):
        """
        PyTorch Implementation of CongestionLSTM with fallback support.
        Architecture: 2-layer LSTM (hidden_size=64, dropout=0.2) + FC layers.
        """
        def __init__(self, input_size=5, hidden_size=64, num_layers=2, forecast_horizon=4, dropout=0.2):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.forecast_horizon = forecast_horizon
            self.is_fallback = False

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, forecast_horizon),
                nn.Sigmoid()
            )

        def forward(self, x):
            """
            Forward pass.
            x shape: (batch_size, seq_len, input_size)
            Returns: (batch_size, forecast_horizon)
            """
            out, _ = self.lstm(x)
            last_step = out[:, -1, :]
            preds = self.fc(last_step)
            return preds

        def predict(self, sequence):
            """
            Inference on numpy array or torch tensor.
            sequence shape: (seq_len, input_size) or (batch_size, seq_len, input_size)
            Returns: numpy array of predictions shape (batch, forecast_horizon) or (forecast_horizon,)
            """
            self.eval()
            with torch.no_grad():
                seq_arr = np.asarray(sequence, dtype=np.float32)
                single_sample = (seq_arr.ndim == 2)
                if single_sample:
                    seq_arr = np.expand_dims(seq_arr, axis=0)

                tensor_x = torch.from_numpy(seq_arr)
                preds = self.forward(tensor_x).cpu().numpy()

                if single_sample:
                    return preds[0]
                return preds

        def save_weights(self, path):
            """Save model weights to file."""
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(self.state_dict(), path)
            print(f"[CongestionLSTM] Weights saved to: {path}")

        def load_weights(self, path):
            """Load model weights from file."""
            if os.path.exists(path):
                self.load_state_dict(torch.load(path, map_location='cpu'))
                self.eval()
                print(f"[CongestionLSTM] Weights loaded from: {path}")
            else:
                print(f"[CongestionLSTM] Warning: Weights file not found at {path}")

        @staticmethod
        def classify_level(val):
            return classify_congestion(val)

else:
    class CongestionLSTM:
        """
        NumPy-Only Fallback for CongestionLSTM.
        Implements recurrent prediction using weighted moving averages and trend momentum.
        """
        def __init__(self, input_size=5, hidden_size=64, num_layers=2, forecast_horizon=4, dropout=0.2):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.forecast_horizon = forecast_horizon
            self.dropout = dropout
            self.is_fallback = True
            self.feature_weights = np.array([0.35, -0.25, 0.25, 0.15, 0.30], dtype=np.float32)
            if input_size != 5:
                self.feature_weights = np.ones(input_size, dtype=np.float32) / input_size

        def forward(self, x):
            """
            NumPy recurrent prediction approximation.
            x: (batch, seq_len, input_size) or (seq_len, input_size)
            """
            arr = np.asarray(x, dtype=np.float32)
            single = (arr.ndim == 2)
            if single:
                arr = np.expand_dims(arr, axis=0)

            batch_size, seq_len, feat_dim = arr.shape
            preds = np.zeros((batch_size, self.forecast_horizon), dtype=np.float32)

            # Exponential decay weights for time steps (more weight to recent observations)
            time_weights = np.exp(np.linspace(-1.5, 0.0, seq_len))
            time_weights /= np.sum(time_weights)

            weights = self.feature_weights[:feat_dim]
            if len(weights) < feat_dim:
                weights = np.pad(weights, (0, feat_dim - len(weights)), 'mean')

            for b in range(batch_size):
                sample = arr[b]  # (seq_len, feat_dim)
                # Compute composite congestion score per time step
                # For speed (feat 1), inverted contribution is already in feature_weights
                step_scores = np.dot(sample, weights)
                # Normalize step scores roughly to [0, 1]
                step_scores = np.clip(step_scores, 0.0, 1.0)

                # Weighted moving average
                wma = np.sum(step_scores * time_weights)

                # Recent trend / slope
                recent_window = min(4, seq_len)
                if recent_window > 1:
                    recent = step_scores[-recent_window:]
                    slope = (recent[-1] - recent[0]) / (recent_window - 1)
                else:
                    slope = 0.0

                # Project forward across forecast_horizon
                for h in range(self.forecast_horizon):
                    step_ahead = h + 1
                    # Dampen trend projection over time
                    projected = wma + slope * step_ahead * (0.8 ** step_ahead)
                    preds[b, h] = np.clip(projected, 0.05, 0.98)

            if single:
                return preds[0]
            return preds

        def predict(self, sequence):
            """Predict next N congestion levels."""
            return self.forward(sequence)

        def save_weights(self, path):
            """Save fallback parameters."""
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            np.savez(path, feature_weights=self.feature_weights)
            print(f"[CongestionLSTM-NumPy] Weights saved to: {path}")

        def load_weights(self, path):
            """Load fallback parameters."""
            if os.path.exists(path):
                data = np.load(path)
                if 'feature_weights' in data:
                    self.feature_weights = data['feature_weights']
                print(f"[CongestionLSTM-NumPy] Weights loaded from: {path}")
            else:
                print(f"[CongestionLSTM-NumPy] Warning: File not found at {path}")

        @staticmethod
        def classify_level(val):
            return classify_congestion(val)


if __name__ == '__main__':
    print("=" * 60)
    print("Autonomous Smart Traffic Management - CongestionLSTM Demo")
    print(f"Backend: {'PyTorch' if TORCH_AVAILABLE else 'NumPy Fallback'}")
    print("=" * 60)

    model = CongestionLSTM(input_size=5, hidden_size=64, num_layers=2, forecast_horizon=4)

    # 12 time steps of 5 features: [vehicle_count, avg_speed, queue_len, wait_time, congestion_idx]
    # Simulate rising traffic: counts rising, speeds dropping, queue rising
    seq = np.zeros((12, 5), dtype=np.float32)
    for t in range(12):
        progress = t / 11.0
        seq[t, 0] = 0.3 + 0.5 * progress          # vehicle count
        seq[t, 1] = 0.8 - 0.5 * progress          # speed (decreasing)
        seq[t, 2] = 0.2 + 0.6 * progress          # queue length
        seq[t, 3] = 0.1 + 0.4 * progress          # wait time
        seq[t, 4] = 0.25 + 0.55 * progress        # prior congestion

    predictions = model.predict(seq)
    print("\nSimulated Input: 12-step sequence of escalating traffic congestion")
    print(f"Predicted Congestion for next 4 steps (1-4 min ahead):")
    for step_idx, pred_score in enumerate(predictions, 1):
        level = model.classify_level(pred_score)
        print(f"  Step +{step_idx}: score = {pred_score:.4f} -> Congestion Level: {level.upper()}")

    # Test weight saving / loading
    tmp_path = os.path.join(os.path.dirname(__file__), "lstm_test_weights.npz" if not TORCH_AVAILABLE else "lstm_test_weights.pth")
    model.save_weights(tmp_path)
    model.load_weights(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("\n[SUCCESS] CongestionLSTM verification complete.")
