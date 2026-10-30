"""
Data Preprocessing Pipeline for Traffic Prediction and Classification.
Normalizes features, generates sliding window sequences for LSTM (seq_len=12, horizon=4),
constructs 20-feature vectors for CNN density classification, splits datasets,
and saves processed arrays in dl_model/data/processed/.
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dl_model.models.congestion_lstm import classify_congestion

RAW_DIR = os.path.join(PROJECT_ROOT, 'dl_model', 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'dl_model', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


class MinMaxScalerNP:
    """Pure NumPy Min-Max Feature Scaler [0, 1]."""
    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        # Avoid division by zero for constant features
        diff = self.data_max - self.data_min
        diff[diff == 0.0] = 1.0
        self.diff = diff
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        scale = self.feature_range[1] - self.feature_range[0]
        return self.feature_range[0] + scale * (X - self.data_min) / self.diff

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def load_raw_traffic_data():
    """
    Load raw traffic observations from CSV or HDF5 files in RAW_DIR.
    Checks synthetic datasets as well as PeMS-BAY and METR-LA.
    """
    synthetic_ts_path = os.path.join(RAW_DIR, 'synthetic_traffic_30days.csv')
    pems_path = os.path.join(RAW_DIR, 'pems_bay.csv')
    metr_path = os.path.join(RAW_DIR, 'metr_la.csv')

    dfs = []
    if os.path.exists(synthetic_ts_path):
        print(f"[Preprocess] Loading 30-day synthetic traffic series: {synthetic_ts_path}")
        try:
            df_syn = pd.read_csv(synthetic_ts_path)
            dfs.append(('synthetic_ts', df_syn))
        except Exception as e:
            print(f"[Preprocess] Error loading synthetic_traffic_30days: {e}")

    if os.path.exists(pems_path):
        print(f"[Preprocess] Loading PeMS-BAY data: {pems_path}")
        try:
            df_pems = pd.read_csv(pems_path)
            dfs.append(('pems_bay', df_pems))
        except Exception as e:
            print(f"[Preprocess] Error loading pems_bay: {e}")

    if os.path.exists(metr_path):
        print(f"[Preprocess] Loading METR-LA data: {metr_path}")
        try:
            df_metr = pd.read_csv(metr_path)
            dfs.append(('metr_la', df_metr))
        except Exception as e:
            print(f"[Preprocess] Error loading metr_la: {e}")

    return dfs


def create_sliding_window_lstm(time_series_features, window_size=12, horizon=4):
    """
    Construct sliding windows:
    X shape: (num_samples, window_size, num_features)
    y shape: (num_samples, horizon)
    """
    X_list, y_list = [], []
    N = len(time_series_features)

    for i in range(N - window_size - horizon + 1):
        x_window = time_series_features[i : i + window_size]
        # Target: congestion index (feature index 4, or composite) over next 'horizon' steps
        if time_series_features.shape[1] >= 5:
            y_window = time_series_features[i + window_size : i + window_size + horizon, 4]
        else:
            y_window = time_series_features[i + window_size : i + window_size + horizon, 0]

        X_list.append(x_window)
        y_list.append(y_window)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def create_cnn_features_from_series(time_series_features, window_size=12):
    """
    Transform rolling windows of traffic data into 20-dimensional intersection feature vectors
    and 4-class categorical labels.
    """
    X_cnn = []
    y_cnn = []
    N = len(time_series_features)

    for i in range(0, N - window_size, max(1, window_size // 2)):
        window = time_series_features[i : i + window_size]

        # Extract summary features
        mean_v = np.mean(window[:, 0]) if window.shape[1] > 0 else 0.5
        mean_speed = np.mean(window[:, 1]) if window.shape[1] > 1 else 0.5
        speed_var = np.var(window[:, 1]) if window.shape[1] > 1 else 0.1
        mean_queue = np.mean(window[:, 2]) if window.shape[1] > 2 else 0.5
        max_queue = np.max(window[:, 2]) if window.shape[1] > 2 else 0.5
        mean_wait = np.mean(window[:, 3]) if window.shape[1] > 3 else 0.5
        congestion_val = np.mean(window[:, 4]) if window.shape[1] > 4 else mean_v

        # Reconstruct vehicle breakdown consistent with distribution
        total_vehicles = mean_v * 60.0
        cars = total_vehicles * 0.60
        bikes = total_vehicles * 0.15
        autos = total_vehicles * 0.10
        buses = total_vehicles * 0.08
        trucks = total_vehicles * 0.07

        weighted_units = cars * 1.0 + bikes * 0.5 + autos * 0.8 + buses * 3.0 + trucks * 3.5
        speed_kmh = (1.0 - mean_speed) * 45.0 + 10.0
        queue_m = mean_queue * 120.0
        wait_s = mean_wait * 90.0

        feat20 = np.zeros(20, dtype=np.float32)
        feat20[0] = cars
        feat20[1] = bikes
        feat20[2] = autos
        feat20[3] = buses
        feat20[4] = trucks
        feat20[5] = total_vehicles
        feat20[6] = weighted_units
        feat20[7] = speed_kmh
        feat20[8] = speed_var * 15.0
        feat20[9] = queue_m
        feat20[10] = max_queue * 60.0
        feat20[11] = wait_s
        feat20[12] = total_vehicles * 0.80
        feat20[13] = total_vehicles * 0.75
        feat20[14] = congestion_val
        feat20[15] = (i % 1440) / 1440.0
        feat20[16] = ((i // 1440) % 7) / 7.0
        feat20[17] = 1.0 if ((i // 1440) % 7) >= 5 else 0.0
        feat20[18] = 1.0 if np.random.rand() > 0.95 else 0.0
        feat20[19] = 0.0

        # Assign class based on congestion value
        lvl_str = classify_congestion(congestion_val)
        class_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        class_id = class_mapping[lvl_str]

        X_cnn.append(feat20)
        y_cnn.append(class_id)

    return np.array(X_cnn, dtype=np.float32), np.array(y_cnn, dtype=np.int64)


def split_data(X, y, train_ratio=0.70, val_ratio=0.15):
    """Split into train, val, and test subsets."""
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        'X_train': X[:n_train],
        'y_train': y[:n_train],
        'X_val': X[n_train:n_train + n_val],
        'y_val': y[n_train:n_train + n_val],
        'X_test': X[n_train + n_val:],
        'y_test': y[n_train + n_val:],
    }


def run_preprocessing_pipeline():
    """Execute complete data preprocessing pipeline."""
    print("=" * 65)
    print("Autonomous Smart Traffic Management - Data Preprocessing Pipeline")
    print(f"Processed Directory: {PROCESSED_DIR}")
    print("=" * 65)

    raw_dfs = load_raw_traffic_data()

    # If raw data has time series (like our synthesized pems_bay), extract features
    combined_series = []
    for tag, df in raw_dfs:
        if 'total_vehicles' in df.columns and 'avg_speed_kmh' in df.columns:
            print(f"[Preprocess] Extracting feature series from {tag} ({len(df)} rows)...")
            v_norm = (df['total_vehicles'] - df['total_vehicles'].min()) / max(1, (df['total_vehicles'].max() - df['total_vehicles'].min()))
            s_norm = (df['avg_speed_kmh'] - df['avg_speed_kmh'].min()) / max(1, (df['avg_speed_kmh'].max() - df['avg_speed_kmh'].min()))
            q_norm = (df['queue_length_m'] - df['queue_length_m'].min()) / max(1, (df['queue_length_m'].max() - df['queue_length_m'].min()))
            w_norm = (df['wait_time_sec'] - df['wait_time_sec'].min()) / max(1, (df['wait_time_sec'].max() - df['wait_time_sec'].min()))
            cong_norm = df['congestion_score'].values
            series = np.column_stack([v_norm, s_norm, q_norm, w_norm, cong_norm])
            combined_series.append(series)
        elif 'flow' in df.columns and 'speed_mph' in df.columns:
            print(f"[Preprocess] Extracting feature series from {tag} ({len(df)} rows)...")
            flow_norm = (df['flow'] - df['flow'].min()) / max(1, (df['flow'].max() - df['flow'].min()))
            speed_norm = (df['speed_mph'] - df['speed_mph'].min()) / max(1, (df['speed_mph'].max() - df['speed_mph'].min()))
            occ_norm = df['occupancy'] if 'occupancy' in df.columns else flow_norm * 0.5
            queue_norm = np.clip(flow_norm * 1.1 - speed_norm * 0.3, 0.0, 1.0)
            wait_norm = np.clip(queue_norm * 0.9, 0.0, 1.0)
            cong_idx = np.clip(0.4 * flow_norm + 0.3 * (1.0 - speed_norm) + 0.3 * occ_norm, 0.05, 0.98)

            series = np.column_stack([flow_norm, speed_norm, queue_norm, wait_norm, cong_idx])
            combined_series.append(series)

    if combined_series:
        all_features = np.vstack(combined_series)
    else:
        print("[Preprocess] No raw time-series CSV found. Generating synthetic base series...")
        np.random.seed(500)
        n_steps = 2880
        t = np.linspace(0, 10 * np.pi, n_steps)
        flow = 0.5 + 0.4 * np.sin(t) + np.random.normal(0, 0.05, n_steps)
        flow = np.clip(flow, 0.05, 0.98)
        speed = np.clip(1.0 - flow + np.random.normal(0, 0.03, n_steps), 0.05, 0.98)
        queue = np.clip(flow * 1.1, 0.0, 1.0)
        wait = np.clip(flow * 0.9, 0.0, 1.0)
        cong = flow
        all_features = np.column_stack([flow, speed, queue, wait, cong])

    scaler = MinMaxScalerNP(feature_range=(0.0, 1.0))
    normalized_features = scaler.fit_transform(all_features)
    print(f"[Preprocess] Normalized {len(normalized_features)} time steps across 5 traffic features.")

    # 1. Prepare LSTM sliding window data (window=12, horizon=4)
    X_lstm, y_lstm = create_sliding_window_lstm(normalized_features, window_size=12, horizon=4)
    lstm_splits = split_data(X_lstm, y_lstm, train_ratio=0.70, val_ratio=0.15)

    lstm_save_path = os.path.join(PROCESSED_DIR, 'lstm_data.npz')
    np.savez(
        lstm_save_path,
        X=X_lstm, y=y_lstm,
        X_train=lstm_splits['X_train'], y_train=lstm_splits['y_train'],
        X_val=lstm_splits['X_val'], y_val=lstm_splits['y_val'],
        X_test=lstm_splits['X_test'], y_test=lstm_splits['y_test']
    )
    print(f"[Preprocess] Saved LSTM dataset: {X_lstm.shape} -> {y_lstm.shape} to {lstm_save_path}")

    # 2. Prepare CNN intersection vectors (features=20, classes=4)
    X_cnn, y_cnn = create_cnn_features_from_series(all_features, window_size=12)
    cnn_splits = split_data(X_cnn, y_cnn, train_ratio=0.70, val_ratio=0.15)

    cnn_save_path = os.path.join(PROCESSED_DIR, 'cnn_data.npz')
    np.savez(
        cnn_save_path,
        X=X_cnn, y=y_cnn,
        X_train=cnn_splits['X_train'], y_train=cnn_splits['y_train'],
        X_val=cnn_splits['X_val'], y_val=cnn_splits['y_val'],
        X_test=cnn_splits['X_test'], y_test=cnn_splits['y_test']
    )
    print(f"[Preprocess] Saved CNN dataset:  {X_cnn.shape} -> {y_cnn.shape} to {cnn_save_path}")

    # 3. Save scaler parameters
    scaler_save_path = os.path.join(PROCESSED_DIR, 'scaler_params.npz')
    np.savez(scaler_save_path, data_min=scaler.data_min, data_max=scaler.data_max, diff=scaler.diff)
    print(f"[Preprocess] Saved Normalization parameters to: {scaler_save_path}")

    print("\n" + "=" * 65)
    print("SUMMARY OF PROCESSED DATASETS:")
    print(f"  LSTM Total Sequences: {len(X_lstm)} (Window=12, Horizon=4, Features=5)")
    print(f"  LSTM Train: {len(lstm_splits['X_train'])} | Val: {len(lstm_splits['X_val'])} | Test: {len(lstm_splits['X_test'])}")
    print(f"  CNN Total Vectors:    {len(X_cnn)} (Input Features=20, Num Classes=4)")
    print(f"  CNN Train:  {len(cnn_splits['X_train'])} | Val: {len(cnn_splits['X_val'])} | Test: {len(cnn_splits['X_test'])}")
    print("=" * 65)
    print("[SUCCESS] Data preprocessing completed successfully.")


if __name__ == '__main__':
    run_preprocessing_pipeline()
