"""
Synthetic Traffic Data Generation Pipeline.
Generates realistic multi-modal traffic distributions across 5 Indian cities
(Bhopal, Gwalior, New Delhi, Bangalore, Chandigarh) following 3-hour time-slot intensities.
Produces 1-minute interval time-series over 30 days (43,200 steps) and 20-feature intersection vectors.
Saves data as both CSV and compressed NumPy arrays.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_DIR = os.path.join(PROJECT_ROOT, 'dl_model', 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'dl_model', 'data', 'processed')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

from dl_model.models.congestion_lstm import classify_congestion

# Supported Indian Cities and relative traffic density scale
CITIES = {
    'newdelhi':   {'name': 'New Delhi',   'scale': 1.40, 'capacity': 95},
    'bangalore':  {'name': 'Bangalore',   'scale': 1.35, 'capacity': 90},
    'bhopal':     {'name': 'Bhopal',      'scale': 0.95, 'capacity': 65},
    'gwalior':    {'name': 'Gwalior',     'scale': 0.85, 'capacity': 55},
    'chandigarh': {'name': 'Chandigarh',  'scale': 1.05, 'capacity': 70},
}

# 3-Hour Time Slots with characteristic intensity factors (0.0 to 1.0)
TIME_SLOTS = {
    '00-03': {'label': 'Late Night',    'intensity': 0.15, 'hours': (0, 3)},
    '03-06': {'label': 'Early Morning', 'intensity': 0.20, 'hours': (3, 6)},
    '06-09': {'label': 'Morning Rush',  'intensity': 0.90, 'hours': (6, 9)},
    '09-12': {'label': 'Late Morning',  'intensity': 0.65, 'hours': (9, 12)},
    '12-15': {'label': 'Afternoon',     'intensity': 0.55, 'hours': (12, 15)},
    '15-18': {'label': 'Evening Rush',  'intensity': 0.95, 'hours': (15, 18)},
    '18-21': {'label': 'Evening',       'intensity': 0.70, 'hours': (18, 21)},
    '21-00': {'label': 'Night',         'intensity': 0.30, 'hours': (21, 24)},
}

# Strict multi-modal vehicle distribution for Indian cities
VEHICLE_DISTRIBUTION = {
    'car':   {'pct': 0.60, 'weight': 1.0},
    'bike':  {'pct': 0.15, 'weight': 0.5},
    'auto':  {'pct': 0.10, 'weight': 0.8},
    'bus':   {'pct': 0.08, 'weight': 3.0},
    'truck': {'pct': 0.07, 'weight': 3.5},
}


def get_slot_info(hour):
    """Retrieve slot key and base intensity for a given hour."""
    for slot_id, info in TIME_SLOTS.items():
        h_start, h_end = info['hours']
        if h_start <= hour < h_end:
            return slot_id, info['intensity'], info['label']
    return '21-00', 0.30, 'Night'


def generate_timeseries_data(days=30, primary_city='bhopal'):
    """
    Generate 1-minute interval time series over specified number of days.
    Total records = days * 24 * 60 = 43,200 rows.
    """
    total_minutes = days * 24 * 60
    print(f"[Generate] Synthesizing {days}-day time series ({total_minutes:,} minutes) for '{primary_city}'...")

    start_date = datetime(2026, 3, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(minutes=m) for m in range(total_minutes)]

    city_cfg = CITIES[primary_city]
    city_scale = city_cfg['scale']
    capacity = city_cfg['capacity']

    # Vectorized hour and day of week
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps], dtype=np.float32)
    weekdays = np.array([ts.weekday() for ts in timestamps], dtype=np.int32)
    is_weekend = (weekdays >= 5).astype(np.float32)

    # Base intensity according to 3-hour slots with smooth sinusoidal transition
    slot_ids = []
    base_intensities = np.zeros(total_minutes, dtype=np.float32)
    for m in range(total_minutes):
        h = timestamps[m].hour
        s_id, s_int, _ = get_slot_info(h)
        slot_ids.append(s_id)
        base_intensities[m] = s_int

    # Weekend adjustments: weekend morning rush is flatter, afternoon is slightly higher
    intensity = base_intensities.copy()
    weekend_mask = is_weekend == 1.0
    intensity[weekend_mask] = intensity[weekend_mask] * 0.85 + 0.05

    # Add realistic high-frequency traffic noise (per-minute arrivals fluctuate)
    np.random.seed(42)
    traffic_noise = np.random.normal(0.0, 0.04, total_minutes).astype(np.float32)
    effective_intensity = np.clip(intensity * city_scale + traffic_noise, 0.08, 0.98)

    # Total vehicles passing intersection approach per minute
    base_vehicle_vol = (effective_intensity * capacity).astype(np.int32)
    base_vehicle_vol = np.maximum(base_vehicle_vol, 2)

    # Split into vehicle classes
    cars = np.round(base_vehicle_vol * VEHICLE_DISTRIBUTION['car']['pct']).astype(np.int32)
    bikes = np.round(base_vehicle_vol * VEHICLE_DISTRIBUTION['bike']['pct']).astype(np.int32)
    autos = np.round(base_vehicle_vol * VEHICLE_DISTRIBUTION['auto']['pct']).astype(np.int32)
    buses = np.round(base_vehicle_vol * VEHICLE_DISTRIBUTION['bus']['pct']).astype(np.int32)
    trucks = np.round(base_vehicle_vol * VEHICLE_DISTRIBUTION['truck']['pct']).astype(np.int32)

    total_vehicles = cars + bikes + autos + buses + trucks
    weighted_units = (
        cars * VEHICLE_DISTRIBUTION['car']['weight'] +
        bikes * VEHICLE_DISTRIBUTION['bike']['weight'] +
        autos * VEHICLE_DISTRIBUTION['auto']['weight'] +
        buses * VEHICLE_DISTRIBUTION['bus']['weight'] +
        trucks * VEHICLE_DISTRIBUTION['truck']['weight']
    )

    # Correlated speeds and queues
    # Free-flow speed ~50 km/h, congested ~8 km/h
    avg_speed = np.clip(52.0 - (effective_intensity * 44.0) + np.random.normal(0, 1.8, total_minutes), 5.0, 55.0)
    queue_len = np.clip((effective_intensity ** 1.3) * 135.0 + np.random.normal(0, 4.0, total_minutes), 0.0, 150.0)
    wait_sec = np.clip((effective_intensity ** 1.2) * 110.0 + np.random.normal(0, 3.5, total_minutes), 2.0, 120.0)

    # Congestion score and category
    congestion_scores = effective_intensity
    congestion_levels = [classify_congestion(s) for s in congestion_scores]

    df = pd.DataFrame({
        'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
        'city': primary_city,
        'time_slot': slot_ids,
        'slot_intensity': np.round(intensity, 3),
        'car_count': cars,
        'bike_count': bikes,
        'auto_count': autos,
        'bus_count': buses,
        'truck_count': trucks,
        'total_vehicles': total_vehicles,
        'weighted_units': np.round(weighted_units, 2),
        'avg_speed_kmh': np.round(avg_speed, 1),
        'queue_length_m': np.round(queue_len, 1),
        'wait_time_sec': np.round(wait_sec, 1),
        'congestion_score': np.round(congestion_scores, 4),
        'congestion_level': congestion_levels,
    })

    return df


def generate_intersection_feature_vectors(num_samples_per_city=600):
    """
    Generate 20-dimensional intersection feature vectors across all 5 Indian cities.
    Includes time slots, vehicle breakdown, speeds, queues, weather, and emergency flags.
    """
    print(f"[Generate] Creating intersection vectors across all 5 Indian cities ({num_samples_per_city} samples/city)...")
    np.random.seed(123)

    records = []
    slot_keys = list(TIME_SLOTS.keys())

    for city_key, city_meta in CITIES.items():
        city_scale = city_meta['scale']
        capacity = city_meta['capacity']

        for _ in range(num_samples_per_city):
            slot_id = np.random.choice(slot_keys)
            slot_info = TIME_SLOTS[slot_id]
            base_intensity = slot_info['intensity']

            hour_start, hour_end = slot_info['hours']
            hour = np.random.uniform(hour_start, hour_end)
            day_of_week = np.random.randint(0, 7)
            is_weekend = 1.0 if day_of_week >= 5 else 0.0

            intensity = base_intensity * (0.85 if is_weekend else 1.0)
            intensity = np.clip(intensity * city_scale + np.random.normal(0, 0.04), 0.05, 0.98)

            total_v = int(intensity * capacity + np.random.normal(0, 3))
            total_v = max(3, total_v)

            cars = int(round(total_v * VEHICLE_DISTRIBUTION['car']['pct']))
            bikes = int(round(total_v * VEHICLE_DISTRIBUTION['bike']['pct']))
            autos = int(round(total_v * VEHICLE_DISTRIBUTION['auto']['pct']))
            buses = int(round(total_v * VEHICLE_DISTRIBUTION['bus']['pct']))
            trucks = int(round(total_v * VEHICLE_DISTRIBUTION['truck']['pct']))

            weighted_units = (
                cars * 1.0 + bikes * 0.5 + autos * 0.8 +
                buses * 3.0 + trucks * 3.5
            )

            speed_kmh = float(np.clip(50.0 - intensity * 42.0 + np.random.normal(0, 2.0), 5.0, 55.0))
            speed_var = float(np.random.uniform(2.0, 7.0))
            queue_m = float(np.clip(intensity * 125.0 + np.random.normal(0, 4.0), 0.0, 140.0))
            max_queue_m = float(queue_m * np.random.uniform(0.35, 0.55))
            wait_s = float(np.clip(intensity * 95.0 + np.random.normal(0, 3.0), 2.0, 110.0))
            inflow = float(total_v * np.random.uniform(0.7, 0.9))
            outflow = float(total_v * np.random.uniform(0.65, 0.85))

            emergency_flag = 1.0 if np.random.rand() < 0.04 else 0.0
            weather_factor = float(np.random.choice([0.0, 0.3, 0.6], p=[0.85, 0.10, 0.05]))

            feat20 = [
                float(cars),
                float(bikes),
                float(autos),
                float(buses),
                float(trucks),
                float(total_v),
                round(weighted_units, 2),
                round(speed_kmh, 1),
                round(speed_var, 1),
                round(queue_m, 1),
                round(max_queue_m, 1),
                round(wait_s, 1),
                round(inflow, 1),
                round(outflow, 1),
                round(intensity, 3),
                round(hour / 24.0, 3),
                round(day_of_week / 7.0, 3),
                is_weekend,
                emergency_flag,
                weather_factor,
            ]

            level_str = classify_congestion(intensity)
            class_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

            records.append({
                'city': city_key,
                'time_slot': slot_id,
                'features': feat20,
                'congestion_score': round(float(intensity), 4),
                'congestion_level': level_str,
                'class_id': class_map[level_str]
            })

    return records


def run_synthetic_pipeline():
    """Execute complete synthetic data generation and export."""
    print("=" * 65)
    print("Autonomous Smart Traffic Management - Synthetic Data Generator")
    print(f"Target Cities: {', '.join([c.title() for c in CITIES.keys()])}")
    print(f"Vehicle Mix:   60% Car, 15% Bike, 10% Auto, 8% Bus, 7% Truck")
    print("=" * 65)

    # 1. 30-Day Time Series Generation
    df_ts = generate_timeseries_data(days=30, primary_city='bhopal')
    csv_ts_path = os.path.join(RAW_DIR, 'synthetic_traffic_30days.csv')
    df_ts.to_csv(csv_ts_path, index=False)
    print(f"[Export] Saved 30-day time series CSV ({len(df_ts):,} rows) to: {csv_ts_path}")

    # Extract 5 continuous time-series channels for LSTM sequence modeling
    # [total_vehicles_norm, speed_norm, queue_norm, wait_norm, congestion_score]
    v_norm = (df_ts['total_vehicles'].values - df_ts['total_vehicles'].min()) / max(1, (df_ts['total_vehicles'].max() - df_ts['total_vehicles'].min()))
    s_norm = (df_ts['avg_speed_kmh'].values - df_ts['avg_speed_kmh'].min()) / max(1, (df_ts['avg_speed_kmh'].max() - df_ts['avg_speed_kmh'].min()))
    q_norm = (df_ts['queue_length_m'].values - df_ts['queue_length_m'].min()) / max(1, (df_ts['queue_length_m'].max() - df_ts['queue_length_m'].min()))
    w_norm = (df_ts['wait_time_sec'].values - df_ts['wait_time_sec'].min()) / max(1, (df_ts['wait_time_sec'].max() - df_ts['wait_time_sec'].min()))
    c_norm = df_ts['congestion_score'].values

    ts_features = np.column_stack([v_norm, s_norm, q_norm, w_norm, c_norm]).astype(np.float32)
    npz_ts_path = os.path.join(PROCESSED_DIR, 'synthetic_traffic_timeseries.npz')
    np.savez_compressed(npz_ts_path, features=ts_features, timestamps=df_ts['timestamp'].values)
    print(f"[Export] Saved NumPy time-series array {ts_features.shape} to: {npz_ts_path}")

    # 2. 20-Feature Intersection Vectors across all 5 cities
    intersection_records = generate_intersection_feature_vectors(num_samples_per_city=600)
    total_vectors = len(intersection_records)

    feature_matrix = np.array([r['features'] for r in intersection_records], dtype=np.float32)
    class_ids = np.array([r['class_id'] for r in intersection_records], dtype=np.int64)
    scores = np.array([r['congestion_score'] for r in intersection_records], dtype=np.float32)
    city_list = np.array([r['city'] for r in intersection_records])

    # Flatten feature matrix to CSV columns
    feat_cols = [
        'car_count', 'bike_count', 'auto_count', 'bus_count', 'truck_count',
        'total_vehicles', 'weighted_units', 'avg_speed_kmh', 'speed_var',
        'queue_length_m', 'max_approach_queue_m', 'wait_time_sec', 'inflow_rate',
        'outflow_rate', 'slot_intensity', 'hour_norm', 'day_norm',
        'is_weekend', 'emergency_flag', 'weather_factor'
    ]
    df_cnn = pd.DataFrame(feature_matrix, columns=feat_cols)
    df_cnn['city'] = city_list
    df_cnn['congestion_score'] = scores
    df_cnn['class_id'] = class_ids
    df_cnn['congestion_level'] = [r['congestion_level'] for r in intersection_records]

    csv_cnn_path = os.path.join(RAW_DIR, 'synthetic_intersection_features.csv')
    df_cnn.to_csv(csv_cnn_path, index=False)
    print(f"[Export] Saved intersection feature vectors CSV ({total_vectors:,} rows) to: {csv_cnn_path}")

    npz_cnn_path = os.path.join(PROCESSED_DIR, 'synthetic_intersection_features.npz')
    np.savez_compressed(npz_cnn_path, X=feature_matrix, y=class_ids, scores=scores)
    print(f"[Export] Saved NumPy intersection feature array {feature_matrix.shape} to: {npz_cnn_path}")

    print("\n" + "=" * 65)
    print("DATASET GENERATION SUMMARY:")
    print(f"  Time Series (30 Days @ 1-min): {len(df_ts):,} entries")
    print(f"  Intersection Vectors:          {total_vectors:,} samples (5 cities x 600)")
    print("  Class Distribution in Intersection Vectors:")
    for c_id, c_name in enumerate(['low', 'medium', 'high', 'critical']):
        c_count = int(np.sum(class_ids == c_id))
        print(f"    - {c_name.upper():<8}: {c_count:>4d} ({c_count / total_vectors:.1%})")
    print("=" * 65)
    print("[SUCCESS] Synthetic traffic data generation complete.")


if __name__ == '__main__':
    run_synthetic_pipeline()
