"""
Central Configuration - Autonomous Smart Traffic Management System.
All shared constants, paths, vehicle types, quantum params, and DL hyperparams.
"""
import os
from datetime import datetime

# ──────────────── Paths ────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
DL_MODEL_DIR = os.path.join(BASE_DIR, 'dl_model')

# ──────────────── Server ────────────────
HOST = '0.0.0.0'
PORT = 5000

# ──────────────── Google Maps ────────────────
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', 'AIzaSyAhxsKRb8xexsYYd2lRt7mJebkpYnnTLhs')

# ──────────────── Supported Cities ────────────────
SUPPORTED_CITIES = ['bhopal', 'gwalior', 'newdelhi', 'bangalore', 'chandigarh']
DEFAULT_CITY = 'bhopal'

# ──────────────── 3-Hour Time Slots ────────────────
# Each slot has a traffic intensity factor (0.0 = empty, 1.0 = maximum congestion)
TIME_SLOTS = {
    '00-03': {'label': 'Late Night',    'intensity': 0.15, 'hour_range': (0, 3)},
    '03-06': {'label': 'Early Morning', 'intensity': 0.20, 'hour_range': (3, 6)},
    '06-09': {'label': 'Morning Rush',  'intensity': 0.90, 'hour_range': (6, 9)},
    '09-12': {'label': 'Late Morning',  'intensity': 0.65, 'hour_range': (9, 12)},
    '12-15': {'label': 'Afternoon',     'intensity': 0.55, 'hour_range': (12, 15)},
    '15-18': {'label': 'Evening Rush',  'intensity': 0.95, 'hour_range': (15, 18)},
    '18-21': {'label': 'Evening',       'intensity': 0.70, 'hour_range': (18, 21)},
    '21-00': {'label': 'Night',         'intensity': 0.30, 'hour_range': (21, 24)},
}


def get_current_time_slot():
    """Returns (slot_id, slot_dict) for the current 3-hour window."""
    hour = datetime.now().hour
    for slot_id, slot in TIME_SLOTS.items():
        start, end = slot['hour_range']
        if start <= hour < end:
            return slot_id, slot
    return '21-00', TIME_SLOTS['21-00']


# ──────────────── Vehicle Types ────────────────
# weight = congestion contribution. Heavy vehicles >> light vehicles.
VEHICLE_TYPES = {
    'car':   {'weight': 1.0,  'length_m': 4.5,  'color': '#3498db', 'label': 'Car',   'size': 'light'},
    'bike':  {'weight': 0.5,  'length_m': 2.0,  'color': '#2ecc71', 'label': 'Bike',  'size': 'light'},
    'auto':  {'weight': 0.8,  'length_m': 3.0,  'color': '#f1c40f', 'label': 'Auto',  'size': 'light'},
    'bus':   {'weight': 3.0,  'length_m': 12.0, 'color': '#e67e22', 'label': 'Bus',   'size': 'heavy'},
    'truck': {'weight': 3.5,  'length_m': 15.0, 'color': '#e74c3c', 'label': 'Truck', 'size': 'heavy'},
}

EMERGENCY_TYPES = {
    'ambulance':  {'priority': 10, 'color': '#ff0000', 'label': 'Ambulance'},
    'firetruck':  {'priority': 9,  'color': '#ff4500', 'label': 'Fire Truck'},
}

# ──────────────── Quantum Engine ────────────────
NUM_QUBITS = 5             # One per intersection
QAOA_DEPTH = 1             # QAOA circuit depth (p=1)
QUEUE_WEIGHT = 1.0         # h_i linear term weight for queue length
WAIT_WEIGHT = 0.3          # h_i weight for accumulated wait time
WAVE_SYNC_WEIGHT = 0.5     # J_ij coupling for green wave sync
SPILLBACK_PENALTY = 2.0    # Penalty for downstream spillback
EMERGENCY_WEIGHT = 500.0   # Emergency vehicle preemption weight
GREEN_MIN_SEC = 8          # Minimum green phase duration
GREEN_MAX_SEC = 70         # Maximum green phase duration
YELLOW_SEC = 3             # Yellow clearance interval

# ──────────────── DL Model Hyperparameters ────────────────
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_SEQ_LENGTH = 12         # Input sequence length (time steps)
LSTM_FORECAST_HORIZON = 4   # Predict next N steps
CNN_INPUT_FEATURES = 20     # Feature vector dimension per intersection
CNN_NUM_CLASSES = 4          # Congestion levels: low, medium, high, critical
YOLO_CONFIDENCE = 0.5

CONGESTION_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8,
    'critical': 1.0,
}
