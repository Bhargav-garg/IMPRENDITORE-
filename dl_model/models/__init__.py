"""
Deep Learning Models for Autonomous Smart Traffic Management.
- CongestionLSTM: LSTM for time-series congestion prediction
- TrafficCNN: CNN for traffic density classification
- VehicleDetector: YOLO-based vehicle detection and counting
"""

from .congestion_lstm import CongestionLSTM, classify_congestion
from .vehicle_cnn import TrafficCNN
from .yolo_detector import VehicleDetector


def get_lstm_model():
    return CongestionLSTM


def get_cnn_model():
    return TrafficCNN


def get_yolo_detector():
    return VehicleDetector


__all__ = [
    'CongestionLSTM',
    'TrafficCNN',
    'VehicleDetector',
    'classify_congestion',
    'get_lstm_model',
    'get_cnn_model',
    'get_yolo_detector',
]
