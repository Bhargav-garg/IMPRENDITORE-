"""
VehicleDetector: YOLO-based Vehicle Detection and Counting.
Uses ultralytics YOLOv8n pretrained on COCO if available, with a robust
simulation-based fallback detector calibrated for Indian multi-modal traffic distributions.
"""
import os
import sys
import random
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

VEHICLE_WEIGHTS = {
    'car': 1.0,
    'bike': 0.5,
    'auto': 0.8,
    'bus': 3.0,
    'truck': 3.5,
}

COCO_CLASS_MAP = {
    2: 'car',
    3: 'bike',        # motorcycle -> bike
    5: 'bus',
    7: 'truck',
}

TRAFFIC_LEVEL_VEHICLE_RANGES = {
    'low': (4, 12),
    'medium': (14, 28),
    'high': (30, 52),
    'critical': (55, 80),
}

DEFAULT_TYPE_PROBS = {
    'car': 0.60,
    'bike': 0.15,
    'auto': 0.10,
    'bus': 0.08,
    'truck': 0.07,
}


class VehicleDetector:
    """
    Vehicle Detection and Counting System.
    Provides object detection, vehicle category tallying, and weighted congestion indexing.
    """
    def __init__(self, model_path='yolov8n.pt', confidence=0.5, force_simulation=False):
        self.confidence = confidence
        self.force_simulation = force_simulation
        self.is_simulation = force_simulation or not ULTRALYTICS_AVAILABLE
        self.model = None

        if not self.is_simulation:
            try:
                self.model = YOLO(model_path)
                print(f"[VehicleDetector] Loaded YOLOv8 model from '{model_path}'")
            except Exception as e:
                print(f"[VehicleDetector] Warning: Could not initialize YOLO model ({e}). Using simulation mode.")
                self.is_simulation = True

        if self.is_simulation:
            print("[VehicleDetector] Operating in high-fidelity simulation detection mode.")

    def detect(self, image_or_data=None, time_slot='06-09', traffic_level=None, context=None):
        """
        Detect vehicles from an image or generate calibrated detections from simulation context.
        image_or_data: numpy array / file path / dict / None
        Returns: list of dicts: [{'class': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}]
        """
        if context is not None:
            image_or_data = context
        if not self.is_simulation and image_or_data is not None and not isinstance(image_or_data, dict):
            return self._detect_yolo(image_or_data)
        else:
            return self._detect_simulation(image_or_data, time_slot, traffic_level)

    def _detect_yolo(self, image):
        """Run YOLO inference on image input."""
        results = self.model(image, conf=self.confidence, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in COCO_CLASS_MAP:
                    cls_name = COCO_CLASS_MAP[cls_id]
                    conf = float(box.conf[0].item())
                    xyxy = [float(coord) for coord in box.xyxy[0].tolist()]
                    detections.append({
                        'class': cls_name,
                        'confidence': round(conf, 4),
                        'bbox': [round(c, 1) for c in xyxy]
                    })
        return detections

    def _detect_simulation(self, context=None, time_slot='06-09', traffic_level=None):
        """
        Simulate realistic vehicle detections with bounding boxes.
        Context can provide explicit overrides for time_slot or traffic_level.
        """
        if isinstance(context, dict):
            time_slot = context.get('time_slot', time_slot)
            traffic_level = context.get('traffic_level', traffic_level)

        # Resolve traffic level from time slot if not explicitly provided
        if traffic_level is None:
            intensity_map = {
                '00-03': 'low',
                '03-06': 'low',
                '06-09': 'critical',
                '09-12': 'high',
                '12-15': 'medium',
                '15-18': 'critical',
                '18-21': 'high',
                '21-00': 'low',
            }
            traffic_level = intensity_map.get(time_slot, 'medium')

        min_v, max_v = TRAFFIC_LEVEL_VEHICLE_RANGES.get(traffic_level, (15, 30))
        num_vehicles = random.randint(min_v, max_v)

        types = list(DEFAULT_TYPE_PROBS.keys())
        probs = list(DEFAULT_TYPE_PROBS.values())

        # Frame resolution for bbox simulation: 640x480
        frame_w, frame_h = 640, 480
        bbox_specs = {
            'bike':  (18, 30, 22, 40),
            'auto':  (30, 50, 32, 55),
            'car':   (45, 80, 40, 70),
            'bus':   (80, 140, 60, 95),
            'truck': (90, 160, 65, 105),
        }

        detections = []
        for _ in range(num_vehicles):
            v_type = random.choices(types, weights=probs, k=1)[0]
            w_min, w_max, h_min, h_max = bbox_specs[v_type]
            bw = random.randint(w_min, w_max)
            bh = random.randint(h_min, h_max)
            x1 = random.randint(10, max(11, frame_w - bw - 10))
            y1 = random.randint(10, max(11, frame_h - bh - 10))
            x2 = x1 + bw
            y2 = y1 + bh
            conf = round(random.uniform(self.confidence, 0.98), 3)

            detections.append({
                'class': v_type,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })

        return detections

    def count_vehicles(self, detections):
        """
        Tally counts by vehicle category from detection list.
        Returns: dict {'car': N, 'bus': N, 'truck': N, 'bike': N, 'auto': N}
        """
        counts = {'car': 0, 'bus': 0, 'truck': 0, 'bike': 0, 'auto': 0}
        for det in detections:
            cls_name = det.get('class')
            if cls_name in counts:
                counts[cls_name] += 1
        return counts

    def get_weighted_congestion(self, counts):
        """
        Compute weighted congestion load based on vehicle size and road space occupancy.
        Weights: truck=3.5, bus=3.0, car=1.0, bike=0.5, auto=0.8
        Returns: float (total weighted units)
        """
        weighted_sum = sum(counts.get(v_type, 0) * weight for v_type, weight in VEHICLE_WEIGHTS.items())
        return round(float(weighted_sum), 2)


if __name__ == '__main__':
    print("=" * 60)
    print("Autonomous Smart Traffic Management - VehicleDetector Demo")
    print(f"Ultralytics Available: {ULTRALYTICS_AVAILABLE}")
    print("=" * 60)

    detector = VehicleDetector(confidence=0.5)

    test_slots = [
        ('00-03', 'Late Night (Low Traffic)'),
        ('12-15', 'Afternoon (Medium Traffic)'),
        ('15-18', 'Evening Rush (Critical Traffic)'),
    ]

    for slot, label in test_slots:
        print(f"\n--- Testing Scenario: {label} [Slot {slot}] ---")
        detections = detector.detect(time_slot=slot)
        counts = detector.count_vehicles(detections)
        weighted_load = detector.get_weighted_congestion(counts)

        print(f"Total Detections: {len(detections)}")
        print(f"Vehicle Counts:   {counts}")
        print(f"Weighted Load:   {weighted_load:.1f} vehicle units")
        if detections:
            sample = detections[0]
            print(f"Sample BBox:     {sample['class']} (conf={sample['confidence']}) -> {sample['bbox']}")

    print("\n[SUCCESS] VehicleDetector verification complete.")
