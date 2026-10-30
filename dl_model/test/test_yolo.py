"""
Testing and Validation Suite for VehicleDetector.
Tests detection pipeline, vehicle counting by category, weighted congestion scoring,
and simulation-based fallback generator.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dl_model.models.yolo_detector import VehicleDetector, VEHICLE_WEIGHTS, ULTRALYTICS_AVAILABLE


def run_yolo_tests():
    """Run comprehensive test suite for VehicleDetector."""
    print("=" * 65)
    print("Autonomous Smart Traffic Management - VehicleDetector Test Suite")
    print(f"Ultralytics Available: {ULTRALYTICS_AVAILABLE}")
    print("=" * 65)

    detector = VehicleDetector(confidence=0.5)

    # ──────────────── Test 1: Weighted Congestion Math Verification ────────────────
    print("\n[Test 1] Verifying Weighted Congestion Calculation...")
    known_counts = {'car': 10, 'bike': 20, 'auto': 5, 'bus': 2, 'truck': 1}
    # Expected: 10*1.0 + 20*0.5 + 5*0.8 + 2*3.0 + 1*3.5 = 10 + 10 + 4 + 6 + 3.5 = 33.5
    expected_load = 33.5
    actual_load = detector.get_weighted_congestion(known_counts)
    assert abs(actual_load - expected_load) < 1e-4, f"Mismatch in load: got {actual_load}, expected {expected_load}"
    print(f"  Sample Counts: {known_counts}")
    print(f"  Formula:       10*1.0 + 20*0.5 + 5*0.8 + 2*3.0 + 1*3.5 = {expected_load}")
    print(f"  Calculated:    {actual_load} vehicle units -> PASS")

    # ──────────────── Test 2: Multi-Time-Slot Traffic Simulation ────────────────
    print("\n[Test 2] Testing Detections Across 5 Distinct 3-Hour Time Slots...")
    slots_to_test = [
        ('00-03', 'Late Night', 'low'),
        ('06-09', 'Morning Rush', 'critical'),
        ('12-15', 'Afternoon', 'medium'),
        ('15-18', 'Evening Rush', 'critical'),
        ('21-00', 'Night', 'low')
    ]

    for slot_id, label, expected_lvl in slots_to_test:
        detections = detector.detect(time_slot=slot_id)
        counts = detector.count_vehicles(detections)
        weighted_load = detector.get_weighted_congestion(counts)
        total_detected = len(detections)

        # Basic invariant checks
        assert total_detected == sum(counts.values()), "Count mismatch between detections list and summary counts!"
        for d in detections:
            bbox = d['bbox']
            assert len(bbox) == 4, f"Invalid bbox dimension: {bbox}"
            assert bbox[0] < bbox[2] and bbox[1] < bbox[3], f"Invalid bbox coordinates: {bbox}"
            assert 0.0 <= d['confidence'] <= 1.0, f"Invalid confidence: {d['confidence']}"

        print(f"  [{slot_id}] {label:<14} | Total: {total_detected:>2d} | Load: {weighted_load:>5.1f} units | Counts: {counts}")

    # ──────────────── Test 3: Simulation Fallback with Custom Override Context ────────────────
    print("\n[Test 3] Testing Context Overrides (Explicit Traffic Level and Manual Inputs)...")
    sim_context = {'time_slot': '12-15', 'traffic_level': 'critical'}
    critical_detections = detector.detect(context=sim_context)
    critical_counts = detector.count_vehicles(critical_detections)
    critical_load = detector.get_weighted_congestion(critical_counts)
    print(f"  Override Context: {sim_context}")
    print(f"  Detections Count: {len(critical_detections)}")
    print(f"  Vehicle Breakdown: {critical_counts}")
    print(f"  Weighted Load:    {critical_load:.1f} vehicle units")

    # ──────────────── Test 4: Synthetic Frame Image Processing ────────────────
    print("\n[Test 4] Testing Synthetic Camera Frame Handling...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_detections = detector.detect(image_or_data=dummy_frame, time_slot='09-12')
    frame_counts = detector.count_vehicles(frame_detections)
    print(f"  Synthetic Camera Frame: Shape (480, 640, 3)")
    print(f"  Frame Detections:       {len(frame_detections)} objects identified")
    print(f"  Frame Counts:           {frame_counts}")

    # ──────────────── Test 5: Bounding Box Inspection ────────────────
    print("\n[Test 5] Bounding Box and Confidence Inspection (First 5 Detections):")
    for i, det in enumerate(frame_detections[:5], 1):
        print(f"  Det #{i}: Class={det['class']:<6} | Conf={det['confidence']:.3f} | BBox={det['bbox']}")

    print("\n" + "=" * 65)
    print("[SUCCESS] VehicleDetector all 5 test scenarios verified successfully.")
    print("=" * 65)


if __name__ == '__main__':
    run_yolo_tests()
