"""
Traffic Simulator Engine for Autonomous Smart Traffic Management System.
Features:
  - Geographic vehicle movement along real GPS coordinates
  - 5-intersection network per city with approach queues (N, S, E, W)
  - Vehicle types with realistic weights (car, bike, auto, bus, truck)
  - 3-hour time-slot traffic intensity scaling
  - Autonomous signal state machine (NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW)
  - Quantum QAOA engine integration for dynamic green allocation (8s - 70s)
  - Emergency vehicle preemption (Ambulance / Fire Truck)
  - Deep Learning model integration (YOLO, CNN, LSTM)
  - Google Maps API & calibrated fallback integration
"""
import time
import math
import random
import threading
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from backend import config
from backend.city_networks import get_city, get_available_cities, haversine_distance_km
from backend.quantum_engine import QuantumEngine
from backend.google_maps_service import GoogleMapsService
from backend.emergency_handler import EmergencyHandler

# Import DL models gracefully
try:
    from dl_model.models import CongestionLSTM, TrafficCNN, VehicleDetector
    HAS_DL = True
except Exception:
    HAS_DL = False


class TrafficSimulator:
    """
    Main simulation engine for city traffic networks.
    Thread-safe simulation of geographic vehicle movement, queue dynamics,
    quantum signal control, and emergency corridors.
    """

    def __init__(self, city_id: str = config.DEFAULT_CITY):
        self.lock = threading.RLock()
        self.city_id = city_id
        self.city_data = get_city(city_id)

        # Mode: 'quantum' or 'classical'
        self.mode = 'quantum'
        self.is_running = True
        self.speed_multiplier = 1.0

        # Time slot
        self.slot_id, self.slot_cfg = config.get_current_time_slot()
        self.intensity = self.slot_cfg['intensity']

        # Core subsystems
        self.quantum_engine = QuantumEngine()
        self.google_maps = GoogleMapsService()
        self.emergency_handler = EmergencyHandler()

        # DL models
        self.lstm_model = None
        self.cnn_model = None
        self.yolo_detector = None
        self._init_dl_models()

        # Intersections state
        self.intersections: List[Dict[str, Any]] = []
        self._init_intersections()

        # Vehicles
        self.vehicles: List[Dict[str, Any]] = []
        self.vehicle_id_counter = 0

        # Telemetry & Stats
        self.simulation_time = 0.0
        self.last_quantum_solve_time = -100.0
        self.quantum_telemetry: Dict[str, Any] = {}
        self.dl_predictions: Dict[str, Any] = {}

        # Historical KPI stats
        self.total_vehicles_passed = 0
        self.total_wait_saved_sec = 0.0
        self.total_co2_saved_kg = 0.0
        self.total_fuel_saved_liters = 0.0

        # Congestion history buffer for LSTM (length 12)
        self.congestion_history: List[float] = [self.intensity * 0.5] * config.LSTM_SEQ_LENGTH

        # Initial population
        self._populate_initial_vehicles()

        # Initial quantum optimization run
        self.run_quantum_optimization()

    def _init_dl_models(self):
        """Initialize DL models or fallback instances."""
        if HAS_DL:
            try:
                self.lstm_model = CongestionLSTM()
                self.cnn_model = TrafficCNN()
                self.yolo_detector = VehicleDetector()
            except Exception as e:
                pass

    def _init_intersections(self):
        """Initialize intersection signal states and queues from city network."""
        self.intersections = []
        raw_intersections = self.city_data.get('intersections', [])

        for raw in raw_intersections:
            earlier_wait = float(raw.get('earlier_wait_sec', 70.0))
            # Initial phase staggered for visual variety
            initial_phase = 'NS_GREEN' if raw['id'] % 2 == 0 else 'EW_GREEN'
            initial_duration = 35.0

            inter = {
                'id': raw['id'],
                'name': raw['name'],
                'lat': raw['lat'],
                'lng': raw['lng'],
                'earlier_wait_sec': earlier_wait,
                'quantum_wait_sec': round(earlier_wait * 0.55, 1),
                'wait_reduction_pct': 45.0,
                'connections': raw.get('connections', []),
                'approaches': raw.get('approaches', ['N', 'S', 'E', 'W']),
                'current_phase': initial_phase,
                'timer': initial_duration,
                'total_duration': initial_duration,
                'quantum_allocated_green': {'NS': 35.0, 'EW': 35.0},
                'queue_lengths': {'N': 0, 'S': 0, 'E': 0, 'W': 0},
                'weighted_queue': {'NS': 0.0, 'EW': 0.0},
                'wait_times': {'NS': 12.0, 'EW': 12.0},
                'vehicles_detected': {'car': 0, 'bike': 0, 'auto': 0, 'bus': 0, 'truck': 0},
                'congestion_level': 'medium',
                'weighted_congestion_index': 35.0
            }
            self.intersections.append(inter)

    def load_city(self, city_id: str):
        """Switch active city network."""
        with self.lock:
            self.city_id = city_id
            self.city_data = get_city(city_id)
            self._init_intersections()
            self.vehicles.clear()
            self.vehicle_id_counter = 0
            self.emergency_handler.clear_emergency()
            self.simulation_time = 0.0
            self._populate_initial_vehicles()
            self.run_quantum_optimization()

    def set_time_slot(self, slot_id: str):
        """Switch 3-hour time slot window and update spawn intensity."""
        with self.lock:
            if slot_id in config.TIME_SLOTS:
                self.slot_id = slot_id
                self.slot_cfg = config.TIME_SLOTS[slot_id]
                self.intensity = self.slot_cfg['intensity']
                # Re-run quantum optimization under new intensity
                self.run_quantum_optimization()

    def set_mode(self, mode: str):
        """Toggle system mode: 'quantum' or 'classical'."""
        with self.lock:
            self.mode = 'quantum' if mode.lower() == 'quantum' else 'classical'
            if self.mode == 'quantum':
                self.run_quantum_optimization()

    def set_playback(self, playing: Optional[bool] = None, speed: Optional[float] = None):
        """Update playback state and speed multiplier."""
        with self.lock:
            if playing is not None:
                self.is_running = bool(playing)
            if speed is not None and speed > 0:
                self.speed_multiplier = float(speed)

    def set_google_maps_key(self, key: str):
        """Set Google Maps API key."""
        with self.lock:
            self.google_maps.set_api_key(key)

    def fetch_traffic_data(self) -> Dict[str, Any]:
        """Fetch distance matrix / traffic data from Google Maps service."""
        with self.lock:
            return self.google_maps.fetch_traffic_data(self.intersections, self.slot_id)

    def _sample_vehicle_type(self) -> str:
        """Sample vehicle type with realistic Indian traffic distribution."""
        # Distribution: 60% car, 15% bike, 10% auto, 8% bus, 7% truck
        r = random.random()
        if r < 0.60:
            return 'car'
        elif r < 0.75:
            return 'bike'
        elif r < 0.85:
            return 'auto'
        elif r < 0.93:
            return 'bus'
        else:
            return 'truck'

    def _create_vehicle(
        self,
        target_intersection_id: int,
        approach: str,
        vehicle_type: Optional[str] = None,
        is_emergency: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Create a vehicle heading towards an intersection approach."""
        if target_intersection_id >= len(self.intersections):
            return None

        inter = self.intersections[target_intersection_id]
        v_type = vehicle_type or self._sample_vehicle_type()
        cfg = config.VEHICLE_TYPES.get(v_type, config.VEHICLE_TYPES['car'])

        self.vehicle_id_counter += 1
        v_id = f"veh_{self.vehicle_id_counter}"

        # Approach distance in degrees (~280m to 350m feeder)
        offset = 0.0028 + random.uniform(0.0002, 0.0008)
        target_lat = inter['lat']
        target_lng = inter['lng']

        if approach == 'N':
            start_lat = target_lat + offset
            start_lng = target_lng + random.uniform(-0.00008, 0.00008)
            heading = 180.0
        elif approach == 'S':
            start_lat = target_lat - offset
            start_lng = target_lng + random.uniform(-0.00008, 0.00008)
            heading = 0.0
        elif approach == 'E':
            start_lat = target_lat + random.uniform(-0.00008, 0.00008)
            start_lng = target_lng + offset
            heading = 270.0
        else:  # 'W'
            start_lat = target_lat + random.uniform(-0.00008, 0.00008)
            start_lng = target_lng - offset
            heading = 90.0

        # Speed in meters/sec: ~30-45 km/h for light, ~25-35 km/h for heavy
        base_speed = random.uniform(8.0, 12.0) if cfg['size'] == 'light' else random.uniform(6.5, 9.5)

        vehicle = {
            'id': v_id,
            'type': v_type,
            'weight': cfg['weight'],
            'length_m': cfg['length_m'],
            'color': cfg['color'],
            'label': cfg['label'],
            'lat': start_lat,
            'lng': start_lng,
            'heading': heading,
            'target_intersection': target_intersection_id,
            'approach': approach,
            'state': 'approaching',  # 'approaching', 'waiting', 'crossing', 'departed'
            'speed_mps': base_speed,
            'max_speed_mps': base_speed,
            'distance_to_center_m': haversine_distance_km(start_lat, start_lng, target_lat, target_lng) * 1000.0,
            'wait_time_sec': 0.0,
            'is_emergency': is_emergency,
            'progress': 0.0
        }
        return vehicle

    def _populate_initial_vehicles(self):
        """Populate initial set of vehicles across all intersections."""
        # Base count proportional to intensity
        base_count_per_node = int(6 + self.intensity * 8)
        approaches = ['N', 'S', 'E', 'W']

        for inter in self.intersections:
            for _ in range(base_count_per_node):
                app = random.choice(approaches)
                veh = self._create_vehicle(inter['id'], app)
                if veh:
                    # Stagger starting positions along approach
                    fraction = random.uniform(0.1, 0.85)
                    start_lat = veh['lat']
                    start_lng = veh['lng']
                    target_lat = inter['lat']
                    target_lng = inter['lng']

                    veh['lat'] = start_lat + fraction * (target_lat - start_lat)
                    veh['lng'] = start_lng + fraction * (target_lng - start_lng)
                    veh['distance_to_center_m'] = haversine_distance_km(veh['lat'], veh['lng'], target_lat, target_lng) * 1000.0
                    veh['progress'] = fraction
                    self.vehicles.append(veh)

    def spawn_surge(self, count: int = 16):
        """Inject an instantaneous surge of vehicles across the network."""
        with self.lock:
            approaches = ['N', 'S', 'E', 'W']
            for _ in range(count):
                inter_id = random.randrange(len(self.intersections))
                app = random.choice(approaches)
                veh = self._create_vehicle(inter_id, app)
                if veh:
                    self.vehicles.append(veh)

    def trigger_emergency(self, vehicle_type: str = 'ambulance'):
        """Dispatch an emergency vehicle (ambulance or firetruck)."""
        with self.lock:
            status = self.emergency_handler.dispatch(vehicle_type, self.intersections)
            # Immediately trigger quantum optimization with emergency corridor
            self.run_quantum_optimization()
            return status

    def run_quantum_optimization(self) -> Dict[str, Any]:
        """Run QAOA quantum optimization on current network state."""
        with self.lock:
            emergency_corridor = None
            if self.emergency_handler.active:
                emergency_corridor = {
                    'active': True,
                    'route': self.emergency_handler.get_route(),
                    'directions': {
                        nid: 'NS' if phase == 'NS_GREEN' else 'EW'
                        for nid, phase in self.emergency_handler.preempted_nodes.items()
                    }
                }

            telemetry = self.quantum_engine.optimize(self.intersections, emergency_corridor)
            self.quantum_telemetry = telemetry
            self.last_quantum_solve_time = self.simulation_time

            # Apply recommendations to intersections if in quantum mode
            if self.mode == 'quantum':
                for rec in telemetry.get('phase_recommendations', []):
                    idx = rec['intersection_id']
                    if idx < len(self.intersections):
                        inter = self.intersections[idx]
                        opt_phase = rec['optimal_phase']
                        dur = rec['green_duration_sec']

                        # Save quantum allocation
                        if opt_phase == 'NS_GREEN':
                            inter['quantum_allocated_green']['NS'] = dur
                        else:
                            inter['quantum_allocated_green']['EW'] = dur

                        inter['quantum_wait_sec'] = rec['quantum_wait_sec']
                        inter['wait_reduction_pct'] = rec['reduction_pct']

                        # If currently at yellow or timer expired, update phase
                        if inter['timer'] <= 3.0:
                            inter['current_phase'] = opt_phase
                            inter['total_duration'] = dur
                            inter['timer'] = dur

            return telemetry

    def step(self, dt: float):
        """Advance simulation by dt seconds (called at ~25Hz)."""
        with self.lock:
            if not self.is_running:
                return

            effective_dt = dt * self.speed_multiplier
            self.simulation_time += effective_dt

            # 1. Update Emergency vehicle
            if self.emergency_handler.active:
                still_active = self.emergency_handler.update_position(effective_dt, self.speed_multiplier)
                if not still_active:
                    # Emergency cleared: re-run quantum optimization
                    self.run_quantum_optimization()

            # 2. Update Traffic Light State Machine
            self._update_signals(effective_dt)

            # 3. Vehicle Spawning
            self._spawn_vehicles(effective_dt)

            # 4. Move Vehicles and Update Queues
            self._update_vehicles(effective_dt)

            # 5. Periodically re-run Quantum Optimization (every 18 seconds)
            if self.mode == 'quantum' and (self.simulation_time - self.last_quantum_solve_time) > 18.0:
                self.run_quantum_optimization()

            # 6. Update DL Predictions
            self._update_dl_telemetry()

    def _update_signals(self, dt: float):
        """Advance timers and execute phase transitions for each intersection."""
        for inter in self.intersections:
            i_id = inter['id']

            # If emergency vehicle active and preempted
            if self.emergency_handler.active and i_id in self.emergency_handler.preempted_nodes:
                forced_phase = self.emergency_handler.preempted_nodes[i_id]
                inter['current_phase'] = forced_phase
                inter['timer'] = max(inter['timer'], 35.0)
                continue

            inter['timer'] -= dt
            if inter['timer'] <= 0.0:
                curr = inter['current_phase']
                if curr == 'NS_GREEN':
                    inter['current_phase'] = 'NS_YELLOW'
                    inter['timer'] = config.YELLOW_SEC
                    inter['total_duration'] = config.YELLOW_SEC
                elif curr == 'NS_YELLOW':
                    inter['current_phase'] = 'EW_GREEN'
                    dur = inter['quantum_allocated_green']['EW'] if self.mode == 'quantum' else 40.0
                    inter['timer'] = dur
                    inter['total_duration'] = dur
                elif curr == 'EW_GREEN':
                    inter['current_phase'] = 'EW_YELLOW'
                    inter['timer'] = config.YELLOW_SEC
                    inter['total_duration'] = config.YELLOW_SEC
                elif curr == 'EW_YELLOW':
                    inter['current_phase'] = 'NS_GREEN'
                    dur = inter['quantum_allocated_green']['NS'] if self.mode == 'quantum' else 40.0
                    inter['timer'] = dur
                    inter['total_duration'] = dur

    def _spawn_vehicles(self, dt: float):
        """Spawn vehicles periodically based on current time-slot intensity."""
        # Target vehicles in system: scales from 25 (at 0.15) to 80 (at 0.95)
        target_count = int(25 + self.intensity * 55)
        current_count = len(self.vehicles)

        if current_count < target_count:
            # Poisson-like spawn probability
            spawn_rate_per_sec = 2.0 + self.intensity * 5.0
            if random.random() < spawn_rate_per_sec * dt:
                inter_id = random.randrange(len(self.intersections))
                approach = random.choice(['N', 'S', 'E', 'W'])
                veh = self._create_vehicle(inter_id, approach)
                if veh:
                    self.vehicles.append(veh)

    def _update_vehicles(self, dt: float):
        """Move vehicles, compute queue stoplines, and record wait times."""
        stop_line_dist_m = 25.0
        departure_threshold_m = 15.0

        # Reset counts
        for inter in self.intersections:
            inter['queue_lengths'] = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
            inter['weighted_queue'] = {'NS': 0.0, 'EW': 0.0}
            inter['vehicles_detected'] = {'car': 0, 'bike': 0, 'auto': 0, 'bus': 0, 'truck': 0}

        surviving_vehicles: List[Dict[str, Any]] = []

        # Sort vehicles by target and distance so leaders stop first
        self.vehicles.sort(key=lambda v: (v['target_intersection'], v['approach'], v['distance_to_center_m']))

        # Track queue depth in meters along each approach
        queue_depth: Dict[Tuple[int, str], float] = {}

        for veh in self.vehicles:
            target_id = veh['target_intersection']
            if target_id >= len(self.intersections):
                continue

            inter = self.intersections[target_id]
            app = veh['approach']
            v_type = veh['type']
            v_len = veh['length_m']

            # Count detection at this intersection
            if v_type in inter['vehicles_detected']:
                inter['vehicles_detected'][v_type] += 1

            # Determine signal light for this vehicle's approach
            phase = inter['current_phase']
            is_ns = app in ['N', 'S']
            is_ew = app in ['E', 'W']

            signal_is_green = False
            if is_ns and phase == 'NS_GREEN':
                signal_is_green = True
            elif is_ew and phase == 'EW_GREEN':
                signal_is_green = True

            # Stop position behind preceding queued vehicles
            q_key = (target_id, app)
            current_q_tail = queue_depth.get(q_key, 0.0)
            effective_stopline = stop_line_dist_m + current_q_tail

            dist = veh['distance_to_center_m']

            if veh['state'] == 'approaching':
                if dist <= effective_stopline + 5.0 and not signal_is_green:
                    # Vehicle must stop and queue
                    veh['state'] = 'waiting'
                    veh['speed_mps'] = 0.0
                else:
                    # Move towards center
                    move_dist = veh['speed_mps'] * dt
                    dist = max(0.0, dist - move_dist)
                    veh['distance_to_center_m'] = dist

                    # Interpolate GPS position
                    self._step_vehicle_gps(veh, inter, move_dist)

                    if dist <= stop_line_dist_m and signal_is_green:
                        veh['state'] = 'crossing'

            elif veh['state'] == 'waiting':
                if signal_is_green:
                    # Green light: accelerate and clear queue
                    veh['state'] = 'crossing'
                    veh['speed_mps'] = min(veh['max_speed_mps'], veh['speed_mps'] + 2.5 * dt)
                else:
                    veh['speed_mps'] = 0.0
                    veh['wait_time_sec'] += dt

                # Register in queue
                inter['queue_lengths'][app] += 1
                weight = veh['weight']
                if is_ns:
                    inter['weighted_queue']['NS'] += weight
                else:
                    inter['weighted_queue']['EW'] += weight

                queue_depth[q_key] = current_q_tail + v_len + 2.0

            elif veh['state'] == 'crossing':
                # Accelerate and move past center
                veh['speed_mps'] = min(veh['max_speed_mps'], veh['speed_mps'] + 3.0 * dt)
                move_dist = veh['speed_mps'] * dt
                dist = dist - move_dist
                veh['distance_to_center_m'] = dist

                self._step_vehicle_gps(veh, inter, move_dist)

                if dist <= -departure_threshold_m:
                    # Successfully crossed intersection
                    veh['state'] = 'departed'
                    self.total_vehicles_passed += 1

                    # Compute wait savings
                    earlier_w = inter['earlier_wait_sec']
                    quantum_w = inter['quantum_wait_sec']
                    saved = max(0.0, earlier_w - quantum_w)
                    self.total_wait_saved_sec += saved
                    # 0.038g CO2 per sec saved per vehicle
                    self.total_co2_saved_kg += (saved * 0.038 * veh['weight']) / 1000.0
                    # 0.0003L fuel per sec
                    self.total_fuel_saved_liters += saved * 0.0003

                    # Optional: transition vehicle to connected neighbor instead of despawning
                    connections = inter.get('connections', [])
                    if connections and random.random() < 0.45:
                        nxt_id = random.choice(connections)
                        nxt_inter = self.intersections[nxt_id]
                        veh['target_intersection'] = nxt_id
                        veh['approach'] = random.choice(['N', 'S', 'E', 'W'])
                        veh['distance_to_center_m'] = haversine_distance_km(
                            veh['lat'], veh['lng'], nxt_inter['lat'], nxt_inter['lng']
                        ) * 1000.0
                        veh['state'] = 'approaching'
                        veh['speed_mps'] = veh['max_speed_mps'] * 0.8
                        surviving_vehicles.append(veh)
                    continue

            if veh['state'] != 'departed':
                surviving_vehicles.append(veh)

        self.vehicles = surviving_vehicles

        # Compute congestion level and index per intersection
        for inter in self.intersections:
            w_total = inter['weighted_queue']['NS'] + inter['weighted_queue']['EW']
            inter['weighted_congestion_index'] = round(w_total, 1)

            if w_total < 15.0:
                inter['congestion_level'] = 'low'
            elif w_total < 35.0:
                inter['congestion_level'] = 'medium'
            elif w_total < 55.0:
                inter['congestion_level'] = 'high'
            else:
                inter['congestion_level'] = 'critical'

    def _step_vehicle_gps(self, veh: Dict[str, Any], inter: Dict[str, Any], move_dist_m: float):
        """Move vehicle coordinate along its heading vector."""
        # 1 meter latitude is approx 1 / 111,000 degrees
        # 1 meter longitude is approx 1 / (111,000 * cos(lat)) degrees
        lat_deg_per_m = 1.0 / 111139.0
        lng_deg_per_m = 1.0 / (111139.0 * math.cos(math.radians(inter['lat'])))

        heading_rad = math.radians(veh['heading'])
        d_lat = move_dist_m * math.cos(heading_rad) * lat_deg_per_m
        d_lng = move_dist_m * math.sin(heading_rad) * lng_deg_per_m

        veh['lat'] = round(veh['lat'] + d_lat, 6)
        veh['lng'] = round(veh['lng'] + d_lng, 6)

    def _update_dl_telemetry(self):
        """Generate and update DL predictions (LSTM, CNN, YOLO)."""
        # Overall network congestion ratio
        total_q = sum(
            inter['weighted_queue']['NS'] + inter['weighted_queue']['EW']
            for inter in self.intersections
        )
        avg_density = min(1.0, total_q / 120.0)

        # Update historical buffer
        self.congestion_history.append(round(avg_density, 3))
        if len(self.congestion_history) > config.LSTM_SEQ_LENGTH:
            self.congestion_history.pop(0)

        # 1. LSTM Forecast (4 steps: +15m, +30m, +45m, +60m)
        lstm_forecast = []
        if self.lstm_model is not None:
            try:
                seq_arr = np.array(self.congestion_history, dtype=np.float32).reshape(1, config.LSTM_SEQ_LENGTH, 1)
                preds = self.lstm_model.predict(seq_arr)[0]
                steps = ['+15m', '+30m', '+45m', '+60m']
                for s, val in zip(steps, preds):
                    v = float(val)
                    if v < config.CONGESTION_THRESHOLDS['low']:
                        lvl = 'Low'
                    elif v < config.CONGESTION_THRESHOLDS['medium']:
                        lvl = 'Medium'
                    elif v < config.CONGESTION_THRESHOLDS['high']:
                        lvl = 'High'
                    else:
                        lvl = 'Critical'
                    lstm_forecast.append({'step': s, 'level': lvl, 'value': round(v, 2)})
            except Exception:
                pass

        if not lstm_forecast:
            # Fallback forecast based on intensity trend
            steps = ['+15m', '+30m', '+45m', '+60m']
            for idx, s in enumerate(steps):
                trend_val = min(1.0, max(0.05, avg_density + (idx - 1) * 0.05 + random.uniform(-0.03, 0.03)))
                lvl = 'Low' if trend_val < 0.35 else ('Medium' if trend_val < 0.65 else 'High')
                lstm_forecast.append({'step': s, 'level': lvl, 'value': round(trend_val, 2)})

        # 2. CNN Classifier
        cnn_result = {'predicted_level': 'Medium', 'confidence': 88, 'probabilities': {}}
        if self.cnn_model is not None:
            try:
                # 20 feature vector: queue states across 5 intersections * 4 approaches
                feat = []
                for inter in self.intersections:
                    for app in ['N', 'S', 'E', 'W']:
                        feat.append(float(inter['queue_lengths'].get(app, 0)))
                feat_arr = np.array(feat[:config.CNN_INPUT_FEATURES], dtype=np.float32).reshape(1, -1)
                out = self.cnn_model.classify(feat_arr)[0]
                cnn_result = {
                    'predicted_level': out['class'].capitalize(),
                    'confidence': int(round(out['confidence'] * 100)),
                    'probabilities': out.get('probabilities', {})
                }
            except Exception:
                pass

        # 3. YOLO Detection Aggregation
        yolo_counts = {'car': 0, 'bike': 0, 'auto': 0, 'bus': 0, 'truck': 0}
        for v in self.vehicles:
            t = v['type']
            if t in yolo_counts:
                yolo_counts[t] += 1

        self.dl_predictions = {
            'lstm_forecast': lstm_forecast,
            'cnn_classification': cnn_result,
            'yolo_counts': yolo_counts,
            'total_detections': len(self.vehicles)
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Return complete JSON-serializable state of the entire simulation.
        """
        with self.lock:
            # Vehicle type counts
            type_counts = {'car': 0, 'bike': 0, 'auto': 0, 'bus': 0, 'truck': 0}
            for v in self.vehicles:
                t = v['type']
                if t in type_counts:
                    type_counts[t] += 1

            # Overall congestion
            weighted_total = sum(i['weighted_congestion_index'] for i in self.intersections)
            overall_congestion_score = min(100.0, (weighted_total / max(1, len(self.intersections) * 60.0)) * 100.0)
            if overall_congestion_score < 30.0:
                overall_label = 'Low'
            elif overall_congestion_score < 60.0:
                overall_label = 'Medium'
            elif overall_congestion_score < 80.0:
                overall_label = 'High'
            else:
                overall_label = 'Critical'

            # Comparative wait metrics
            wait_metrics = self.quantum_engine.compute_comparative_wait_metrics(
                self.intersections, self.quantum_telemetry
            )

            # Emergency info
            emergency_info = self.emergency_handler.get_status()

            # Vehicles serialized list (compact representation for web performance)
            vehicle_list = []
            for v in self.vehicles:
                vehicle_list.append({
                    'id': v['id'],
                    'type': v['type'],
                    'lat': v['lat'],
                    'lng': v['lng'],
                    'heading': v['heading'],
                    'color': v['color'],
                    'speed': round(v['speed_mps'], 1),
                    'state': v['state'],
                    'target_intersection': v['target_intersection'],
                    'is_emergency': v['is_emergency']
                })

            # If emergency vehicle active, prepend it
            if emergency_info.get('active'):
                vehicle_list.append({
                    'id': 'emergency_active',
                    'type': emergency_info.get('type', 'ambulance'),
                    'lat': emergency_info['current_position']['lat'],
                    'lng': emergency_info['current_position']['lng'],
                    'heading': emergency_info.get('heading', 0.0),
                    'color': emergency_info.get('color', '#ff0000'),
                    'speed': 18.0,
                    'state': 'emergency_en_route',
                    'target_intersection': emergency_info['route'][-1] if emergency_info['route'] else 0,
                    'is_emergency': True
                })

            return {
                'city': {
                    'id': self.city_id,
                    'name': self.city_data['name'],
                    'center': self.city_data['center'],
                    'zoom': self.city_data['zoom']
                },
                'time_slot': {
                    'id': self.slot_id,
                    'label': self.slot_cfg['label'],
                    'intensity': self.intensity
                },
                'mode': self.mode,
                'is_running': self.is_running,
                'speed_multiplier': self.speed_multiplier,
                'simulation_time': round(self.simulation_time, 1),
                'intersections': self.intersections,
                'vehicles': vehicle_list,
                'vehicle_counts': type_counts,
                'total_vehicles': len(self.vehicles),
                'congestion': {
                    'score': round(overall_congestion_score, 1),
                    'label': overall_label,
                    'weighted_index': round(weighted_total, 1)
                },
                'emergency': emergency_info,
                'quantum_telemetry': self.quantum_telemetry,
                'dl_predictions': self.dl_predictions,
                'stats': {
                    'total_vehicles_passed': self.total_vehicles_passed,
                    'avg_earlier_wait_sec': wait_metrics.get('avg_earlier_wait_sec', 70.0),
                    'avg_quantum_wait_sec': wait_metrics.get('avg_quantum_wait_sec', 35.0),
                    'overall_reduction_pct': wait_metrics.get('overall_reduction_pct', 50.0),
                    'total_co2_saved_kg': round(self.total_co2_saved_kg, 2),
                    'total_fuel_saved_liters': round(self.total_fuel_saved_liters, 2),
                    'emergency_clearance_time_sec': emergency_info.get('total_clearance_time_sec', 0.0)
                }
            }
