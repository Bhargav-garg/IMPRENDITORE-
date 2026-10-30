"""
Flask REST API Server for Autonomous Smart Traffic Management System.
Factory pattern with background 25Hz simulation thread and thread-safe endpoints.
"""
import os
import time
import threading
from typing import Optional
from flask import Flask, jsonify, request, render_template

from backend import config
from backend.city_networks import get_available_cities
from backend.traffic_simulator import TrafficSimulator

# Global simulator instance
_global_simulator: Optional[TrafficSimulator] = None
_sim_lock = threading.Lock()
_sim_thread: Optional[threading.Thread] = None


def _simulation_worker(simulator: TrafficSimulator):
    """Background worker updating simulation state at ~25Hz."""
    last_time = time.time()
    while True:
        try:
            now = time.time()
            dt = now - last_time
            last_time = now
            # Clamp dt to prevent teleportation on OS lag
            clamped_dt = max(0.01, min(0.1, dt))
            simulator.step(clamped_dt)
            time.sleep(0.04)  # ~25 Hz update rate
        except Exception as e:
            time.sleep(0.05)


def get_simulator() -> TrafficSimulator:
    """Retrieve or initialize the singleton TrafficSimulator."""
    global _global_simulator, _sim_thread
    with _sim_lock:
        if _global_simulator is None:
            _global_simulator = TrafficSimulator(config.DEFAULT_CITY)
            _sim_thread = threading.Thread(
                target=_simulation_worker,
                args=(_global_simulator,),
                daemon=True,
                name="TrafficSimWorker"
            )
            _sim_thread.start()
            print("[INFO] Traffic simulator initialized and 25Hz background thread started.")
    return _global_simulator


def create_app() -> Flask:
    """Application factory for Flask server."""
    app = Flask(
        __name__,
        template_folder=os.path.join(config.BASE_DIR, 'frontend', 'templates'),
        static_folder=os.path.join(config.BASE_DIR, 'frontend', 'static')
    )

    # Initialize simulator
    sim = get_simulator()

    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    @app.route('/')
    def index():
        """Render main traffic management dashboard."""
        return render_template('index.html')

    @app.route('/api/state', methods=['GET'])
    def get_state():
        """Return full real-time simulation state (5Hz/25Hz telemetry)."""
        state = sim.get_state()
        return jsonify(state)

    @app.route('/api/cities', methods=['GET'])
    def get_cities():
        """Return list of supported Indian cities with GPS center."""
        cities = get_available_cities()
        return jsonify({
            'status': 'ok',
            'current_city': sim.city_id,
            'cities': cities
        })

    @app.route('/api/city', methods=['POST'])
    def switch_city():
        """Switch active city network."""
        data = request.get_json(silent=True) or request.form
        city_id = (data.get('city_id') or data.get('city', '')).strip().lower()
        if not city_id:
            return jsonify({'status': 'error', 'message': 'Missing city_id parameter'}), 400

        sim.load_city(city_id)
        print(f"[API] Switched to city: {sim.city_id}")
        return jsonify({
            'status': 'ok',
            'city_id': sim.city_id,
            'name': sim.city_data['name'],
            'center': sim.city_data['center'],
            'zoom': sim.city_data['zoom']
        })

    @app.route('/api/time_slot', methods=['POST'])
    def switch_time_slot():
        """Change 3-hour time slot window."""
        data = request.get_json(silent=True) or request.form
        slot_id = (data.get('slot_id') or data.get('slot', '')).strip()
        if slot_id not in config.TIME_SLOTS:
            return jsonify({'status': 'error', 'message': f'Invalid slot_id: {slot_id}'}), 400

        sim.set_time_slot(slot_id)
        print(f"[API] Set time slot: {slot_id} (intensity={sim.intensity})")
        return jsonify({
            'status': 'ok',
            'slot_id': sim.slot_id,
            'label': sim.slot_cfg['label'],
            'intensity': sim.intensity
        })

    @app.route('/api/emergency', methods=['POST'])
    def trigger_emergency():
        """Dispatch an emergency vehicle (ambulance / firetruck)."""
        data = request.get_json(silent=True) or request.form
        v_type = (data.get('type') or data.get('vehicle_type', 'ambulance')).strip().lower()
        status = sim.trigger_emergency(v_type)
        print(f"[API] Emergency vehicle dispatched: {v_type}")
        return jsonify({
            'status': 'ok',
            'emergency': status
        })

    @app.route('/api/emergency/clear', methods=['POST'])
    def clear_emergency():
        """Clear active emergency vehicle corridor."""
        sim.emergency_handler.clear_emergency()
        sim.run_quantum_optimization()
        print("[API] Emergency cleared.")
        return jsonify({
            'status': 'ok',
            'message': 'Emergency cleared'
        })

    @app.route('/api/burst', methods=['POST'])
    def traffic_surge():
        """Inject an instantaneous surge of vehicles."""
        data = request.get_json(silent=True) or request.form
        try:
            count = int(data.get('count', 16))
        except (ValueError, TypeError):
            count = 16

        sim.spawn_surge(count)
        print(f"[API] Injected traffic surge (+{count} vehicles)")
        return jsonify({
            'status': 'ok',
            'surge_count': count,
            'total_vehicles': len(sim.vehicles)
        })

    @app.route('/api/quantum_solve', methods=['POST'])
    def force_quantum_solve():
        """Force manual QAOA quantum optimization solver."""
        print("[API] Manual QAOA solver requested.")
        telemetry = sim.run_quantum_optimization()
        return jsonify({
            'status': 'ok',
            'quantum_telemetry': telemetry
        })

    @app.route('/api/google_key', methods=['POST'])
    def set_google_key():
        """Configure Google Maps API Key at runtime."""
        data = request.get_json(silent=True) or request.form
        key = (data.get('api_key') or data.get('key', '')).strip()
        sim.set_google_maps_key(key)
        has_key = bool(sim.google_maps.api_key)
        print(f"[API] Google Maps API key updated (configured={has_key})")
        return jsonify({
            'status': 'ok',
            'has_key': has_key
        })

    @app.route('/api/mode', methods=['POST'])
    def switch_mode():
        """Switch between Quantum and Classical modes."""
        data = request.get_json(silent=True) or request.form
        mode = (data.get('mode', 'quantum')).strip().lower()
        sim.set_mode(mode)
        print(f"[API] Switched operation mode to: {sim.mode}")
        return jsonify({
            'status': 'ok',
            'mode': sim.mode
        })

    @app.route('/api/playback', methods=['POST'])
    def set_playback():
        """Control playback (play/pause/speed)."""
        data = request.get_json(silent=True) or request.form
        playing = data.get('playing')
        speed = data.get('speed')
        if speed is not None:
            try:
                speed = float(speed)
            except (ValueError, TypeError):
                speed = 1.0

        sim.set_playback(playing, speed)
        return jsonify({
            'status': 'ok',
            'is_running': sim.is_running,
            'speed_multiplier': sim.speed_multiplier
        })

    @app.route('/api/traffic_data', methods=['GET'])
    def get_traffic_data():
        """Fetch Google Maps / calibrated distance matrix."""
        data = sim.fetch_traffic_data()
        return jsonify(data)

    return app


if __name__ == '__main__':
    application = create_app()
    application.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
