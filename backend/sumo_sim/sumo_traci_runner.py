"""
SUMO TraCI Real-Time Simulation Runner.
Integrates Eclipse SUMO with TraCI, spawning cars, bikes, trucks, and buses,
adjusting traffic signals autonomously, and exporting live vehicle positions.
"""
import os
import sys
import time
import random
import json
from typing import Dict, List, Any

# Ensure SUMO tools are in Python path
SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\SUMO')
tools_path = os.path.join(SUMO_HOME, 'tools')
if tools_path not in sys.path:
    sys.path.append(tools_path)

import traci

SUMO_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_CFG = os.path.join(SUMO_DIR, 'city.sumocfg')
SUMO_BINARY = os.path.join(SUMO_HOME, 'bin', 'sumo.exe')


class SumoTraCiEngine:
    def __init__(self, port: int = 8873):
        self.port = port
        self.step_count = 0
        self.is_connected = False
        self.vehicle_counter = 0
        self.routes = ['route_W_E', 'route_E_W', 'route_S_N', 'route_N_S']
        self.vtypes = ['car', 'bike', 'truck', 'bus']
        self.vweights = [0.60, 0.18, 0.10, 0.12]  # Probability distribution
        
    def start(self, gui: bool = False):
        """Start SUMO TraCI instance."""
        binary = os.path.join(SUMO_HOME, 'bin', 'sumo-gui.exe' if gui else 'sumo.exe')
        sumo_cmd = [
            binary,
            "-c", SUMO_CFG,
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--collision.action", "none",
            "--start", "true"
        ]
        print(f"[SUMO] Starting TraCI on binary: {binary}")
        traci.start(sumo_cmd, port=self.port)
        self.is_connected = True
        print("[SUMO] TraCI connection established!")

    def step(self, spawn_prob: float = 0.4) -> Dict[str, Any]:
        """Perform one microscopic simulation step (0.1s)."""
        if not self.is_connected:
            return {}

        self.step_count += 1
        
        # Dynamically inject vehicles into the network
        if random.random() < spawn_prob:
            self._spawn_random_vehicle()
            
        traci.simulationStep()
        
        # Collect live vehicle telemetry from TraCI
        vehicles = []
        veh_ids = traci.vehicle.getIDList()
        
        for vid in veh_ids:
            try:
                x, y = traci.vehicle.getPosition(vid)
                vtype = traci.vehicle.getTypeID(vid)
                speed = traci.vehicle.getSpeed(vid)
                angle = traci.vehicle.getAngle(vid)
                road_id = traci.vehicle.getRoadID(vid)
                
                vehicles.append({
                    'id': vid,
                    'type': vtype,
                    'x': round(x, 2),
                    'y': round(y, 2),
                    'speed_kmh': round(speed * 3.6, 1),
                    'heading': round(angle, 1),
                    'road': road_id,
                    'is_emergency': (vtype in ['ambulance', 'firetruck'])
                })
            except Exception:
                continue
                
        # Count vehicles by class
        counts = {'car': 0, 'bike': 0, 'truck': 0, 'bus': 0, 'ambulance': 0, 'firetruck': 0}
        for v in vehicles:
            t = v['type']
            if t in counts:
                counts[t] += 1
                
        return {
            'step': self.step_count,
            'sim_time': round(self.step_count * 0.1, 1),
            'total_vehicles': len(vehicles),
            'counts': counts,
            'vehicles': vehicles
        }

    def _spawn_random_vehicle(self):
        """Inject a random vehicle into a SUMO route."""
        self.vehicle_counter += 1
        vid = f"veh_{self.vehicle_counter}"
        vtype = random.choices(self.vtypes, weights=self.vweights, k=1)[0]
        route = random.choice(self.routes)
        try:
            traci.vehicle.add(vehID=vid, routeID=route, typeID=vtype)
        except Exception:
            pass

    def inject_emergency_vehicle(self, emergency_type: str = 'ambulance'):
        """Inject an emergency vehicle into the primary arterial route."""
        self.vehicle_counter += 1
        vid = f"EMERGENCY_{emergency_type.upper()}_{self.vehicle_counter}"
        route = 'route_W_E'  # Main arterial green wave corridor
        try:
            traci.vehicle.add(vehID=vid, routeID=route, typeID=emergency_type)
            traci.vehicle.setSpeedMode(vid, 0)  # Ignore normal speed limits
            traci.vehicle.setMaxSpeed(vid, 25.0)  # 90 km/h
            print(f"[SUMO] Dispatched {emergency_type} with ID: {vid}")
            return vid
        except Exception as e:
            print(f"[SUMO] Error dispatching emergency: {e}")
            return None

    def close(self):
        """Close TraCI connection safely."""
        if self.is_connected:
            traci.close()
            self.is_connected = False
            print("[SUMO] TraCI closed.")


if __name__ == '__main__':
    engine = SumoTraCiEngine()
    engine.start(gui=False)
    print("[SUMO] Running 50 simulation steps with TraCI...")
    for i in range(50):
        data = engine.step(spawn_prob=0.8)
        if i % 10 == 0:
            print(f"Step {i}: {data['total_vehicles']} active vehicles on track: {data['counts']}")
    engine.close()
    print("[SUMO] Test run completed successfully!")
