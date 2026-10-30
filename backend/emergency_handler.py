"""
Emergency Handler for Autonomous Smart Traffic Management System.
Manages emergency vehicle dispatching, shortest path routing, signal preemption,
and dynamic waypoint tracking with auto-clearance.
"""
import math
import heapq
from typing import Dict, List, Any, Optional, Tuple

from backend import config
from backend.city_networks import haversine_distance_km


class EmergencyHandler:
    """
    Handles emergency corridors (Ambulance / Fire Truck):
      - Calculates shortest path across city intersection topology
      - Preempts traffic lights along the corridor (Green Wave)
      - Holds all non-corridor approaches at Red
      - Tracks emergency vehicle movement and auto-clears on arrival
    """

    def __init__(self):
        self.active: bool = False
        self.vehicle_type: Optional[str] = None
        self.route: List[int] = []               # Sequence of intersection IDs
        self.route_coords: List[List[float]] = [] # [[lat, lng], ...]
        self.current_leg_idx: int = 0
        self.leg_progress: float = 0.0           # 0.0 to 1.0 along current leg
        self.current_pos: Dict[str, float] = {'lat': 0.0, 'lng': 0.0}
        self.heading: float = 0.0
        self.speed_kmh: float = 65.0             # High-speed emergency vehicle
        self.elapsed_sec: float = 0.0
        self.total_clearance_time_sec: float = 0.0
        self.preempted_nodes: Dict[int, str] = {} # node_id -> required phase ('NS_GREEN'/'EW_GREEN')

    def dispatch(
        self,
        vehicle_type: str,
        intersections: List[Dict[str, Any]],
        start_id: Optional[int] = None,
        target_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Dispatch an emergency vehicle (ambulance/firetruck) across the network.
        Finds shortest corridor route and initiates green wave preemption.
        """
        self.active = True
        self.vehicle_type = vehicle_type if vehicle_type in config.EMERGENCY_TYPES else 'ambulance'
        self.elapsed_sec = 0.0
        self.leg_progress = 0.0
        self.current_leg_idx = 0

        num_nodes = len(intersections)
        if num_nodes == 0:
            return self.get_status()

        # If start and target are not specified, pick a realistic arterial corridor:
        # e.g., from outer arterial node (like 4 or 3) to central/hospital node (0 or 2)
        if start_id is None or start_id >= num_nodes:
            start_id = 4 if num_nodes > 4 else (num_nodes - 1)
        if target_id is None or target_id >= num_nodes or target_id == start_id:
            target_id = 2 if num_nodes > 2 else 0

        # Dijkstra shortest path on intersection graph
        self.route = self._calculate_shortest_path(intersections, start_id, target_id)

        # Build coordinate path
        id_to_inter = {inter['id']: inter for inter in intersections}
        self.route_coords = [[id_to_inter[nid]['lat'], id_to_inter[nid]['lng']] for nid in self.route]

        if self.route_coords:
            self.current_pos = {
                'lat': self.route_coords[0][0],
                'lng': self.route_coords[0][1]
            }

        # Calculate corridor orientation for each node in route
        self.preempt_signals(intersections, self.route)

        return self.get_status()

    def _calculate_shortest_path(
        self,
        intersections: List[Dict[str, Any]],
        start_id: int,
        target_id: int
    ) -> List[int]:
        """
        Dijkstra algorithm over the intersection graph using geographic distance.
        """
        id_to_inter = {inter['id']: inter for inter in intersections}
        adj = {}
        for inter in intersections:
            nid = inter['id']
            adj[nid] = []
            for nxt in inter.get('connections', []):
                if nxt in id_to_inter:
                    dist = haversine_distance_km(
                        inter['lat'], inter['lng'],
                        id_to_inter[nxt]['lat'], id_to_inter[nxt]['lng']
                    )
                    adj[nid].append((nxt, dist))

        # Priority queue: (cost, current_node, path)
        pq = [(0.0, start_id, [start_id])]
        visited = set()

        while pq:
            cost, u, path = heapq.heappop(pq)
            if u == target_id:
                return path
            if u in visited:
                continue
            visited.add(u)

            for v, weight in adj.get(u, []):
                if v not in visited:
                    heapq.heappush(pq, (cost + weight, v, path + [v]))

        # Fallback if disconnected: direct line
        return [start_id, target_id]

    def preempt_signals(
        self,
        intersections: List[Dict[str, Any]],
        route: List[int]
    ) -> Dict[int, str]:
        """
        Preempt signals: determine required green direction ('NS_GREEN' or 'EW_GREEN')
        for each intersection along the emergency corridor.
        Other intersections receive holding clearance.
        """
        id_to_inter = {inter['id']: inter for inter in intersections}
        preemptions: Dict[int, str] = {}

        for k in range(len(route)):
            curr_id = route[k]
            curr_node = id_to_inter.get(curr_id)
            if not curr_node:
                continue

            # Determine predominant corridor heading at this intersection
            # Compare latitude delta vs longitude delta to next (or previous) node
            other_node = None
            if k + 1 < len(route):
                other_node = id_to_inter.get(route[k + 1])
            elif k - 1 >= 0:
                other_node = id_to_inter.get(route[k - 1])

            if other_node:
                d_lat = abs(other_node['lat'] - curr_node['lat'])
                d_lng = abs(other_node['lng'] - curr_node['lng'])
                # If latitude change is greater, corridor is North-South; else East-West
                phase = 'NS_GREEN' if d_lat >= d_lng else 'EW_GREEN'
            else:
                phase = 'NS_GREEN'

            preemptions[curr_id] = phase

        self.preempted_nodes = preemptions
        return preemptions

    def update_position(self, dt: float, speed_multiplier: float = 1.0) -> bool:
        """
        Update the emergency vehicle position along the route.
        Returns True if emergency is still active, False if completed/cleared.
        """
        if not self.active or len(self.route_coords) < 2:
            return False

        self.elapsed_sec += dt
        effective_speed_mps = (self.speed_kmh / 3.6) * speed_multiplier

        if self.current_leg_idx >= len(self.route_coords) - 1:
            # Reached destination!
            self.total_clearance_time_sec = round(self.elapsed_sec, 1)
            self.clear_emergency()
            return False

        # Current leg endpoints
        p1 = self.route_coords[self.current_leg_idx]
        p2 = self.route_coords[self.current_leg_idx + 1]

        leg_dist_km = haversine_distance_km(p1[0], p1[1], p2[0], p2[1])
        leg_dist_m = max(10.0, leg_dist_km * 1000.0)

        # Distance covered in dt
        travel_dist_m = effective_speed_mps * dt
        fraction_step = travel_dist_m / leg_dist_m

        self.leg_progress += fraction_step

        # Calculate heading
        d_lat = p2[0] - p1[0]
        d_lng = p2[1] - p1[1]
        self.heading = round(math.degrees(math.atan2(d_lng, d_lat)), 1)

        if self.leg_progress >= 1.0:
            # Move to next leg
            self.current_leg_idx += 1
            self.leg_progress = 0.0
            if self.current_leg_idx >= len(self.route_coords) - 1:
                # Finished corridor
                self.current_pos = {'lat': p2[0], 'lng': p2[1]}
                self.total_clearance_time_sec = round(self.elapsed_sec, 1)
                self.clear_emergency()
                return False
            else:
                p1_next = self.route_coords[self.current_leg_idx]
                self.current_pos = {'lat': p1_next[0], 'lng': p1_next[1]}
        else:
            # Interpolate
            lat = p1[0] + self.leg_progress * (p2[0] - p1[0])
            lng = p1[1] + self.leg_progress * (p2[1] - p1[1])
            self.current_pos = {'lat': round(lat, 6), 'lng': round(lng, 6)}

        return True

    def clear_emergency(self) -> None:
        """Clear active emergency and restore normal signal operations."""
        self.active = False
        self.preempted_nodes.clear()
        self.leg_progress = 0.0
        self.current_leg_idx = 0

    def get_route(self) -> List[int]:
        """Return list of intersection IDs on active route."""
        return list(self.route)

    def get_status(self) -> Dict[str, Any]:
        """
        Return serializable emergency status for telemetry and frontend.
        """
        type_cfg = config.EMERGENCY_TYPES.get(self.vehicle_type or 'ambulance', config.EMERGENCY_TYPES['ambulance'])
        total_legs = max(1, len(self.route_coords) - 1)
        overall_progress = round(min(100.0, ((self.current_leg_idx + self.leg_progress) / total_legs) * 100.0), 1)

        return {
            'active': self.active,
            'type': self.vehicle_type,
            'label': type_cfg.get('label', 'Emergency Vehicle'),
            'color': type_cfg.get('color', '#ff0000'),
            'priority': type_cfg.get('priority', 10),
            'route': self.route,
            'route_coords': self.route_coords,
            'current_position': self.current_pos,
            'current_leg': self.current_leg_idx,
            'heading': self.heading,
            'progress_pct': overall_progress,
            'elapsed_sec': round(self.elapsed_sec, 1),
            'total_clearance_time_sec': self.total_clearance_time_sec,
            'preempted_signals': self.preempted_nodes
        }
