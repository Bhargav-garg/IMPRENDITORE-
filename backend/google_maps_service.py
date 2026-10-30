"""
Google Maps Service Integration for Autonomous Smart Traffic Management System.
Fetches real traffic data, travel times, and emergency routes via Google Maps API
with a calibrated fallback model when an API key is not configured.
"""
import time
import math
import requests
from typing import Dict, List, Any, Optional, Tuple

from backend import config
from backend.city_networks import haversine_distance_km


class GoogleMapsService:
    """
    Integrates Google Maps Distance Matrix and Directions APIs
    with in-memory TTL caching and calibrated simulation fallbacks.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key: str = (api_key or config.GOOGLE_MAPS_API_KEY or "").strip()
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.cache_ttl: float = 60.0  # seconds

    def set_api_key(self, key: str) -> None:
        """Update Google Maps API key at runtime."""
        self.api_key = (key or "").strip()
        # Invalidate old cache
        self.cache.clear()

    def _get_from_cache(self, key: str) -> Optional[Any]:
        if key in self.cache:
            ts, val = self.cache[key]
            if time.time() - ts < self.cache_ttl:
                return val
            del self.cache[key]
        return None

    def _set_in_cache(self, key: str, val: Any) -> None:
        self.cache[key] = (time.time(), val)

    def fetch_traffic_data(
        self,
        intersections: List[Dict[str, Any]],
        time_slot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch real or calibrated traffic data between all connected intersections.
        Returns travel times, distance, speed, and congestion scores.
        """
        cache_key = f"traffic_data_{len(intersections)}_{time_slot_id or 'auto'}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # Check if Google Maps API key is configured
        if self.api_key:
            try:
                live_data = self._fetch_live_distance_matrix(intersections)
                if live_data:
                    self._set_in_cache(cache_key, live_data)
                    return live_data
            except Exception as e:
                # Log error and seamlessly fall back to calibrated data
                pass

        # Calibrated fallback model
        calibrated = self._generate_calibrated_traffic(intersections, time_slot_id)
        self._set_in_cache(cache_key, calibrated)
        return calibrated

    def _fetch_live_distance_matrix(self, intersections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Query Google Maps Distance Matrix API."""
        if not intersections or len(intersections) < 2:
            return None

        origins = "|".join(f"{inter['lat']},{inter['lng']}" for inter in intersections)
        destinations = origins

        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            'origins': origins,
            'destinations': destinations,
            'departure_time': 'now',
            'traffic_model': 'best_guess',
            'key': self.api_key
        }

        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if data.get('status') != 'OK':
            return None

        matrix: List[List[Dict[str, Any]]] = []
        rows = data.get('rows', [])
        for r_idx, row in enumerate(rows):
            matrix_row = []
            elements = row.get('elements', [])
            for c_idx, elem in enumerate(elements):
                if elem.get('status') == 'OK':
                    dist_m = elem.get('distance', {}).get('value', 0)
                    dur_s = elem.get('duration', {}).get('value', 0)
                    dur_traffic_s = elem.get('duration_in_traffic', {}).get('value', dur_s)
                    ratio = dur_traffic_s / max(1, dur_s)
                    matrix_row.append({
                        'distance_km': round(dist_m / 1000.0, 2),
                        'duration_sec': dur_s,
                        'duration_in_traffic_sec': dur_traffic_s,
                        'congestion_ratio': round(ratio, 2)
                    })
                else:
                    matrix_row.append({
                        'distance_km': 0.0,
                        'duration_sec': 0,
                        'duration_in_traffic_sec': 0,
                        'congestion_ratio': 1.0
                    })
            matrix.append(matrix_row)

        return {
            'source': 'google_maps_api',
            'status': 'live',
            'matrix': matrix,
            'intersections_count': len(intersections)
        }

    def _generate_calibrated_traffic(
        self,
        intersections: List[Dict[str, Any]],
        time_slot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calibrated mathematical traffic model based on time-slot intensity:
          - Morning rush (06-09): 0.90x
          - Evening rush (15-18): 0.95x
          - Late night (00-03): 0.15x
        """
        if not time_slot_id:
            slot_id, slot_cfg = config.get_current_time_slot()
            intensity = slot_cfg['intensity']
        else:
            slot_cfg = config.TIME_SLOTS.get(time_slot_id, config.TIME_SLOTS['06-09'])
            intensity = slot_cfg['intensity']

        n = len(intersections)
        matrix: List[List[Dict[str, Any]]] = []

        # Base uncongested speed in city: 45 km/h -> 12.5 m/s
        base_speed_kmh = 42.0

        for i in range(n):
            row = []
            i_lat, i_lng = intersections[i]['lat'], intersections[i]['lng']
            for j in range(n):
                if i == j:
                    row.append({
                        'distance_km': 0.0,
                        'duration_sec': 0,
                        'duration_in_traffic_sec': 0,
                        'congestion_ratio': 1.0
                    })
                else:
                    j_lat, j_lng = intersections[j]['lat'], intersections[j]['lng']
                    dist_km = haversine_distance_km(i_lat, i_lng, j_lat, j_lng)

                    # Free flow duration
                    free_flow_sec = (dist_km / base_speed_kmh) * 3600.0

                    # Congestion multiplier: 1.0 to 2.4x depending on intensity
                    # Plus slight variation based on earlier_wait of nodes
                    earlier_i = intersections[i].get('earlier_wait_sec', 70.0)
                    earlier_j = intersections[j].get('earlier_wait_sec', 70.0)
                    node_bias = ((earlier_i + earlier_j) / 160.0)

                    congestion_ratio = 1.0 + (intensity * 1.35 * node_bias)
                    dur_traffic = free_flow_sec * congestion_ratio

                    row.append({
                        'distance_km': round(dist_km, 2),
                        'duration_sec': round(free_flow_sec, 1),
                        'duration_in_traffic_sec': round(dur_traffic, 1),
                        'congestion_ratio': round(congestion_ratio, 2)
                    })
            matrix.append(row)

        return {
            'source': 'calibrated_model',
            'status': 'calibrated',
            'intensity_factor': intensity,
            'time_slot': time_slot_id,
            'matrix': matrix,
            'intersections_count': n
        }

    def get_travel_times(
        self,
        origin: Dict[str, float],
        dest: Dict[str, float],
        time_slot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get travel time and traffic duration between two coordinates.
        """
        orig_lat, orig_lng = origin['lat'], origin['lng']
        dest_lat, dest_lng = dest['lat'], dest['lng']
        cache_key = f"tt_{orig_lat:.4f}_{orig_lng:.4f}_{dest_lat:.4f}_{dest_lng:.4f}"

        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        if self.api_key:
            try:
                url = "https://maps.googleapis.com/maps/api/distancematrix/json"
                params = {
                    'origins': f"{orig_lat},{orig_lng}",
                    'destinations': f"{dest_lat},{dest_lng}",
                    'departure_time': 'now',
                    'key': self.api_key
                }
                resp = requests.get(url, params=params, timeout=4)
                if resp.status_code == 200:
                    data = resp.json()
                    elem = data.get('rows', [{}])[0].get('elements', [{}])[0]
                    if elem.get('status') == 'OK':
                        dist_m = elem.get('distance', {}).get('value', 0)
                        dur_s = elem.get('duration', {}).get('value', 0)
                        dur_traffic_s = elem.get('duration_in_traffic', {}).get('value', dur_s)
                        result = {
                            'distance_km': round(dist_m / 1000.0, 2),
                            'duration_sec': dur_s,
                            'duration_in_traffic_sec': dur_traffic_s,
                            'congestion_ratio': round(dur_traffic_s / max(1, dur_s), 2),
                            'source': 'google_maps_api'
                        }
                        self._set_in_cache(cache_key, result)
                        return result
            except Exception:
                pass

        # Calibrated calculation
        dist_km = haversine_distance_km(orig_lat, orig_lng, dest_lat, dest_lng)
        intensity = config.TIME_SLOTS.get(time_slot_id, config.TIME_SLOTS['06-09'])['intensity']
        free_flow_sec = (dist_km / 40.0) * 3600.0
        dur_traffic = free_flow_sec * (1.0 + intensity * 1.3)
        result = {
            'distance_km': round(dist_km, 2),
            'duration_sec': round(free_flow_sec, 1),
            'duration_in_traffic_sec': round(dur_traffic, 1),
            'congestion_ratio': round(dur_traffic / max(1.0, free_flow_sec), 2),
            'source': 'calibrated_model'
        }
        self._set_in_cache(cache_key, result)
        return result

    def get_emergency_route(
        self,
        origin: Dict[str, float],
        dest: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Obtain turn-by-turn route coordinates for emergency vehicle corridor.
        Uses Directions API if key present, else computes smooth interpolation points.
        """
        orig_lat, orig_lng = origin['lat'], origin['lng']
        dest_lat, dest_lng = dest['lat'], dest['lng']
        cache_key = f"em_route_{orig_lat:.4f}_{orig_lng:.4f}_{dest_lat:.4f}_{dest_lng:.4f}"

        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        if self.api_key:
            try:
                url = "https://maps.googleapis.com/maps/api/directions/json"
                params = {
                    'origin': f"{orig_lat},{orig_lng}",
                    'destination': f"{dest_lat},{dest_lng}",
                    'mode': 'driving',
                    'key': self.api_key
                }
                resp = requests.get(url, params=params, timeout=4)
                if resp.status_code == 200:
                    data = resp.json()
                    routes = data.get('routes', [])
                    if routes:
                        points = []
                        legs = routes[0].get('legs', [])
                        for leg in legs:
                            for step in leg.get('steps', []):
                                end_loc = step.get('end_location', {})
                                if end_loc:
                                    points.append([end_loc['lat'], end_loc['lng']])
                        if points:
                            res = {'waypoints': points, 'source': 'google_directions_api'}
                            self._set_in_cache(cache_key, res)
                            return res
            except Exception:
                pass

        # Fallback straight/interpolated segment
        steps = 10
        waypoints = []
        for s in range(steps + 1):
            fraction = s / float(steps)
            lat = orig_lat + fraction * (dest_lat - orig_lat)
            lng = orig_lng + fraction * (dest_lng - orig_lng)
            waypoints.append([round(lat, 6), round(lng, 6)])

        res = {'waypoints': waypoints, 'source': 'interpolated_geometry'}
        self._set_in_cache(cache_key, res)
        return res
