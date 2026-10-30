"""
City Networks Definition for Autonomous Smart Traffic Management System.
Defines 5 Indian cities with real-world GPS coordinates and 5 key intersections each.
"""
import math
from typing import Dict, List, Any, Optional


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


CITIES: Dict[str, Dict[str, Any]] = {
    'bhopal': {
        'id': 'bhopal',
        'name': 'Bhopal',
        'state': 'Madhya Pradesh',
        'center': {'lat': 23.2450, 'lng': 77.4260},
        'zoom': 13,
        'intersections': [
            {
                'id': 0,
                'name': 'DB Mall Chowk',
                'lat': 23.2332,
                'lng': 77.4341,
                'earlier_wait_sec': 75.0,
                'connections': [1, 2, 3],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 1,
                'name': 'New Market Square',
                'lat': 23.2599,
                'lng': 77.4126,
                'earlier_wait_sec': 70.0,
                'connections': [0, 3, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 2,
                'name': 'Habibganj Station Junction',
                'lat': 23.2295,
                'lng': 77.4382,
                'earlier_wait_sec': 65.0,
                'connections': [0, 3],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 3,
                'name': 'MP Nagar Zone-I',
                'lat': 23.2260,
                'lng': 77.4340,
                'earlier_wait_sec': 80.0,
                'connections': [0, 1, 2],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 4,
                'name': 'ISBT Bhopal',
                'lat': 23.2855,
                'lng': 77.4122,
                'earlier_wait_sec': 60.0,
                'connections': [1],
                'approaches': ['N', 'S', 'E', 'W']
            }
        ]
    },
    'gwalior': {
        'id': 'gwalior',
        'name': 'Gwalior',
        'state': 'Madhya Pradesh',
        'center': {'lat': 26.2120, 'lng': 78.1780},
        'zoom': 13,
        'intersections': [
            {
                'id': 0,
                'name': 'Phool Bagh Square',
                'lat': 26.2183,
                'lng': 78.1828,
                'earlier_wait_sec': 65.0,
                'connections': [1, 3, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 1,
                'name': 'Maharaj Bada',
                'lat': 26.2124,
                'lng': 78.1779,
                'earlier_wait_sec': 75.0,
                'connections': [0, 2, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 2,
                'name': 'Jayendraganj',
                'lat': 26.2052,
                'lng': 78.1706,
                'earlier_wait_sec': 60.0,
                'connections': [1, 3],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 3,
                'name': 'City Center Mall Junction',
                'lat': 26.2098,
                'lng': 78.1858,
                'earlier_wait_sec': 70.0,
                'connections': [0, 2],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 4,
                'name': 'Bada Chowk',
                'lat': 26.2152,
                'lng': 78.1752,
                'earlier_wait_sec': 60.0,
                'connections': [0, 1],
                'approaches': ['N', 'S', 'E', 'W']
            }
        ]
    },
    'newdelhi': {
        'id': 'newdelhi',
        'name': 'New Delhi',
        'state': 'Delhi NCR',
        'center': {'lat': 28.6250, 'lng': 77.2200},
        'zoom': 13,
        'intersections': [
            {
                'id': 0,
                'name': 'ITO Junction',
                'lat': 28.6285,
                'lng': 77.2413,
                'earlier_wait_sec': 90.0,
                'connections': [1, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 1,
                'name': 'Connaught Place',
                'lat': 28.6315,
                'lng': 77.2167,
                'earlier_wait_sec': 85.0,
                'connections': [0, 2, 3, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 2,
                'name': 'AIIMS Crossing',
                'lat': 28.5672,
                'lng': 77.2100,
                'earlier_wait_sec': 90.0,
                'connections': [1, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 3,
                'name': 'Karol Bagh Square',
                'lat': 28.6519,
                'lng': 77.1905,
                'earlier_wait_sec': 75.0,
                'connections': [1, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 4,
                'name': 'Rajiv Chowk',
                'lat': 28.6328,
                'lng': 77.2197,
                'earlier_wait_sec': 80.0,
                'connections': [0, 1, 2, 3],
                'approaches': ['N', 'S', 'E', 'W']
            }
        ]
    },
    'bangalore': {
        'id': 'bangalore',
        'name': 'Bangalore',
        'state': 'Karnataka',
        'center': {'lat': 12.9650, 'lng': 77.6300},
        'zoom': 12,
        'intersections': [
            {
                'id': 0,
                'name': 'Silk Board Junction',
                'lat': 12.9172,
                'lng': 77.6227,
                'earlier_wait_sec': 90.0,
                'connections': [3, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 1,
                'name': 'KR Puram Junction',
                'lat': 13.0012,
                'lng': 77.6966,
                'earlier_wait_sec': 85.0,
                'connections': [2, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 2,
                'name': 'Hebbal Flyover',
                'lat': 13.0358,
                'lng': 77.5970,
                'earlier_wait_sec': 80.0,
                'connections': [1, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 3,
                'name': 'Koramangala Signal',
                'lat': 12.9352,
                'lng': 77.6245,
                'earlier_wait_sec': 75.0,
                'connections': [0, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 4,
                'name': 'MG Road Junction',
                'lat': 12.9756,
                'lng': 77.6068,
                'earlier_wait_sec': 70.0,
                'connections': [0, 1, 2, 3],
                'approaches': ['N', 'S', 'E', 'W']
            }
        ]
    },
    'chandigarh': {
        'id': 'chandigarh',
        'name': 'Chandigarh',
        'state': 'Punjab/Haryana',
        'center': {'lat': 30.7380, 'lng': 76.7750},
        'zoom': 13,
        'intersections': [
            {
                'id': 0,
                'name': 'Sector 17 Plaza',
                'lat': 30.7414,
                'lng': 76.7838,
                'earlier_wait_sec': 60.0,
                'connections': [1, 2, 3, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 1,
                'name': 'ISBT-43 Junction',
                'lat': 30.7268,
                'lng': 76.7598,
                'earlier_wait_sec': 65.0,
                'connections': [0, 2, 3],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 2,
                'name': 'Tribune Chowk',
                'lat': 30.7286,
                'lng': 76.7881,
                'earlier_wait_sec': 75.0,
                'connections': [0, 1, 3],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 3,
                'name': 'Sector 22 Market',
                'lat': 30.7330,
                'lng': 76.7773,
                'earlier_wait_sec': 55.0,
                'connections': [0, 1, 2, 4],
                'approaches': ['N', 'S', 'E', 'W']
            },
            {
                'id': 4,
                'name': 'PGI Roundabout',
                'lat': 30.7640,
                'lng': 76.7759,
                'earlier_wait_sec': 65.0,
                'connections': [0, 3],
                'approaches': ['N', 'S', 'E', 'W']
            }
        ]
    }
}


def get_city(city_id: str) -> Dict[str, Any]:
    """
    Get city network details by city ID.
    Falls back to 'bhopal' if city_id is invalid or missing.
    """
    normalized_id = (city_id or 'bhopal').strip().lower()
    return CITIES.get(normalized_id, CITIES['bhopal'])


def get_available_cities() -> List[Dict[str, Any]]:
    """
    Returns list of all supported cities with metadata for UI dropdown/buttons.
    """
    result = []
    for cid, data in CITIES.items():
        result.append({
            'id': cid,
            'name': data['name'],
            'state': data.get('state', ''),
            'center': data['center'],
            'zoom': data['zoom'],
            'intersection_count': len(data['intersections'])
        })
    return result
