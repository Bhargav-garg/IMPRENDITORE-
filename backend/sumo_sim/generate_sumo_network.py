"""
SUMO Network Generator for Autonomous Smart Traffic Management.
Builds nodes, edges, routes, and compiles the .net.xml network using netconvert.
"""
import os
import subprocess
import xml.etree.ElementTree as ET

SUMO_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_BIN = os.path.join(os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\SUMO'), 'bin')
NETCONVERT = os.path.join(SUMO_BIN, 'netconvert.exe')


def build_nodes():
    """Create node file for 5 intersections in a coordinated urban corridor."""
    nodes = ET.Element('nodes')
    
    # 5 Major Intersections (Junctions with Traffic Lights)
    intersections = [
        ('J0', 0.0, 500.0, 'traffic_light'),
        ('J1', 500.0, 500.0, 'traffic_light'),
        ('J2', 1000.0, 500.0, 'traffic_light'),
        ('J3', 500.0, 0.0, 'traffic_light'),
        ('J4', 500.0, 1000.0, 'traffic_light'),
    ]
    
    # Perimeter Entry/Exit Nodes
    perimeters = [
        ('W_IN', -500.0, 500.0, 'priority'),
        ('W_OUT', -500.0, 500.0, 'priority'),
        ('E_IN', 1500.0, 500.0, 'priority'),
        ('E_OUT', 1500.0, 500.0, 'priority'),
        ('S_IN', 500.0, -500.0, 'priority'),
        ('S_OUT', 500.0, -500.0, 'priority'),
        ('N_IN', 500.0, 1500.0, 'priority'),
        ('N_OUT', 500.0, 1500.0, 'priority'),
    ]
    
    for nid, x, y, ntype in intersections + perimeters:
        ET.SubElement(nodes, 'node', id=nid, x=str(x), y=str(y), type=ntype)
        
    tree = ET.ElementTree(nodes)
    tree.write(os.path.join(SUMO_DIR, 'city.nod.xml'), encoding='utf-8', xml_declaration=True)
    print("[SUMO] Generated city.nod.xml")


def build_edges():
    """Create bidirectional multi-lane edges between nodes."""
    edges = ET.Element('edges')
    
    # Corridor connections with 3 lanes each
    road_segments = [
        ('W_to_J0', 'W_IN', 'J0', 3, '13.89'),
        ('J0_to_W', 'J0', 'W_OUT', 3, '13.89'),
        
        ('J0_to_J1', 'J0', 'J1', 3, '16.67'),
        ('J1_to_J0', 'J1', 'J0', 3, '16.67'),
        
        ('J1_to_J2', 'J1', 'J2', 3, '16.67'),
        ('J2_to_J1', 'J2', 'J1', 3, '16.67'),
        
        ('J2_to_E', 'J2', 'E_OUT', 3, '13.89'),
        ('E_to_J2', 'E_IN', 'J2', 3, '13.89'),
        
        ('S_to_J3', 'S_IN', 'J3', 3, '13.89'),
        ('J3_to_S', 'J3', 'S_OUT', 3, '13.89'),
        
        ('J3_to_J1', 'J3', 'J1', 3, '16.67'),
        ('J1_to_J3', 'J1', 'J3', 3, '16.67'),
        
        ('J1_to_J4', 'J1', 'J4', 3, '16.67'),
        ('J4_to_J1', 'J4', 'J1', 3, '16.67'),
        
        ('J4_to_N', 'J4', 'N_OUT', 3, '13.89'),
        ('N_to_J4', 'N_IN', 'J4', 3, '13.89'),
    ]
    
    for eid, frm, to, lanes, speed in road_segments:
        ET.SubElement(edges, 'edge', id=eid, **{'from': frm, 'to': to, 'numLanes': str(lanes), 'speed': speed})
        
    tree = ET.ElementTree(edges)
    tree.write(os.path.join(SUMO_DIR, 'city.edg.xml'), encoding='utf-8', xml_declaration=True)
    print("[SUMO] Generated city.edg.xml")


def build_routes():
    """Create vehicle types (cars, bikes, trucks, buses, ambulance) and routes."""
    routes = ET.Element('routes')
    
    # Vehicle Types with distinct colors and dynamics
    vtypes = [
        {'id': 'car', 'accel': '2.6', 'decel': '4.5', 'sigma': '0.5', 'length': '4.5', 'maxSpeed': '14.0', 'color': '52,152,219', 'vClass': 'passenger'},
        {'id': 'bike', 'accel': '3.2', 'decel': '5.0', 'sigma': '0.7', 'length': '2.0', 'maxSpeed': '12.0', 'color': '46,204,113', 'vClass': 'motorcycle'},
        {'id': 'truck', 'accel': '1.2', 'decel': '3.5', 'sigma': '0.3', 'length': '12.0', 'maxSpeed': '9.0', 'color': '231,76,60', 'vClass': 'truck'},
        {'id': 'bus', 'accel': '1.5', 'decel': '4.0', 'sigma': '0.4', 'length': '11.0', 'maxSpeed': '10.0', 'color': '230,126,34', 'vClass': 'bus'},
        {'id': 'ambulance', 'accel': '4.0', 'decel': '6.0', 'sigma': '0.1', 'length': '6.5', 'maxSpeed': '22.0', 'color': '255,0,0', 'vClass': 'emergency'},
        {'id': 'firetruck', 'accel': '3.5', 'decel': '5.5', 'sigma': '0.1', 'length': '8.5', 'maxSpeed': '20.0', 'color': '255,69,0', 'vClass': 'emergency'},
    ]
    
    for vt in vtypes:
        ET.SubElement(routes, 'vType', **vt)
        
    # Defined arterial routes
    edge_routes = [
        ('route_W_E', 'W_to_J0 J0_to_J1 J1_to_J2 J2_to_E'),
        ('route_E_W', 'E_to_J2 J2_to_J1 J1_to_J0 J0_to_W'),
        ('route_S_N', 'S_to_J3 J3_to_J1 J1_to_J4 J4_to_N'),
        ('route_N_S', 'N_to_J4 J4_to_J1 J1_to_J3 J3_to_S'),
    ]
    
    for rid, edges_str in edge_routes:
        ET.SubElement(routes, 'route', id=rid, edges=edges_str)
        
    tree = ET.ElementTree(routes)
    tree.write(os.path.join(SUMO_DIR, 'city.rou.xml'), encoding='utf-8', xml_declaration=True)
    print("[SUMO] Generated city.rou.xml")


def compile_network():
    """Run SUMO netconvert to compile .net.xml."""
    cmd = [
        NETCONVERT,
        '--node-files', os.path.join(SUMO_DIR, 'city.nod.xml'),
        '--edge-files', os.path.join(SUMO_DIR, 'city.edg.xml'),
        '--output-file', os.path.join(SUMO_DIR, 'city.net.xml'),
        '--junctions.join', 'true',
        '--tls.guess', 'true'
    ]
    print(f"[SUMO] Running netconvert: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("[SUMO] Successfully generated city.net.xml")
    else:
        print(f"[SUMO] netconvert error: {result.stderr}")
        

def create_sumo_config():
    """Create .sumocfg configuration file."""
    cfg = f"""<configuration>
    <input>
        <net-file value="city.net.xml"/>
        <route-files value="city.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="36000"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <collision.action value="none"/>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
"""
    with open(os.path.join(SUMO_DIR, 'city.sumocfg'), 'w', encoding='utf-8') as f:
        f.write(cfg)
    print("[SUMO] Generated city.sumocfg")


if __name__ == '__main__':
    os.makedirs(SUMO_DIR, exist_ok=True)
    build_nodes()
    build_edges()
    build_routes()
    compile_network()
    create_sumo_config()
    print("[SUMO] Full SUMO network compiled successfully!")
