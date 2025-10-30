#!/usr/bin/env python3
"""
Generate flows using vehicle-suitable edges only
This filters out internal, pedestrian-only, and restricted edges
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import random

print("GENERATING IMPROVED EDGE-TO-EDGE FLOWS")
print("=" * 40)

# File paths
SCRIPT_DIR = Path(__file__).parent
OD_MATRIX_FILE = SCRIPT_DIR / 'network' / 'od_matrix.json'
TAZ_FILE = SCRIPT_DIR / 'network' / 'traffic.taz.xml'
NETWORK_FILE = SCRIPT_DIR / 'network' / 'map.net.xml'
OUTPUT_FLOWS_FILE = SCRIPT_DIR / 'demand' / 'flows_edge_based_filtered.flows.xml'

# First, get all vehicle-suitable edges from network
print("Analyzing network for vehicle-suitable edges...")
tree = ET.parse(NETWORK_FILE)
root = tree.getroot()

vehicle_edges = []
for edge in root.findall('edge'):
    edge_id = edge.get('id')
    edge_func = edge.get('function', 'normal')
    
    # Skip internal edges and walkingarea
    if edge_func in ['internal', 'walkingarea']:
        continue
    
    # Check if edge allows vehicles
    lanes = edge.findall('lane')
    allows_vehicles = False
    
    for lane in lanes:
        allow = lane.get('allow', 'all')
        disallow = lane.get('disallow', 'none')
        
        # Check if vehicles are allowed
        if allow == 'all':
            # Check what's disallowed - if passenger/private not disallowed, it's good
            if 'passenger' not in disallow and 'private' not in disallow:
                allows_vehicles = True
                break
        elif 'passenger' in allow or 'private' in allow:
            allows_vehicles = True
            break
    
    if allows_vehicles:
        vehicle_edges.append(edge_id)

print(f"Found {len(vehicle_edges)} vehicle-suitable edges")

# Load OD matrix
with open(OD_MATRIX_FILE, 'r') as f:
    od_matrix = json.load(f)

# Load TAZ file to get edge mappings
taz_edges = {}
with open(TAZ_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

tree = ET.fromstring(content)
for taz in tree.findall('taz'):
    taz_id = taz.get('id')
    edges_str = taz.get('edges', '')
    edges_list = edges_str.split() if edges_str else []
    
    # Filter to only vehicle-suitable edges
    filtered_edges = [edge for edge in edges_list if edge in vehicle_edges]
    
    if filtered_edges:
        taz_edges[taz_id] = filtered_edges

print(f"Loaded {len(od_matrix)} OD pairs")
print(f"Loaded {len(taz_edges)} TAZ zones with filtered vehicle edges")

# Generate flows using filtered edges
flows = []
flow_id = 0
skipped_no_edges = 0

for origin_taz, destinations in od_matrix.items():
    if origin_taz not in taz_edges or not taz_edges[origin_taz]:
        continue
    
    origin_edges = taz_edges[origin_taz]
    
    for dest_taz, od_data in destinations.items():
        if dest_taz not in taz_edges or not taz_edges[dest_taz]:
            skipped_no_edges += 1
            continue
        
        dest_edges = taz_edges[dest_taz]
        time_rates = od_data.get('time_based_rates', {})
        
        for time_str, rate_data in time_rates.items():
            hour = int(time_str.split(':')[0])
            insertion_rate = rate_data.get('insertion_rate', 0)
            
            if insertion_rate > 0:
                # Calculate begin time
                begin_time = (hour - 4) * 1000  # 4 AM = step 0
                end_time = begin_time + 1000
                
                # Pick random origin and destination edges (from filtered lists)
                from_edge = random.choice(origin_edges)
                to_edge = random.choice(dest_edges)
                
                flows.append({
                    'id': f'flow_{flow_id}',
                    'from': from_edge,
                    'to': to_edge,
                    'begin': str(begin_time),
                    'end': str(end_time),
                    'vehsPerHour': str(max(1, int(insertion_rate))),
                    'departLane': 'best',
                    'departSpeed': 'max'
                })
                
                flow_id += 1

print(f"Generated {len(flows)} filtered edge-to-edge flows")
print(f"Skipped {skipped_no_edges} OD pairs with no vehicle edges")

# Sort flows by begin time
flows.sort(key=lambda x: float(x['begin']))
print(f"Flows sorted from time {flows[0]['begin']} to {flows[-1]['begin']}")

# Write flows file
OUTPUT_FLOWS_FILE.parent.mkdir(exist_ok=True)

with open(OUTPUT_FLOWS_FILE, 'w', encoding='UTF-8') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ')
    f.write('xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n')
    
    for flow in flows:
        f.write(f'    <flow id="{flow["id"]}" ')
        f.write(f'from="{flow["from"]}" ')     # from edge (not fromTaz)
        f.write(f'to="{flow["to"]}" ')         # to edge (not toTaz)
        f.write(f'begin="{flow["begin"]}" ')
        f.write(f'end="{flow["end"]}" ')
        f.write(f'vehsPerHour="{flow["vehsPerHour"]}" ')
        f.write(f'departLane="{flow["departLane"]}" ')
        f.write(f'departSpeed="{flow["departSpeed"]}" />\n')
    
    f.write('\n</routes>\n')

print(f"SUCCESS: Wrote {len(flows)} filtered edge-based flows to {OUTPUT_FLOWS_FILE}")
print("These flows use only vehicle-suitable edges and should have fewer routing errors!")