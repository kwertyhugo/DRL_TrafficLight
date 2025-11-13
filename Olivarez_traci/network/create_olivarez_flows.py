"""
Generate SUMO flow file for Olivarez intersection based on:
- 4 TAZ zones (created by create_olivarez_taz.py)
- 12 OD pairs (4x3, excluding same-edge combinations)

Vehicle Distribution:
- Car: 41.3%
- Jeep: 17%
- Bus: 6%
- Truck: 13%
- Motorcycle: 11.35%
- Tricycle: 11.35%

Output:
    olivarez_flows.xml - SUMO flow definitions
    flow_summary.json - Flow statistics and metadata
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import random

# Configuration
SCRIPT_DIR = Path(__file__).parent
OD_MATRIX_FILE = SCRIPT_DIR / 'od_matrix.json'
TAZ_INFO_FILE = SCRIPT_DIR / 'taz_info.json'
NETWORK_FILE = SCRIPT_DIR / 'map.net.xml'

# Output files for different traffic categories
OUTPUT_FILES = {
    'heavy_traffic': SCRIPT_DIR / 'olivarez_flows_heavy.xml',
    'moderate_traffic': SCRIPT_DIR / 'olivarez_flows_moderate.xml', 
    'light_traffic': SCRIPT_DIR / 'olivarez_flows_light.xml'
}

OUTPUT_SUMMARY_FILE = SCRIPT_DIR / 'flow_summary_all_categories.json'

# Output files for different traffic scenarios
OUTPUT_FILES = {
    'heavy_traffic': SCRIPT_DIR / 'olivarez_flows_heavy.xml',
    'moderate_traffic': SCRIPT_DIR / 'olivarez_flows_moderate.xml',
    'light_traffic': SCRIPT_DIR / 'olivarez_flows_light.xml'
}

OUTPUT_SUMMARY_FILE = SCRIPT_DIR / 'flow_summary.json'

# Vehicle type distribution (Filipino traffic pattern)
VEHICLE_DISTRIBUTION = {
    'car': 0.413,      # 41.3%
    'jeep': 0.17,      # 17%
    'bus': 0.06,       # 6%
    'truck': 0.13,     # 13%
    'motorcycle': 0.1135,  # 11.35%
    'tricycle': 0.1135     # 11.35%
}

# Insertion rate calculation parameters
# Formula: (density × capacity × length) / trip_duration
# Capacity will be calculated from network file: (vehicle_lanes × base_capacity_per_lane × length_factor)
# Density is calculated as: (100% - speed_percentage) where speed_percentage = current_speed / max_speed
INSERTION_RATE_PARAMS = {
    'max_speed': 60,       # maximum speed in km/h (free-flowing traffic)
    'current_speed': 30,   # current average speed in km/h (adjust based on traffic conditions)
    'base_capacity_per_lane': 600,  # vehicles per hour per lane (base rate)
    'average_speed_ms': 8.33  # average speed in m/s (30 km/h) for calculating trip duration
}

# Capacity calculation factors
CAPACITY_FACTORS = {
    'highway.trunk': 1.2,      # arterial roads - higher capacity
    'highway.secondary': 1.0,  # collector roads - standard capacity  
    'highway.residential': 0.6, # local roads - lower capacity
    'default': 1.0             # fallback 
}

# Intersection coordinates (Olivarez intersection center)
INTERSECTION_COORDS = {
    'lat': 14.1644,
    'lng': 121.2441
}

# Traffic categories grouped by intensity
TRAFFIC_CATEGORIES = {
    'heavy_traffic': {
        'description': 'Heavy Traffic Conditions (Rush Hours)',
        'periods': {
            'morning_rush': {'start': 0, 'end': 7200, 'multiplier': 1.0},      # 2 hours heavy morning
            'evening_rush': {'start': 7200, 'end': 14400, 'multiplier': 1.0}   # 2 hours heavy evening
        }
    },
    'moderate_traffic': {
        'description': 'Moderate Traffic Conditions (Regular Hours)',
        'periods': {
            'midday_1': {'start': 0, 'end': 7200, 'multiplier': 0.7},          # 2 hours moderate
            'afternoon_1': {'start': 7200, 'end': 14400, 'multiplier': 0.7}    # 2 hours moderate
        }
    },
    'light_traffic': {
        'description': 'Light Traffic Conditions (Off-Peak)',
        'periods': {
            'early_morning': {'start': 0, 'end': 7200, 'multiplier': 0.3},     # 2 hours light
            'late_evening': {'start': 7200, 'end': 14400, 'multiplier': 0.3}   # 2 hours light
        }
    }
}

class OlivarezFlowGenerator:
    def __init__(self):
        self.od_matrix = {}
        self.taz_info = {}
        self.flows = []
        self.flow_summary = {}
        self.edge_capacities = {}  
        
    def parse_network_file(self):
        """Parse SUMO network file to extract edge characteristics"""
        print("Parsing network file for edge characteristics...")
        
        tree = ET.parse(NETWORK_FILE)
        root = tree.getroot()
        
        for edge in root.findall('edge'):
            edge_id = edge.get('id')
            edge_type = edge.get('type', 'default')
            
            # Count vehicle lanes (exclude pedestrian lanes)
            vehicle_lanes = 0
            total_length = 0
            
            for lane in edge.findall('lane'):
                lane_disallow = lane.get('disallow', '')
                lane_allow = lane.get('allow', '')
                
                # Skip pedestrian-only lanes
                if 'pedestrian' in lane_allow and len(lane_allow.split()) == 1:
                    continue  # This is pedestrian-only
                    
                # Count as vehicle lane if it's not pedestrian-only
                # Vehicle lanes either have no specific 'allow' or have pedestrian in 'disallow'
                if not (lane_allow and 'pedestrian' in lane_allow):
                    vehicle_lanes += 1
                    # Get length from first vehicle lane
                    if total_length == 0:
                        total_length = float(lane.get('length', 0))
            
            if vehicle_lanes > 0:
                # Calculate capacity: base_capacity × lanes × road_type_factor × length_factor
                base_capacity = INSERTION_RATE_PARAMS['base_capacity_per_lane']
                type_factor = CAPACITY_FACTORS.get(edge_type, CAPACITY_FACTORS['default'])
                
                # Length factor: longer roads can hold more vehicles
                length_factor = min(total_length / 100, 2.0)  # Cap at 2x for very long roads
                
                edge_capacity = base_capacity * vehicle_lanes * type_factor * length_factor
                
                self.edge_capacities[edge_id] = {
                    'capacity': edge_capacity,
                    'vehicle_lanes': vehicle_lanes,
                    'length': total_length,
                    'road_type': edge_type,
                    'type_factor': type_factor,
                    'length_factor': length_factor
                }
        
        print(f"PARSED {len(self.edge_capacities)} edges from network")
        
        # Print capacity details for our TAZ edges
        taz_edges = ['-361256298#1', '939610874', '440471861#6', '140483224#12']
        print("\nTAZ Edge Capacities:")
        for edge_id in taz_edges:
            if edge_id in self.edge_capacities:
                info = self.edge_capacities[edge_id]
                print(f"  {edge_id}:")
                print(f"    Type: {info['road_type']}")
                print(f"    Vehicle Lanes: {info['vehicle_lanes']}")
                print(f"    Length: {info['length']:.1f}m")
                print(f"    Capacity: {info['capacity']:.0f} veh/h")
            else:
                print(f"  {edge_id}: Not found in network")
        
    
    def load_data(self):
        """Load OD matrix and TAZ information"""
        print("Loading OD matrix and TAZ data...")
        
        # Load OD matrix
        with open(OD_MATRIX_FILE, 'r') as f:
            self.od_matrix = json.load(f)
        
        # Load TAZ info
        with open(TAZ_INFO_FILE, 'r') as f:
            self.taz_info = json.load(f)
        
        print(f"SUCCESS: Loaded {len(self.od_matrix)} origin zones")
        
        # Count total OD pairs
        total_pairs = sum(len(destinations) for destinations in self.od_matrix.values())
        print(f"SUCCESS: Loaded {total_pairs} OD pairs")
    
    def calculate_density(self):
        """Calculate density based on speed: density = (100% - speed_percentage)"""
        params = INSERTION_RATE_PARAMS
        
        # Calculate speed as percentage of maximum speed
        speed_percentage = params['current_speed'] / params['max_speed']
        
        # Density = 100% - speed_percentage
        density = 1.0 - speed_percentage
        
        print(f"Density Calculation:")
        print(f"  Max Speed: {params['max_speed']} km/h")
        print(f"  Current Speed: {params['current_speed']} km/h")
        print(f"  Speed Percentage: {speed_percentage:.1%}")
        print(f"  Density (100% - speed%): {density:.3f} veh/m")
        
        return density
    
    def calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between two points using Haversine formula"""
        import math
        
        # Convert to radians
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in meters
        r = 6371000
        return c * r
    
    def find_furthest_origin_per_taz(self):
        """Find the furthest POI/origin point for each TAZ zone"""
        print("Finding furthest origins per TAZ zone...")
        
        furthest_origins = {}
        intersection_lat = INTERSECTION_COORDS['lat']
        intersection_lng = INTERSECTION_COORDS['lng']
        
        for zone_id, zone_info in self.taz_info.items():
            max_distance = 0
            furthest_poi = None
            
            # Check POIs in this zone if available
            if 'pois' in zone_info and zone_info['pois']:
                for poi in zone_info['pois']:
                    distance = self.calculate_distance(
                        intersection_lat, intersection_lng,
                        poi['lat'],  # Direct access to lat
                        poi['lng']   # Direct access to lng
                    )
                    
                    if distance > max_distance:
                        max_distance = distance
                        furthest_poi = poi
            
            # If no POIs or very short distance, use minimum realistic distance
            if max_distance < 200:  # minimum 200m
                max_distance = 500  # default 500m
                furthest_poi = {'name': 'Default distance (no POIs found)'}
            
            furthest_origins[zone_id] = {
                'distance': max_distance,
                'poi_name': furthest_poi['name'] if furthest_poi else 'Default',
                'trip_duration': max_distance / INSERTION_RATE_PARAMS['average_speed_ms']
            }
            
            print(f"  {zone_id}: {max_distance:.0f}m to '{furthest_poi['name'] if furthest_poi else 'Default'}' ({furthest_origins[zone_id]['trip_duration']:.1f}s)")
        
        return furthest_origins
    
    def calculate_insertion_rate_per_taz(self, furthest_origins):
        """Calculate insertion rate for each TAZ using its furthest origin and real edge capacity"""
        print(f"\nCalculating insertion rates per TAZ:")
        
        # Calculate density based on current traffic speed
        density = self.calculate_density()
        
        insertion_rates = {}
        params = INSERTION_RATE_PARAMS
        
        for zone_id, origin_data in furthest_origins.items():
            # Get the edge ID for this TAZ zone
            zone_edge_id = None
            for taz_zone_id, taz_info in self.taz_info.items():
                if taz_zone_id == zone_id:
                    zone_edge_id = taz_info['edge_id']
                    break
            
            if zone_edge_id and zone_edge_id in self.edge_capacities:
                # Use real edge capacity from network
                edge_info = self.edge_capacities[zone_edge_id]
                edge_capacity = edge_info['capacity']
                
                length = origin_data['distance']
                trip_duration = origin_data['trip_duration']
                
                insertion_rate = (density * edge_capacity * length) / trip_duration
                
                insertion_rates[zone_id] = insertion_rate
                
                print(f"  {zone_id}:")
                print(f"    Edge: {zone_edge_id}")
                print(f"    Road Type: {edge_info['road_type']}")
                print(f"    Vehicle Lanes: {edge_info['vehicle_lanes']}")
                print(f"    Edge Length: {edge_info['length']:.1f}m")
                print(f"    Edge Capacity: {edge_info['capacity']:.0f} veh/h")
                print(f"    Trip Distance: {length:.0f}m")
                print(f"    Trip Duration: {trip_duration:.1f}s")
                print(f"    Density: {density:.3f} veh/m")
                print(f"    Insertion Rate: {insertion_rate:.2f} veh/h")
            else:
                # Fallback to default
                print(f"  Warning: {zone_id} edge not found, using realistic default capacity")
                default_capacity = params['base_capacity_per_lane'] * 2  # Assume 2 lanes: 3600 veh/h
                
                length = origin_data['distance']
                trip_duration = origin_data['trip_duration']
                
                insertion_rate = (density * default_capacity * length) / trip_duration
                insertion_rates[zone_id] = insertion_rate
                
                print(f"  {zone_id}:")
                print(f"    Default Capacity: {default_capacity:.0f} veh/h (2 lanes)")
                print(f"    Trip Distance: {length:.0f}m") 
                print(f"    Trip Duration: {trip_duration:.1f}s")
                print(f"    Insertion Rate: {insertion_rate:.2f} veh/h")
        
        return insertion_rates
    
    def calculate_flow_rates_per_od(self, insertion_rate_for_taz, num_destinations, period_multiplier):
        """Calculate flow probabilities per OD pair by splitting TAZ insertion rate evenly among destinations"""
        # Split total vehicles evenly among all destination zones from this origin
        vehicles_per_od_per_hour = (insertion_rate_for_taz * period_multiplier) / num_destinations
        
        # Convert to probability per second: veh/hour ÷ 3600
        probability_per_second = vehicles_per_od_per_hour / 3600
        
        # Then distribute by vehicle type
        flows_by_type = {}
        for vehicle_type, probability in VEHICLE_DISTRIBUTION.items():
            type_probability = probability_per_second * probability
            flows_by_type[vehicle_type] = round(type_probability, 6)  # Round to 6 decimal places
        
        return flows_by_type, vehicles_per_od_per_hour, probability_per_second
    
    def create_flows(self):
        print(f"\n{'='*70}")
        print("CREATING FLOWS FOR DIFFERENT TRAFFIC CATEGORIES")
        print(f"{'='*70}")
        
        # Find furthest origins per TAZ
        furthest_origins = self.find_furthest_origin_per_taz()
        
        # Calculate insertion rates per TAZ based on furthest origins
        insertion_rates = self.calculate_insertion_rate_per_taz(furthest_origins)
        
        # Store flows for each category
        self.flows_by_category = {}
        
        for category_name, category_config in TRAFFIC_CATEGORIES.items():
            print(f"\n{category_name.upper().replace('_', ' ')}: {category_config['description']}")
            
            category_flows = []
            flow_id = 0
            
            for period_name, period_config in category_config['periods'].items():
                period_multiplier = period_config['multiplier']
                
                print(f"\n  {period_name} ({period_config['start']}-{period_config['end']}s)")
                print(f"  Period multiplier: {period_multiplier}")
                
                for origin_zone, destinations in self.od_matrix.items():
                    num_destinations = len(destinations)
                    taz_insertion_rate = insertion_rates[origin_zone]
                    
                    # Split TAZ insertion rate evenly among all destinations
                    vehicle_flows, vehicles_per_od_per_hour, probability_per_second = self.calculate_flow_rates_per_od(
                        taz_insertion_rate, num_destinations, period_multiplier
                    )
                    
                    for dest_zone, od_data in destinations.items():
                        
                        # Create flows for each vehicle type
                        for vehicle_type, flow_probability in vehicle_flows.items():
                            if flow_probability > 0:  # Only create flows with positive probabilities
                                flow_id += 1
                                
                                flow = {
                                    'id': f"flow_{category_name}_{flow_id}",
                                    'from': od_data['origin_edge'],
                                    'to': od_data['dest_edge'],
                                    'begin': period_config['start'],
                                    'end': period_config['end'],
                                    'probability': flow_probability,
                                    'type': vehicle_type,
                                    'origin_zone': origin_zone,
                                    'dest_zone': dest_zone,
                                    'period': period_name,
                                    'category': category_name,
                                    'vehicles_per_hour_equivalent': vehicles_per_od_per_hour * VEHICLE_DISTRIBUTION[vehicle_type]
                                }
                                
                                category_flows.append(flow)
            
            self.flows_by_category[category_name] = category_flows
            print(f"\n  SUCCESS: {category_name}: {len(category_flows)} flows created")
        
        print(f"\n{'='*70}")
        total_flows = sum(len(flows) for flows in self.flows_by_category.values())
        print(f"TOTAL FLOWS ACROSS ALL CATEGORIES: {total_flows}")
        print(f"{'='*70}")
    
    def write_flow_files(self):
        """Write separate flow files for each traffic category"""
        print(f"\nWriting flow files for each traffic category:")
        
        for category_name, output_file in OUTPUT_FILES.items():
            flows = self.flows_by_category[category_name]
            
            print(f"\nWriting {category_name}: {output_file}")
            
            root = ET.Element('routes')
            root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
            root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
            
            # Add vehicle type definitions
            for vehicle_type in VEHICLE_DISTRIBUTION.keys():
                vtype = ET.SubElement(root, 'vType')
                vtype.set('id', vehicle_type)
                
                # Set vehicle-specific attributes
                if vehicle_type == 'car':
                    vtype.set('vClass', 'passenger')
                    vtype.set('length', '4.5')
                    vtype.set('maxSpeed', '50')
                elif vehicle_type == 'jeep':
                    vtype.set('vClass', 'bus')
                    vtype.set('length', '7.0')
                    vtype.set('maxSpeed', '40')
                elif vehicle_type == 'bus':
                    vtype.set('vClass', 'bus')
                    vtype.set('length', '12.0')
                    vtype.set('maxSpeed', '45')
                elif vehicle_type == 'truck':
                    vtype.set('vClass', 'truck')
                    vtype.set('length', '8.0')
                    vtype.set('maxSpeed', '40')
                elif vehicle_type == 'motorcycle':
                    vtype.set('vClass', 'motorcycle')
                    vtype.set('length', '2.0')
                    vtype.set('maxSpeed', '60')
                elif vehicle_type == 'tricycle':
                    vtype.set('vClass', 'motorcycle')
                    vtype.set('length', '3.0')
                    vtype.set('maxSpeed', '30')
            
            # Add flows for this category
            for flow in flows:
                flow_elem = ET.SubElement(root, 'flow')
                flow_elem.set('id', flow['id'])
                flow_elem.set('from', flow['from'])
                flow_elem.set('to', flow['to'])
                flow_elem.set('begin', str(flow['begin']))
                flow_elem.set('end', str(flow['end']))
                flow_elem.set('probability', str(flow['probability']))
                flow_elem.set('type', flow['type'])
            
            tree = ET.ElementTree(root)
            ET.indent(tree, space='  ')
            tree.write(output_file, encoding='UTF-8', xml_declaration=True)
            
            print(f"SUCCESS: Wrote {len(flows)} flows to {output_file}")

        print(f"\nSUCCESS: All flow files written successfully!")
    
    def generate_summary(self):
        """Generate flow summary statistics for all categories"""
        print(f"Writing comprehensive flow summary: {OUTPUT_SUMMARY_FILE}")
        
        # Calculate statistics for each category
        summary_by_category = {}
        
        for category_name, flows in self.flows_by_category.items():
            flows_by_type = {}
            flows_by_period = {}
            flows_by_od = {}
            
            for flow in flows:
                # By vehicle type
                vtype = flow['type']
                if vtype not in flows_by_type:
                    flows_by_type[vtype] = {'count': 0, 'total_probability': 0, 'total_veh_per_hour': 0}
                flows_by_type[vtype]['count'] += 1
                flows_by_type[vtype]['total_probability'] += flow['probability']
                flows_by_type[vtype]['total_veh_per_hour'] += flow['vehicles_per_hour_equivalent']
                
                # By time period
                period = flow['period']
                if period not in flows_by_period:
                    flows_by_period[period] = {'count': 0, 'total_probability': 0, 'total_veh_per_hour': 0}
                flows_by_period[period]['count'] += 1
                flows_by_period[period]['total_probability'] += flow['probability']
                flows_by_period[period]['total_veh_per_hour'] += flow['vehicles_per_hour_equivalent']
                
                # By OD pair
                od_key = f"{flow['origin_zone']} -> {flow['dest_zone']}"
                if od_key not in flows_by_od:
                    flows_by_od[od_key] = {'count': 0, 'total_probability': 0, 'total_veh_per_hour': 0}
                flows_by_od[od_key]['count'] += 1
                flows_by_od[od_key]['total_probability'] += flow['probability']
                flows_by_od[od_key]['total_veh_per_hour'] += flow['vehicles_per_hour_equivalent']
            
            summary_by_category[category_name] = {
                'total_flows': len(flows),
                'od_pairs': len(flows_by_od),
                'vehicle_types': len(flows_by_type),
                'time_periods': len(flows_by_period),
                'flows_by_type': flows_by_type,
                'flows_by_period': flows_by_period,
                'flows_by_od': flows_by_od
            }
        
        self.flow_summary = {
            'generation_info': {
                'total_categories': len(TRAFFIC_CATEGORIES),
                'total_flows_all_categories': sum(len(flows) for flows in self.flows_by_category.values()),
                'categories': list(TRAFFIC_CATEGORIES.keys())
            },
            'insertion_rate_params': INSERTION_RATE_PARAMS,
            'traffic_categories': TRAFFIC_CATEGORIES,
            'vehicle_distribution': VEHICLE_DISTRIBUTION,
            'summary_by_category': summary_by_category,
            'output_files': {k: str(v) for k, v in OUTPUT_FILES.items()}
        }
        
        with open(OUTPUT_SUMMARY_FILE, 'w') as f:
            json.dump(self.flow_summary, f, indent=2)
        
        print(f"SUCCESS: Wrote comprehensive flow summary")
    
    def print_summary(self):
        """Print generation summary for all traffic categories"""
        print(f"\n{'='*70}")
        print("OLIVAREZ FLOW GENERATION SUMMARY - ALL CATEGORIES")
        print(f"{'='*70}")
        
        total_flows = sum(len(flows) for flows in self.flows_by_category.values())
        
        print(f"Total Categories: {len(TRAFFIC_CATEGORIES)}")
        print(f"Total Flows (All Categories): {total_flows}")
        print(f"Vehicle Types: {len(VEHICLE_DISTRIBUTION)}")
        print()
        
        # Print summary for each category
        for category_name, flows in self.flows_by_category.items():
            category_info = TRAFFIC_CATEGORIES[category_name]
            print(f"{category_name.upper().replace('_', ' ')}:")
            print(f"  Description: {category_info['description']}")
            print(f"  Total Flows: {len(flows)}")
            print(f"  Time Periods: {len(category_info['periods'])}")
            
            for period_name, period_config in category_info['periods'].items():
                duration = (period_config['end'] - period_config['start']) / 3600
                print(f"    {period_name}: {period_config['start']}-{period_config['end']}s ({duration:.1f}h, {period_config['multiplier']}x)")
            print()
        
        print("Vehicle Distribution:")
        for vtype, percentage in VEHICLE_DISTRIBUTION.items():
            # Sum across all categories for this vehicle type
            total_count = 0
            total_prob = 0
            total_veh = 0
            
            for flows in self.flows_by_category.values():
                for flow in flows:
                    if flow['type'] == vtype:
                        total_count += 1
                        total_prob += flow['probability']
                        total_veh += flow['vehicles_per_hour_equivalent']
            
            print(f"  {vtype:12}: {percentage:6.1%} ({total_count:3d} flows total, {total_prob:.6f} prob/s, {total_veh:.0f} veh/h equiv)")
        
        print(f"\n{'='*70}")

def main():
    """Main execution"""
    print("Olivarez Intersection Flow Generator - Multiple Traffic Categories")
    print("="*70)
    
    generator = OlivarezFlowGenerator()
    
    # Parse network file to get real edge characteristics
    generator.parse_network_file()
    
    # Load existing data
    generator.load_data()
    
    # Create flows for all categories
    generator.create_flows()
    
    # Write separate flow files for each category
    generator.write_flow_files()
    
    # Generate comprehensive summary
    generator.generate_summary()
    
    # Print summary
    generator.print_summary()
    
    print(f"\nSUCCESS: Olivarez flow generation complete!")
    print(f"Files created:")
    for category, filename in OUTPUT_FILES.items():
        print(f"  - {category}: {filename}")
    print(f"  - Summary: {OUTPUT_SUMMARY_FILE}")
    
    print(f"\nTo use these files, update your SUMO config to reference:")
    print(f"  Heavy Traffic: olivarez_flows_heavy.xml")
    print(f"  Moderate Traffic: olivarez_flows_moderate.xml") 
    print(f"  Light Traffic: olivarez_flows_light.xml")

if __name__ == "__main__":
    main()