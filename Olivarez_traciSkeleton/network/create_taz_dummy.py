"""
create_taz_dummy.py

Creates a dummy TAZ file with 10 test zones for testing the traffic generation pipeline.
No Google API required - uses predefined test locations.

Usage:
    python create_taz_dummy.py

Output:
    traffic.taz.xml - SUMO TAZ file with 10 test zones
    taz_info.json - TAZ metadata for later use
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import random

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
COORD_CACHE_FILE = SCRIPT_DIR / 'edge_coordinates.json'
OUTPUT_TAZ_FILE = SCRIPT_DIR / 'traffic.taz.xml'
OUTPUT_INFO_FILE = SCRIPT_DIR / 'taz_info.json'

# Dummy test places (10 zones for testing)
DUMMY_PLACES = [
    {'name': 'Test Shopping Mall', 'type': 'shopping_mall', 'lat': 14.340, 'lng': 121.070},
    {'name': 'Test University', 'type': 'university', 'lat': 14.345, 'lng': 121.075},
    {'name': 'Test Hospital', 'type': 'hospital', 'lat': 14.350, 'lng': 121.080},
    {'name': 'Test School', 'type': 'school', 'lat': 14.355, 'lng': 121.085},
    {'name': 'Test Transit Station', 'type': 'transit_station', 'lat': 14.360, 'lng': 121.090},
    {'name': 'Test Park', 'type': 'park', 'lat': 14.335, 'lng': 121.065},
    {'name': 'Test Office Building', 'type': 'point_of_interest', 'lat': 14.338, 'lng': 121.072},
    {'name': 'Test Restaurant Area', 'type': 'point_of_interest', 'lat': 14.342, 'lng': 121.077},
    {'name': 'Test Residential Area', 'type': 'point_of_interest', 'lat': 14.348, 'lng': 121.082},
    {'name': 'Test Commercial Area', 'type': 'point_of_interest', 'lat': 14.352, 'lng': 121.087}
]

class DummyTAZGenerator:
    def __init__(self):
        self.edge_coords = {}
        self.taz_list = []
        self.taz_info = {}
    
    def load_edge_coordinates(self, coord_file):
        """Load pre-parsed edge coordinates"""
        print(f"Loading edge coordinates from: {coord_file}")
        
        if not Path(coord_file).exists():
            print(f"ERROR: {coord_file} not found. Run parse_network.py first.")
            sys.exit(1)
        
        with open(coord_file, 'r') as f:
            data = json.load(f)
            self.edge_coords = data
        
        print(f"Loaded {len(self.edge_coords):,} edges")
    
    def calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between two coordinates (Haversine formula)"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlng/2) * math.sin(dlng/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def project_to_latlon(self, x, y):
        """Convert SUMO coordinates to lat/lng using network bounds"""
        # Network bounds for the SUMO map (from original create_taz.py)
        min_lat, max_lat = 14.316008, 14.363847
        min_lng, max_lng = 121.054103, 121.098550
        
        # Get network coordinate bounds
        all_coords = []
        for edge_id, edge_data in self.edge_coords.items():
            # Add start and end points
            if 'start' in edge_data:
                all_coords.append((edge_data['start']['x'], edge_data['start']['y']))
            if 'end' in edge_data:
                all_coords.append((edge_data['end']['x'], edge_data['end']['y']))
        
        if not all_coords:
            return None, None
        
        min_x = min(coord[0] for coord in all_coords)
        max_x = max(coord[0] for coord in all_coords)
        min_y = min(coord[1] for coord in all_coords)
        max_y = max(coord[1] for coord in all_coords)
        
        # Linear interpolation
        lat = min_lat + (y - min_y) / (max_y - min_y) * (max_lat - min_lat)
        lng = min_lng + (x - min_x) / (max_x - min_x) * (max_lng - min_lng)
        
        return lat, lng
    
    def find_nearest_edges(self, lat, lng, num_edges=5, max_distance=1000):
        """Find nearest SUMO edges to a lat/lng coordinate"""
        distances = []
        
        for edge_id, edge_data in self.edge_coords.items():
            min_dist = float('inf')
            
            # Check both start and end points of the edge
            for point_key in ['start', 'end']:
                if point_key in edge_data:
                    point = edge_data[point_key]
                    # Convert SUMO coordinate to lat/lng
                    edge_lat, edge_lng = self.project_to_latlon(point['x'], point['y'])
                    if edge_lat is None:
                        continue
                    
                    # Calculate distance
                    dist = self.calculate_distance(lat, lng, edge_lat, edge_lng)
                    min_dist = min(min_dist, dist)
            
            if min_dist < max_distance:
                distances.append((edge_id, min_dist))
        
        # Sort by distance and return top N
        distances.sort(key=lambda x: x[1])
        return distances[:num_edges]
    
    def create_dummy_taz(self):
        """Create TAZ entries from dummy test places"""
        print(f"\n{'='*70}")
        print(f"CREATING DUMMY TAZ ZONES FOR TESTING")
        print(f"{'='*70}\n")
        
        successful_matches = 0
        failed_matches = 0
        
        for idx, place in enumerate(DUMMY_PLACES, 1):
            place_name = place['name']
            place_type = place['type']
            lat = place['lat']
            lng = place['lng']
            
            print(f"[{idx}/{len(DUMMY_PLACES)}] {place_name} ({place_type})")
            print(f"  Location: ({lat}, {lng})")
            
            # Find nearest edges
            nearest_edges = self.find_nearest_edges(lat, lng, num_edges=3, max_distance=1000)
            
            if not nearest_edges:
                print(f"  ❌ No nearby edges found")
                failed_matches += 1
                continue
            
            edge_ids = [edge_id for edge_id, dist in nearest_edges]
            distances = [dist for edge_id, dist in nearest_edges]
            
            print(f"  ✅ Matched to {len(edge_ids)} edges:")
            for edge_id, dist in nearest_edges:
                print(f"     - {edge_id} ({dist:.1f}m)")
            
            # Create TAZ ID
            taz_id = f"dummy_{place_type}_{idx}"
            
            # Store TAZ data
            self.taz_list.append({
                'id': taz_id,
                'edges': edge_ids,
                'source_weights': [1.0] * len(edge_ids),
                'sink_weights': [1.0] * len(edge_ids)
            })
            
            # Store metadata with dummy ratings
            self.taz_info[taz_id] = {
                'name': place_name,
                'type': place_type,
                'lat': lat,
                'lng': lng,
                'edges': edge_ids,
                'distances': distances,
                'rating': random.uniform(3.5, 4.8),  # Dummy rating
                'user_ratings_total': random.randint(50, 1000)  # Dummy review count
            }
            
            successful_matches += 1
        
        print(f"\n{'='*70}")
        print(f"Successful matches: {successful_matches}")
        print(f"Failed matches: {failed_matches}")
        print(f"Total TAZ zones created: {len(self.taz_list)}")
        print(f"{'='*70}")
    
    def write_taz_file(self, output_file):
        """Write TAZ definitions to XML file"""
        print(f"\nWriting TAZ file: {output_file}")
        
        root = ET.Element('additional')
        # Remove schema validation to avoid "can't read local schema" warnings
        # SUMO will still process the file correctly without schema validation
        
        for taz_data in self.taz_list:
            taz = ET.SubElement(root, 'taz')
            taz.set('id', taz_data['id'])
            taz.set('edges', ' '.join(taz_data['edges']))
            
            # Add source edges (origins)
            for edge_id, weight in zip(taz_data['edges'], taz_data['source_weights']):
                source = ET.SubElement(taz, 'tazSource')
                source.set('id', edge_id)
                source.set('weight', str(weight))
            
            # Add sink edges (destinations)
            for edge_id, weight in zip(taz_data['edges'], taz_data['sink_weights']):
                sink = ET.SubElement(taz, 'tazSink')
                sink.set('id', edge_id)
                sink.set('weight', str(weight))
        
        # Write to file with proper formatting
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        print(f"✅ TAZ file written: {len(self.taz_list)} zones")
    
    def write_info_file(self, output_file):
        """Write TAZ metadata to JSON file"""
        print(f"Writing TAZ info file: {output_file}")
        
        info_data = {
            'generated_at': datetime.now().isoformat(),
            'total_zones': len(self.taz_list),
            'dummy_data': True,
            'taz_zones': self.taz_info
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Info file written: {len(self.taz_info)} zones")
    
    def print_summary(self):
        """Print generation summary"""
        print(f"\n{'='*70}")
        print(f"DUMMY TAZ GENERATION SUMMARY")
        print(f"{'='*70}")
        
        # Count by type
        type_counts = {}
        for taz_id, info in self.taz_info.items():
            place_type = info['type']
            type_counts[place_type] = type_counts.get(place_type, 0) + 1
        
        print(f"Total zones created: {len(self.taz_list)}")
        print(f"Breakdown by type:")
        for place_type, count in sorted(type_counts.items()):
            print(f"  - {place_type}: {count}")
        
        # Calculate total OD pairs
        n_zones = len(self.taz_list)
        od_pairs = n_zones * (n_zones - 1)
        print(f"\nTotal O-D pairs: {od_pairs}")
        print(f"This is perfect for testing! 🎯")
        
        print(f"\nFiles created:")
        print(f"  - {OUTPUT_TAZ_FILE}")
        print(f"  - {OUTPUT_INFO_FILE}")

def main():
    print(f"{'='*70}")
    print(f"DUMMY TAZ GENERATOR (10 TEST ZONES)")
    print(f"{'='*70}\n")
    
    # Initialize
    generator = DummyTAZGenerator()
    
    # Step 1: Load edge coordinates
    generator.load_edge_coordinates(COORD_CACHE_FILE)
    
    # Step 2: Create dummy TAZ zones
    generator.create_dummy_taz()
    
    # Step 3: Write output files
    generator.write_taz_file(OUTPUT_TAZ_FILE)
    generator.write_info_file(OUTPUT_INFO_FILE)
    
    # Step 4: Print summary
    generator.print_summary()
    
    print(f"\nDone! Now you can test with: python generate_od_traffic.py")
    print(f"Expected API calls: ~540 (10 zones = 90 OD pairs × 6 hours)")

if __name__ == '__main__':
    import math
    from datetime import datetime
    main()