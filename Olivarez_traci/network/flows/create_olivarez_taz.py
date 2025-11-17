"""
create_olivarez_taz.py

Create 4 TAZ zones for Olivarez intersection with Points of Interest:
- One zone for each direction of the 4 roads meeting at the intersection
- Search for POIs around each road direction using Google Places API
- Generate OD pairs between zones (excluding same-edge pairs)

Usage:
    python create_olivarez_taz.py

Output:
    traffic.taz.xml - SUMO TAZ file with 4 zones
    taz_info.json - TAZ metadata with POI information
    od_matrix.json - OD pairs between zones
"""

import googlemaps
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

# Configuration
GOOGLE_API_KEY = 'AIzaSyAVYU-p8hwxbqrsE-v14QwJklvKYlg5FVk'
SCRIPT_DIR = Path(__file__).parent
COORD_CACHE_FILE = SCRIPT_DIR / 'edge_coordinates.json'
OUTPUT_TAZ_FILE = SCRIPT_DIR / 'traffic.taz.xml'
OUTPUT_INFO_FILE = SCRIPT_DIR / 'taz_info.json'
OUTPUT_OD_FILE = SCRIPT_DIR / 'od_matrix.json'

# Search radius around each road direction (in meters)
SEARCH_RADIUS = 200  # Reduced from 300m to 200m

# Place types to search for (major traffic generators only)
PLACE_TYPES = [
    'shopping_mall',
    'university',
    'school',
    'hospital',
    'supermarket',
    'gas_station'
]

# Olivarez intersection edges with approximate coordinates
OLIVAREZ_ZONES = {
    'zone_a_malvar_north': {
        'name': 'General Malvar Street - North Direction',
        'edge_id': '-361256298#1',
        'road_name': 'General Malvar Street',
        'direction': 'North',
        'search_lat': 14.339070,  # Approximate coordinates for POI search
        'search_lng': 121.084029
    },
    'zone_b_highway_east': {
        'name': 'Pan Philippine Highway - East Direction', 
        'edge_id': '939610874',
        'road_name': 'Pan Philippine Highway',
        'direction': 'East',
        'search_lat': 14.338774,
        'search_lng': 121.078725
    },
    'zone_c_malvar_south': {
        'name': 'General Malvar Street - South Direction',
        'edge_id': '440471861#6',
        'road_name': 'General Malvar Street', 
        'direction': 'South',
        'search_lat': 14.336900,
        'search_lng': 121.084294
    },
    'zone_d_highway_west': {
        'name': 'Pan Philippine Highway - West Direction',
        'edge_id': '140483224#12',
        'road_name': 'Pan Philippine Highway',
        'direction': 'West',
        'search_lat': 14.339556,
        'search_lng': 121.087661
    }
}

class OlivarezTAZGenerator:
    def __init__(self):
        self.gmaps = googlemaps.Client(key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
        self.edge_coords = {}
        self.taz_list = []
        self.taz_info = {}
        self.od_pairs = []
        self.load_edge_coordinates()
    
    def load_edge_coordinates(self):
        """Load edge coordinates for coordinate lookups"""
        if COORD_CACHE_FILE.exists():
            with open(COORD_CACHE_FILE, 'r') as f:
                self.edge_coords = json.load(f)
            print(f"Loaded {len(self.edge_coords)} edge coordinates")
        else:
            print(f"Warning: {COORD_CACHE_FILE} not found")
    
    def search_pois_around_zone(self, zone_id, zone_data):
        """Search for POIs around a specific zone"""
        if not self.gmaps:
            print(f"  Warning: No Google API key, skipping POI search")
            return []
        
        lat = zone_data['search_lat']
        lng = zone_data['search_lng']
        
        print(f"  Searching POIs around ({lat}, {lng}) within {SEARCH_RADIUS}m")
        
        all_pois = []
        seen_place_ids = set()
        
        for place_type in PLACE_TYPES:
            try:
                results = self.gmaps.places_nearby(
                    location=(lat, lng),
                    radius=SEARCH_RADIUS,
                    type=place_type
                )
                
                places = results.get('results', [])
                
                for place in places:
                    place_id = place.get('place_id')
                    
                    if place_id in seen_place_ids:
                        continue
                    seen_place_ids.add(place_id)
                    
                    loc = place.get('geometry', {}).get('location', {})
                    
                    poi_data = {
                        'id': place_id,
                        'name': place.get('name', 'Unknown'),
                        'type': place_type,
                        'lat': loc.get('lat'),
                        'lng': loc.get('lng'),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('user_ratings_total', 0)
                    }
                    
                    all_pois.append(poi_data)
                
                print(f"    {place_type}: {len(places)} places")
                
            except Exception as e:
                print(f"    Error searching {place_type}: {e}")
        
        print(f"  Total POIs found: {len(all_pois)}")
        return all_pois
    
    def create_taz_zones(self):
        """Create TAZ zones for Olivarez intersection with POI information"""
        print(f"{'='*70}")
        print(f"CREATING OLIVAREZ INTERSECTION TAZ ZONES WITH POI")
        print(f"{'='*70}")
        
        for zone_id, zone_data in OLIVAREZ_ZONES.items():
            edge_id = zone_data['edge_id']
            name = zone_data['name']
            
            print(f"\nCreating zone: {zone_id}")
            print(f"  Name: {name}")
            print(f"  Edge: {edge_id}")
            print(f"  Road: {zone_data['road_name']} ({zone_data['direction']})")
            
            # Search for POIs around this zone
            pois = self.search_pois_around_zone(zone_id, zone_data)
            
            # Create TAZ entry
            self.taz_list.append({
                'id': zone_id,
                'edges': [edge_id],
                'source_weights': [1.0],
                'sink_weights': [1.0]
            })
            
            # Store metadata with POI information
            self.taz_info[zone_id] = {
                'name': name,
                'edge_id': edge_id,
                'road_name': zone_data['road_name'],
                'direction': zone_data['direction'],
                'edges': [edge_id],
                'search_coordinates': {
                    'lat': zone_data['search_lat'],
                    'lng': zone_data['search_lng']
                },
                'pois': pois,
                'poi_count': len(pois),
                'poi_summary': self.summarize_pois(pois)
            }
            
            print(f"  ✅ Zone created with {len(pois)} POIs")
        
        print(f"\nTotal zones created: {len(self.taz_list)}")
    
    def summarize_pois(self, pois):
        """Create summary of POIs by type"""
        summary = {}
        for poi in pois:
            poi_type = poi['type']
            summary[poi_type] = summary.get(poi_type, 0) + 1
        return summary
    
    def create_od_pairs(self):
        """Create OD pairs between TAZ zones (excluding same-edge pairs)"""
        print(f"\n{'='*70}")
        print(f"CREATING OD PAIRS")
        print(f"{'='*70}")
        
        zone_list = list(OLIVAREZ_ZONES.keys())
        od_count = 0
        
        for origin_zone in zone_list:
            origin_edge = OLIVAREZ_ZONES[origin_zone]['edge_id']
            origin_name = OLIVAREZ_ZONES[origin_zone]['name']
            
            for dest_zone in zone_list:
                dest_edge = OLIVAREZ_ZONES[dest_zone]['edge_id']
                dest_name = OLIVAREZ_ZONES[dest_zone]['name']
                
                # Skip same-edge OD pairs
                if origin_edge == dest_edge:
                    print(f"SKIPPED: {origin_zone} → {dest_zone} (same edge)")
                    continue
                
                od_count += 1
                od_pair = {
                    'id': od_count,
                    'origin_zone': origin_zone,
                    'dest_zone': dest_zone,
                    'origin_edge': origin_edge,
                    'dest_edge': dest_edge,
                    'origin_name': origin_name,
                    'dest_name': dest_name,
                    'flow_rate': 0.0
                }
                
                self.od_pairs.append(od_pair)
                
                print(f"OD {od_count:2d}: {origin_zone} → {dest_zone}")
        
        print(f"\nTotal OD pairs created: {len(self.od_pairs)}")
    
    def write_taz_file(self, output_file):
        """Write TAZ definitions to XML file"""
        print(f"\nWriting TAZ file: {output_file}")
        
        root = ET.Element('additional')
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/additional_file.xsd')
        
        for taz_data in self.taz_list:
            taz = ET.SubElement(root, 'taz')
            taz.set('id', taz_data['id'])
            taz.set('edges', ' '.join(taz_data['edges']))
            
            for edge_id, weight in zip(taz_data['edges'], taz_data['source_weights']):
                source = ET.SubElement(taz, 'tazSource')
                source.set('id', edge_id)
                source.set('weight', str(weight))
            
            for edge_id, weight in zip(taz_data['edges'], taz_data['sink_weights']):
                sink = ET.SubElement(taz, 'tazSink')
                sink.set('id', edge_id)
                sink.set('weight', str(weight))
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(output_file, encoding='UTF-8', xml_declaration=True)
        
        print(f"✅ Wrote {len(self.taz_list)} TAZ zones")
    
    def write_info_file(self, output_file):
        """Write TAZ metadata with POI information to JSON"""
        print(f"Writing TAZ info with POIs: {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(self.taz_info, f, indent=2)
        
        print(f"✅ Wrote TAZ metadata with POI data")
    
    def write_od_file(self, output_file):
        """Write OD pairs to JSON"""
        print(f"Writing OD matrix: {output_file}")
        
        od_matrix = {}
        for od_pair in self.od_pairs:
            origin_zone = od_pair['origin_zone']
            dest_zone = od_pair['dest_zone']
            
            if origin_zone not in od_matrix:
                od_matrix[origin_zone] = {}
            
            od_matrix[origin_zone][dest_zone] = {
                'origin_name': od_pair['origin_name'],
                'dest_name': od_pair['dest_name'],
                'origin_edge': od_pair['origin_edge'],
                'dest_edge': od_pair['dest_edge'],
                'flow_rate': od_pair['flow_rate']
            }
        
        with open(output_file, 'w') as f:
            json.dump(od_matrix, f, indent=2)
        
        print(f"✅ Wrote {len(self.od_pairs)} OD pairs")
    
    def print_summary(self):
        """Print generation summary with POI information"""
        print(f"\n{'='*70}")
        print(f"OLIVAREZ TAZ + POI GENERATION SUMMARY")
        print(f"{'='*70}")
        
        total_pois = sum(info['poi_count'] for info in self.taz_info.values())
        
        print(f"TAZ Zones: {len(self.taz_list)}")
        print(f"Total POIs: {total_pois}")
        print(f"OD Pairs: {len(self.od_pairs)}")
        print()
        
        for zone_id, info in self.taz_info.items():
            print(f"{zone_id}:")
            print(f"  Name: {info['name']}")
            print(f"  Edge: {info['edge_id']}")
            print(f"  POIs: {info['poi_count']}")
            
            if info['poi_summary']:
                print(f"  POI Types:")
                for poi_type, count in sorted(info['poi_summary'].items()):
                    print(f"    {poi_type}: {count}")
            print()

def main():
    """Main execution"""
    print("Olivarez Intersection TAZ + POI Generator")
    print("="*70)
    
    generator = OlivarezTAZGenerator()
    
    # Create TAZ zones with POI search
    generator.create_taz_zones()
    
    # Create OD pairs
    generator.create_od_pairs()
    
    # Write output files
    generator.write_taz_file(OUTPUT_TAZ_FILE)
    generator.write_info_file(OUTPUT_INFO_FILE)
    generator.write_od_file(OUTPUT_OD_FILE)
    
    # Print summary
    generator.print_summary()
    
    print("\n✅ Olivarez TAZ + POI generation complete!")

if __name__ == "__main__":
    main()