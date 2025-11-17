"""
create_taz.py

Create Traffic Analysis Zones (TAZ) by:
1. Getting Points of Interest from Google Places API
2. Matching them to nearest SUMO edges
3. Generating TAZ file for SUMO

Requires:
- edge_coordinates.json (run parse_network.py first)
- Google Maps API key with Places API enabled

Usage:
    python create_taz.py

Output:
    traffic.taz.xml - SUMO TAZ file
    taz_info.json - TAZ metadata for later use
"""

import googlemaps
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

# Configuration
GOOGLE_API_KEY = 'AIzaSyAVYU-p8hwxbqrsE-v14QwJklvKYlg5FVk'
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
COORD_CACHE_FILE = SCRIPT_DIR / 'edge_coordinates.json'
OUTPUT_TAZ_FILE = SCRIPT_DIR / 'traffic.taz.xml'
OUTPUT_INFO_FILE = SCRIPT_DIR / 'taz_info.json'

# Search configuration
# The area will be AUTO-CALCULATED from your network boundaries
# Buffer adds extra margin around network (in meters)
SEARCH_BUFFER = 500  # Add 500m around network edges

# Place types to search for (Google Places API types)
PLACE_TYPES = [
    'shopping_mall',
    'university',
    'school',
    'hospital',
    'transit_station',
    'airport',
    'train_station',
    'bus_station',
    'park',
    'stadium',
    'point_of_interest'
]

# Limit to most relevant zones
MAX_TAZ_ZONES = 50

# Place type priority weights (higher = more important for traffic generation)
PLACE_TYPE_WEIGHTS = {
    'shopping_mall': 10,
    'university': 9,
    'hospital': 8,
    'school': 7,
    'transit_station': 9,
    'airport': 10,
    'train_station': 9,
    'bus_station': 8,
    'stadium': 6,
    'park': 4,
    'point_of_interest': 3
}

class TAZGenerator:
    def __init__(self, api_key):
        self.gmaps = googlemaps.Client(key=api_key) if api_key != 'YOUR_API_KEY_HERE' else None
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
            self.edge_coords = json.load(f)
        
        print(f"Loaded {len(self.edge_coords):,} edges")
    
    def get_places_from_google(self, location, radius, place_types):
        """
        Get Points of Interest from Google Places API
        
        Args:
            location: {'lat': float, 'lng': float}
            radius: Search radius in meters
            place_types: List of place types to search
            
        Returns:
            List of places with name, location, type
        """
        if not self.gmaps:
            print("ERROR: Google Maps API key required")
            print("Set GOOGLE_API_KEY in the configuration")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print(f"SEARCHING FOR PLACES OF INTEREST")
        print(f"{'='*70}")
        print(f"Center: ({location['lat']}, {location['lng']})")
        print(f"Radius: {radius}m")
        print(f"Types: {', '.join(place_types)}\n")
        
        all_places = []
        seen_place_ids = set()
        
        for place_type in place_types:
            print(f"Searching for: {place_type}...")
            
            try:
                results = self.gmaps.places_nearby(
                    location=(location['lat'], location['lng']),
                    radius=radius,
                    type=place_type
                )
                
                places = results.get('results', [])
                
                for place in places:
                    place_id = place.get('place_id')
                    
                    # Avoid duplicates
                    if place_id in seen_place_ids:
                        continue
                    seen_place_ids.add(place_id)
                    
                    loc = place.get('geometry', {}).get('location', {})
                    
                    place_data = {
                        'id': place_id,
                        'name': place.get('name', 'Unknown'),
                        'type': place_type,
                        'lat': loc.get('lat'),
                        'lng': loc.get('lng'),
                        'types': place.get('types', []),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('user_ratings_total', 0)
                    }
                    
                    all_places.append(place_data)
                
                print(f"  Found {len(places)} places")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"\nTotal unique places found: {len(all_places)}")
        return all_places
    
    def rank_and_filter_places(self, places, max_zones=MAX_TAZ_ZONES):
        """
        Rank places by relevance and select top zones
        
        Ranking criteria:
        1. Place type weight (from PLACE_TYPE_WEIGHTS)
        2. Google rating (0-5 stars)
        3. Number of user ratings (popularity indicator)
        
        Args:
            places: List of place dictionaries
            max_zones: Maximum number of zones to keep
            
        Returns:
            Filtered list of top-ranked places
        """
        print(f"\n{'='*70}")
        print(f"RANKING AND FILTERING PLACES TO TOP {max_zones}")
        print(f"{'='*70}")
        
        if len(places) <= max_zones:
            print(f"Found {len(places)} places, keeping all (within limit)")
            return places
        
        # Calculate relevance score for each place
        for place in places:
            place_type = place['type']
            rating = place.get('rating', 0)
            user_ratings = place.get('user_ratings_total', 0)
            
            # Base score from place type importance
            type_weight = PLACE_TYPE_WEIGHTS.get(place_type, 1)
            
            # Rating bonus (0-5 stars)
            rating_bonus = rating * 2  # Max 10 points
            
            # Popularity bonus (logarithmic scale to avoid extreme values)
            popularity_bonus = math.log10(max(user_ratings, 1)) * 2  # Max ~6 points for 1M ratings
            
            # Calculate final relevance score
            relevance_score = type_weight + rating_bonus + popularity_bonus
            place['relevance_score'] = relevance_score
        
        # Sort by relevance score (descending)
        places_sorted = sorted(places, key=lambda x: x['relevance_score'], reverse=True)
        
        # Take top N places
        top_places = places_sorted[:max_zones]
        
        print(f"Selected top {len(top_places)} places out of {len(places)} total")
        print(f"\nTop 10 most relevant places:")
        for i, place in enumerate(top_places[:10], 1):
            score = place['relevance_score']
            rating = place.get('rating', 0)
            ratings_count = place.get('user_ratings_total', 0)
            print(f"  {i:2d}. {place['name'][:40]:40} [{place['type']:15}] "
                  f"Score: {score:5.1f} (★{rating:.1f}, {ratings_count:,} reviews)")
        
        if len(top_places) > 10:
            print(f"  ... and {len(top_places) - 10} more")
        
        print(f"\nFiltered out {len(places) - len(top_places)} lower-priority places")
        return top_places
    
    def calculate_distance(self, lat1, lng1, lat2, lng2):
        """
        Calculate distance between two coordinates (Haversine formula)
        Returns distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lng2 - lng1)
        
        a = math.sin(delta_phi/2)**2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def find_nearest_edges(self, lat, lng, num_edges=3, max_distance=500):
        """
        Find nearest SUMO edges to a given coordinate
        
        Args:
            lat, lng: Target coordinates (WGS84)
            num_edges: Number of nearest edges to return
            max_distance: Maximum distance in meters
            
        Returns:
            List of (edge_id, distance) tuples
        """
        distances = []
        
        for edge_id, edge_data in self.edge_coords.items():
            # Convert edge coordinates to lat/lng (use local coordinates directly)
            start_lat, start_lng = project_to_latlon(edge_data['start']['x'], edge_data['start']['y'])
            end_lat, end_lng = project_to_latlon(edge_data['end']['x'], edge_data['end']['y'])
            
            # Calculate distance to edge start and end points
            start_dist = self.calculate_distance(lat, lng, start_lat, start_lng)
            end_dist = self.calculate_distance(lat, lng, end_lat, end_lng)
            
            # Use minimum distance
            min_dist = min(start_dist, end_dist)
            
            if min_dist <= max_distance:
                distances.append((edge_id, min_dist))
        
        # Sort by distance and return top N
        distances.sort(key=lambda x: x[1])
        return distances[:num_edges]
    
    def create_taz_from_places(self, places):
        """
        Create TAZ entries from Google Places
        Match each place to nearest SUMO edges
        """
        print(f"\n{'='*70}")
        print(f"MATCHING PLACES TO SUMO EDGES")
        print(f"{'='*70}\n")
        
        successful_matches = 0
        failed_matches = 0
        
        for idx, place in enumerate(places, 1):
            place_name = place['name']
            place_type = place['type']
            lat = place['lat']
            lng = place['lng']
            
            print(f"[{idx}/{len(places)}] {place_name} ({place_type})")
            print(f"  Location: ({lat}, {lng})")
            
            # Find nearest edges
            nearest_edges = self.find_nearest_edges(lat, lng, num_edges=5, max_distance=500)
            
            if not nearest_edges:
                print(f"  ❌ No nearby edges found (within 500m)")
                failed_matches += 1
                continue
            
            edge_ids = [edge_id for edge_id, dist in nearest_edges]
            distances = [dist for edge_id, dist in nearest_edges]
            
            print(f"  ✅ Matched to {len(edge_ids)} edges:")
            for edge_id, dist in nearest_edges[:3]:
                print(f"     - {edge_id} ({dist:.1f}m)")
            
            # Create TAZ ID (sanitize name)
            taz_id = f"{place_type}_{place_name.lower().replace(' ', '_').replace(',', '')[:30]}"
            taz_id = ''.join(c for c in taz_id if c.isalnum() or c == '_')
            
            # Store TAZ data
            self.taz_list.append({
                'id': taz_id,
                'edges': edge_ids,
                'source_weights': [1.0] * len(edge_ids),  # Equal weights
                'sink_weights': [1.0] * len(edge_ids)
            })
            
            # Store metadata
            self.taz_info[taz_id] = {
                'name': place_name,
                'type': place_type,
                'lat': lat,
                'lng': lng,
                'edges': edge_ids,
                'distances': distances
            }
            
            successful_matches += 1
        
        print(f"\n{'='*70}")
        print(f"Successful matches: {successful_matches}")
        print(f"Failed matches: {failed_matches}")
        print(f"{'='*70}")
    
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
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space='    ')
        tree.write(output_file, encoding='UTF-8', xml_declaration=True)
        
        print(f"✅ Wrote {len(self.taz_list)} TAZ zones to {output_file}")
    
    def write_info_file(self, output_file):
        """Write TAZ metadata to JSON for later use"""
        print(f"Writing TAZ info: {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(self.taz_info, f, indent=2)
        
        print(f"✅ Wrote TAZ metadata for {len(self.taz_info)} zones")
    
    def print_summary(self):
        """Print summary statistics"""
        print(f"\n{'='*70}")
        print(f"TAZ GENERATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total TAZ zones: {len(self.taz_list)}")
        
        # Count by type
        type_counts = {}
        for taz_id, info in self.taz_info.items():
            place_type = info['type']
            type_counts[place_type] = type_counts.get(place_type, 0) + 1
        
        print(f"\nZones by type:")
        for place_type, count in sorted(type_counts.items()):
            print(f"  {place_type:20s}: {count:3d}")
        
        # Sample TAZ
        print(f"\nSample TAZ zones:")
        for i, (taz_id, info) in enumerate(list(self.taz_info.items())[:5]):
            print(f"  {taz_id}")
            print(f"    Name: {info['name']}")
            print(f"    Type: {info['type']}")
            print(f"    Edges: {len(info['edges'])}")
        
        print(f"{'='*70}\n")

def project_to_latlon(local_x, local_y):
    """
    Convert local SUMO coordinates to lat/lon using network bounds
    Based on the original boundary from map.net.xml
    """
    # Network bounds from your data (local coordinates)
    net_min_x, net_max_x = -5.73, 2488.72
    net_min_y, net_max_y = -1.50, 1394.98
    
    # Original geographic bounds from SUMO network file
    # origBoundary="121.054103,14.316008,121.098550,14.363847"
    orig_min_lng, orig_min_lat = 121.054103, 14.316008
    orig_max_lng, orig_max_lat = 121.098550, 14.363847
    
    # Linear interpolation from local to geographic coordinates
    lat = orig_min_lat + (local_y - net_min_y) / (net_max_y - net_min_y) * (orig_max_lat - orig_min_lat)
    lon = orig_min_lng + (local_x - net_min_x) / (net_max_x - net_min_x) * (orig_max_lng - orig_min_lng)
    
    return lat, lon

def calculate_network_bounds(edge_coords):
    """Calculate the bounding box of the network and convert to lat/lon"""
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for edge_data in edge_coords.values():
        for point in ['start', 'end']:
            x, y = edge_data[point]['x'], edge_data[point]['y']
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
    
    # Convert network center to lat/lng
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Convert local coordinates to lat/lon
    center_lat, center_lon = project_to_latlon(center_x, center_y)
    
    # Calculate radius in meters (Google Places API max is 50km = 50000m)
    width = max_x - min_x
    height = max_y - min_y
    radius = min(max(width, height) / 2 + SEARCH_BUFFER, 50000)  # Cap at 50km
    
    return {'lat': center_lat, 'lng': center_lon}, int(radius)

def main():
    print(f"{'='*70}")
    print(f"TAZ GENERATOR FROM GOOGLE PLACES")
    print(f"{'='*70}\n")
    
    # Initialize
    generator = TAZGenerator(GOOGLE_API_KEY)
    
    # Step 1: Load edge coordinates
    generator.load_edge_coordinates(COORD_CACHE_FILE)
    
    # Step 1.5: Calculate search area from network bounds
    area_center, search_radius = calculate_network_bounds(generator.edge_coords)
    print(f"Network center: {area_center}")
    print(f"Search radius: {search_radius}m")
    
    # Step 2: Get places from Google
    places = generator.get_places_from_google(
        location=area_center,
        radius=search_radius,
        place_types=PLACE_TYPES
    )
    
    if not places:
        print("ERROR: No places found. Check your API key and search parameters.")
        sys.exit(1)
    
    # Step 2.5: Rank and filter to top zones
    top_places = generator.rank_and_filter_places(places, MAX_TAZ_ZONES)
    
    # Step 3: Match places to SUMO edges
    generator.create_taz_from_places(top_places)
    
    # Step 4: Write output files
    generator.write_taz_file(OUTPUT_TAZ_FILE)
    generator.write_info_file(OUTPUT_INFO_FILE)
    
    # Step 5: Print summary
    generator.print_summary()
    
    print(f"Done! Next step: Run 'generate_od_traffic.py' to create vehicle demand.")

if __name__ == '__main__':
    main()
