"""
parse_network.py

Extract edge coordinates from SUMO network file and cache them.
This only needs to be run ONCE per network.

Usage:
    python parse_network.py

Output:
    edge_coordinates.json - Cached coordinates for all edges
"""

import xml.etree.ElementTree as ET
import json
from pathlib import Path
import sys

# Configuration
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
NETWORK_FILE = SCRIPT_DIR / 'map.net.xml'
OUTPUT_FILE = SCRIPT_DIR / 'edge_coordinates.json'

class NetworkParser:
    def __init__(self):
        self.edge_coords = {}
        self.stats = {
            'total_edges': 0,
            'internal_edges_skipped': 0,
            'valid_edges': 0,
            'edges_without_shape': 0
        }
    
    def parse_network(self, net_file):
        """
        Extract coordinates from SUMO network file.
        
        For each edge:
        - Extract start/end coordinates from lane shape
        - Calculate edge length
        - Store in dictionary for fast lookup
        """
        print(f"{'='*70}")
        print(f"SUMO Network Parser")
        print(f"{'='*70}")
        print(f"Parsing: {net_file}")
        
        if not Path(net_file).exists():
            print(f"ERROR: Network file not found: {net_file}")
            print(f"Please ensure '{net_file}' is in the current directory.")
            sys.exit(1)
        
        print(f"Loading XML... (this may take a moment for large networks)")
        
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"ERROR: Failed to parse XML: {e}")
            sys.exit(1)
        
        print(f"Extracting edge coordinates...")
        
        # Find all edges
        for edge in root.findall('.//edge'):
            self.stats['total_edges'] += 1
            edge_id = edge.get('id')
            
            # Skip internal edges (intersection connectors start with ':')
            if edge_id and edge_id.startswith(':'):
                self.stats['internal_edges_skipped'] += 1
                continue
            
            # Get first lane's shape (all lanes follow same route)
            lane = edge.find('lane')
            if lane is None:
                self.stats['edges_without_shape'] += 1
                continue
            
            shape = lane.get('shape')
            if not shape:
                self.stats['edges_without_shape'] += 1
                continue
            
            # Parse shape: "x1,y1 x2,y2 x3,y3 ..."
            coords = shape.strip().split()
            if len(coords) < 2:
                self.stats['edges_without_shape'] += 1
                continue
            
            # Extract start and end points
            start = coords[0].split(',')
            end = coords[-1].split(',')
            
            # Calculate edge length from polyline
            length = self._calculate_length(coords)
            
            # Store edge data
            self.edge_coords[edge_id] = {
                'start': {
                    'x': float(start[0]),
                    'y': float(start[1])
                },
                'end': {
                    'x': float(end[0]),
                    'y': float(end[1])
                },
                'length_m': length,  # in meters
                'length_km': length / 1000,  # in kilometers
                'num_points': len(coords)
            }
            
            self.stats['valid_edges'] += 1
            
            # Progress indicator
            if self.stats['valid_edges'] % 1000 == 0:
                print(f"  Processed {self.stats['valid_edges']} edges...")
        
        print(f"Extraction complete!")
        return self.edge_coords
    
    def _calculate_length(self, coords):
        """
        Calculate edge length by summing distances between consecutive points.
        This gives accurate road length even for curved roads.
        
        Args:
            coords: List of "x,y" coordinate strings
            
        Returns:
            Length in meters
        """
        total_length = 0.0
        
        for i in range(len(coords) - 1):
            # Parse consecutive points
            p1 = [float(x) for x in coords[i].split(',')]
            p2 = [float(x) for x in coords[i + 1].split(',')]
            
            # Euclidean distance
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            distance = (dx**2 + dy**2)**0.5
            
            total_length += distance
        
        return total_length
    
    def save_to_file(self, output_file):
        """Save extracted coordinates to JSON file"""
        print(f"\nSaving to: {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(self.edge_coords, f, indent=2)
        
        file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
    
    def print_statistics(self):
        """Print parsing statistics"""
        print(f"\n{'='*70}")
        print(f"PARSING STATISTICS")
        print(f"{'='*70}")
        print(f"Total edges in network:        {self.stats['total_edges']:,}")
        print(f"Internal edges (skipped):      {self.stats['internal_edges_skipped']:,}")
        print(f"Edges without shape (skipped): {self.stats['edges_without_shape']:,}")
        print(f"Valid edges extracted:         {self.stats['valid_edges']:,}")
        print(f"{'='*70}")
        
        if self.stats['valid_edges'] > 0:
            # Calculate statistics
            lengths = [e['length_km'] for e in self.edge_coords.values()]
            avg_length = sum(lengths) / len(lengths)
            min_length = min(lengths)
            max_length = max(lengths)
            
            print(f"\nEDGE LENGTH STATISTICS")
            print(f"{'='*70}")
            print(f"Average edge length: {avg_length:.3f} km")
            print(f"Shortest edge:       {min_length:.3f} km")
            print(f"Longest edge:        {max_length:.3f} km")
            print(f"{'='*70}")
    
    def print_sample_edges(self, n=5):
        """Print sample edges for verification"""
        print(f"\nSAMPLE EDGES (first {n}):")
        print(f"{'='*70}")
        
        for i, (edge_id, data) in enumerate(list(self.edge_coords.items())[:n]):
            print(f"\nEdge: {edge_id}")
            print(f"  Start: ({data['start']['x']:.2f}, {data['start']['y']:.2f})")
            print(f"  End:   ({data['end']['x']:.2f}, {data['end']['y']:.2f})")
            print(f"  Length: {data['length_km']:.3f} km ({data['num_points']} points)")

def main():
    parser = NetworkParser()
    
    # Parse network file
    edge_coords = parser.parse_network(NETWORK_FILE)
    
    # Save to JSON
    parser.save_to_file(OUTPUT_FILE)
    
    # Print statistics
    parser.print_statistics()
    parser.print_sample_edges(n=5)
    
    print(f"\n{'='*70}")
    print(f"SUCCESS! Coordinates cached to '{OUTPUT_FILE}'")
    print(f"You can now run 'generate_traffic.py' to create vehicle demand.")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
