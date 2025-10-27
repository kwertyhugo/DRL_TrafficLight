"""
validate_edges.py

Cross-check edges between demand.rou.xml and edge_coordinates.json
to ensure all route edges have coordinate data available.
"""

import xml.etree.ElementTree as ET
import json
from pathlib import Path
import re

def extract_edges_from_demand():
    """Extract all unique edges from demand.rou.xml routes"""
    demand_file = Path('../demand/demand.rou.xml')
    
    if not demand_file.exists():
        print(f"ERROR: {demand_file} not found")
        return set()
    
    print(f"Reading routes from: {demand_file}")
    
    tree = ET.parse(demand_file)
    root = tree.getroot()
    
    all_edges = set()
    route_count = 0
    
    for vehicle in root.findall('vehicle'):
        route = vehicle.find('route')
        if route is not None:
            edges_attr = route.get('edges', '')
            if edges_attr:
                edges = edges_attr.split()
                all_edges.update(edges)
                route_count += 1
    
    print(f"Found {route_count} routes with {len(all_edges)} unique edges")
    return all_edges

def load_edge_coordinates():
    """Load edge coordinates from JSON file"""
    coord_file = Path('edge_coordinates.json')
    
    if not coord_file.exists():
        print(f"ERROR: {coord_file} not found")
        return {}
    
    print(f"Loading coordinates from: {coord_file}")
    
    with open(coord_file, 'r') as f:
        coords = json.load(f)
    
    print(f"Loaded coordinates for {len(coords)} edges")
    return coords

def validate_edge_coverage():
    """Check if all demand edges have coordinates"""
    print("="*70)
    print("EDGE VALIDATION REPORT")
    print("="*70)
    
    # Get edges from both sources
    demand_edges = extract_edges_from_demand()
    coord_edges = set(load_edge_coordinates().keys())
    
    if not demand_edges or not coord_edges:
        print("ERROR: Could not load data from one or both files")
        return
    
    # Find missing edges
    missing_edges = demand_edges - coord_edges
    extra_edges = coord_edges - demand_edges
    common_edges = demand_edges & coord_edges
    
    print(f"\nSUMMARY:")
    print(f"- Edges in demand routes: {len(demand_edges)}")
    print(f"- Edges with coordinates: {len(coord_edges)}")
    print(f"- Common edges: {len(common_edges)}")
    print(f"- Missing coordinates: {len(missing_edges)}")
    print(f"- Extra coordinates: {len(extra_edges)}")
    
    # Coverage percentage
    if demand_edges:
        coverage = (len(common_edges) / len(demand_edges)) * 100
        print(f"- Coverage: {coverage:.1f}%")
    
    # Show missing edges (if any)
    if missing_edges:
        print(f"\nMISSING EDGES (no coordinates):")
        missing_list = sorted(list(missing_edges))
        for i, edge in enumerate(missing_list[:20]):  # Show first 20
            print(f"  {i+1:2d}. {edge}")
        if len(missing_list) > 20:
            print(f"  ... and {len(missing_list) - 20} more")
    
    # Show sample of extra edges
    if extra_edges:
        print(f"\nEXTRA EDGES (coordinates but not in routes):")
        extra_list = sorted(list(extra_edges))
        for i, edge in enumerate(extra_list[:10]):  # Show first 10
            print(f"  {i+1:2d}. {edge}")
        if len(extra_list) > 10:
            print(f"  ... and {len(extra_list) - 10} more")
    
    # Pattern analysis
    print(f"\nEDGE PATTERN ANALYSIS:")
    analyze_edge_patterns(demand_edges, coord_edges)
    
    return missing_edges, extra_edges, common_edges

def analyze_edge_patterns(demand_edges, coord_edges):
    """Analyze patterns in edge naming"""
    
    # Count positive vs negative edges
    demand_pos = len([e for e in demand_edges if not e.startswith('-')])
    demand_neg = len([e for e in demand_edges if e.startswith('-')])
    
    coord_pos = len([e for e in coord_edges if not e.startswith('-')])
    coord_neg = len([e for e in coord_edges if e.startswith('-')])
    
    print(f"  Demand edges: {demand_pos} positive, {demand_neg} negative")
    print(f"  Coord edges:  {coord_pos} positive, {coord_neg} negative")
    
    # Check for patterns in missing edges
    missing_edges = demand_edges - coord_edges
    if missing_edges:
        missing_patterns = {}
        for edge in missing_edges:
            if edge.startswith('-'):
                base_edge = edge[1:]  # Remove minus sign
                if base_edge in coord_edges:
                    missing_patterns.setdefault('negative_of_existing', []).append(edge)
                else:
                    missing_patterns.setdefault('negative_missing_base', []).append(edge)
            else:
                neg_edge = '-' + edge
                if neg_edge in coord_edges:
                    missing_patterns.setdefault('positive_of_existing', []).append(edge)
                else:
                    missing_patterns.setdefault('completely_missing', []).append(edge)
        
        print(f"  Missing edge patterns:")
        for pattern, edges in missing_patterns.items():
            print(f"    {pattern}: {len(edges)} edges")

if __name__ == "__main__":
    validate_edge_coverage()