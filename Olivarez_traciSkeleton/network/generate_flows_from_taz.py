"""
generate_flows_from_taz.py

Generate SUMO flow definitions from OD matrix instead of individual trips.
Flows are much more memory-efficient and realistic for large-scale simulations.

Flows define traffic rates (vehicles per hour) rather than individual trips,
allowing SUMO to generate vehicles dynamically during simulation.

Usage:
    python generate_flows_from_taz.py

Output:
    flows.flows.xml - Flow definitions for SUMO
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import random

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
OD_MATRIX_FILE = SCRIPT_DIR / 'od_matrix.json'
TAZ_FILE = SCRIPT_DIR / 'traffic.taz.xml'
NETWORK_FILE = SCRIPT_DIR / 'map.net.xml'

# Output file
OUTPUT_FLOWS_FILE = SCRIPT_DIR.parent / 'demand' / 'flows.flows.xml'

# Traffic generation parameters
SIMULATION_START_TIME = 4  # 4 AM (real time)
SIMULATION_END_TIME = 22   # 10 PM (real time)
REAL_DURATION_HOURS = SIMULATION_END_TIME - SIMULATION_START_TIME  # 18 hours
SIMULATION_DURATION = 18000  # Map 18 hours to 18,000 simulation steps
TIME_SCALE_FACTOR = SIMULATION_DURATION / (REAL_DURATION_HOURS * 3600)  # steps per real second

class FlowGenerator:
    def __init__(self):
        self.od_matrix = {}
        self.taz_info = {}
        self.flows = []
        
        # Flow scaling - your OD matrix rates are already realistic for Manila traffic!
        self.flow_scaling_factor = 1  # NO SCALING - use Google Maps data as-is
        
    def analyze_rates_and_suggest_scaling(self):
        """Analyze OD matrix rates and suggest appropriate scaling factor"""
        print(f"\nANALYZING RATES TO DETERMINE SCALING FACTOR")
        print("=" * 60)
        
        all_rates = []
        for origin_id, destinations in self.od_matrix.items():
            for dest_id, od_data in destinations.items():
                avg_rate = od_data.get('avg_insertion_rate_per_hour', 0)
                if avg_rate > 0:
                    all_rates.append(avg_rate)
        
        if not all_rates:
            print("ERROR: No rates found in OD matrix")
            return
        
        min_rate = min(all_rates)
        max_rate = max(all_rates)
        avg_rate = sum(all_rates) / len(all_rates)
        median_rate = sorted(all_rates)[len(all_rates)//2]
        
        print(f"Rate statistics:")
        print(f"  Minimum: {min_rate:,.1f} veh/h")
        print(f"  Maximum: {max_rate:,.1f} veh/h")
        print(f"  Average: {avg_rate:,.1f} veh/h")
        print(f"  Median: {median_rate:,.1f} veh/h")
        
        # Suggest scaling factor to get median rate around 1-10 veh/h
        target_median = 5.0  # Target 5 vehicles per hour
        suggested_scaling = int(median_rate / target_median)
        
        print(f"\nSuggested scaling factor: {suggested_scaling:,}")
        print(f"   This would make median rate: {median_rate/suggested_scaling:.1f} veh/h")
        print(f"   Range after scaling: {min_rate/suggested_scaling:.2f} - {max_rate/suggested_scaling:.1f} veh/h")
        
        # Update scaling factor if suggestion is reasonable
        if 10 <= suggested_scaling <= 100000:
            old_factor = self.flow_scaling_factor
            self.flow_scaling_factor = suggested_scaling
            print(f"Updated scaling factor from {old_factor:,} to {suggested_scaling:,}")
        else:
            print(f"WARNING: Keeping current scaling factor: {self.flow_scaling_factor:,}")

    def load_od_matrix(self):
        """Load existing OD matrix from previous run"""
        print(f"Loading OD matrix from: {OD_MATRIX_FILE}")
        
        if not OD_MATRIX_FILE.exists():
            print(f"ERROR: {OD_MATRIX_FILE} not found.")
            print("Run generate_od_traffic.py first to create the OD matrix.")
            sys.exit(1)
            
        with open(OD_MATRIX_FILE, 'r') as f:
            self.od_matrix = json.load(f)
            
        print(f"Loaded OD matrix with {sum(len(dests) for dests in self.od_matrix.values())} OD pairs")
        
        # Load TAZ info
        taz_info_file = SCRIPT_DIR / 'taz_info.json'
        if taz_info_file.exists():
            with open(taz_info_file, 'r') as f:
                data = json.load(f)
                if 'taz_zones' in data:
                    self.taz_info = data['taz_zones']
                else:
                    self.taz_info = data
            print(f"Loaded TAZ info for {len(self.taz_info)} zones")
        
        # Analyze rates and suggest better scaling
        self.analyze_rates_and_suggest_scaling()
    
    def generate_flows(self):
        """Generate flow definitions from OD matrix"""
        print(f"\n{'='*70}")
        print(f"GENERATING FLOWS FROM OD MATRIX")
        print(f"{'='*70}")
        
        flow_id = 0
        total_flows = 0
        processed_pairs = 0
        skipped_too_low = 0
        total_pairs = sum(len(destinations) for destinations in self.od_matrix.values())
        
        print(f"Processing {total_pairs} OD pairs...")
        print(f"Scaling factor: {self.flow_scaling_factor:,}x reduction")
        print(f"Minimum flow rate: 0.1 vehicles/hour")
        
        for origin_id, destinations in self.od_matrix.items():
            for dest_id, od_data in destinations.items():
                processed_pairs += 1
                
                # Show progress
                if processed_pairs % 100 == 0:
                    print(f"  Progress: {processed_pairs}/{total_pairs} ({processed_pairs/total_pairs*100:.1f}%)")
                
                # Get average rate and scale it down
                avg_rate = od_data.get('avg_insertion_rate_per_hour', 0)
                scaled_rate = avg_rate / self.flow_scaling_factor
                
                # Debug: Show sample rates for first few pairs
                if processed_pairs <= 5:
                    origin_name = od_data.get('origin_name', origin_id)
                    dest_name = od_data.get('dest_name', dest_id)
                    print(f"  Sample: {origin_name[:30]} -> {dest_name[:30]}")
                    print(f"    Original rate: {avg_rate:,.1f} veh/h -> Scaled: {scaled_rate:.2f} veh/h")
                
                # Skip very low flows (less than 0.1 vehicles per hour)
                if scaled_rate < 0.1:
                    skipped_too_low += 1
                    continue
                
                # Create time-varying flows based on time_based_rates if available
                if 'time_based_rates' in od_data:
                    time_rates = od_data['time_based_rates']
                    
                    # Group consecutive hours with similar rates into flow periods
                    flow_periods = self.create_flow_periods(time_rates)
                    
                    for period in flow_periods:
                        flow_id += 1
                        
                        # Scale down the rate
                        scaled_period_rate = period['rate'] / self.flow_scaling_factor
                        
                        if scaled_period_rate >= 0.1:  # Only create flows with meaningful rates
                            # Calculate actual speed from Google Maps data
                            route_length = od_data.get('route_length_km', 1.0)
                            duration_hours = period.get('duration_hours', 1.0)
                            speed_kmh = route_length / duration_hours if duration_hours > 0 else 50
                            
                            # Ensure reasonable speed limits (Google Maps might have very small durations)
                            if speed_kmh > 120:  # Too fast, probably calculation error
                                speed_kmh = 60  # Default to 60 km/h
                            elif speed_kmh < 5:  # Too slow, probably calculation error
                                speed_kmh = 30   # Default to 30 km/h
                            
                            speed_ms = speed_kmh / 3.6  # Convert km/h to m/s for SUMO
                            
                            # Use traffic condition to modify speed
                            condition = period.get('condition', 'normal')
                            if condition == 'slow':
                                speed_ms *= 0.7  # 30% slower
                            elif condition == 'traffic_jam':
                                speed_ms *= 0.4  # 60% slower
                            
                            # Ensure minimum reasonable speed for SUMO
                            speed_ms = max(speed_ms, 5.0)  # At least 5 m/s (18 km/h)
                            
                            flow = {
                                'id': f'flow_{flow_id}',
                                'from_taz': origin_id,
                                'to_taz': dest_id,
                                'begin': period['begin'],
                                'end': period['end'],
                                'vehsPerHour': round(scaled_period_rate, 2),
                                'departLane': 'random',
                                'departSpeed': round(speed_ms, 1),  # ACTUAL Google Maps speed!
                                'route_length_km': route_length,
                                'google_speed_kmh': round(speed_kmh, 1),
                                'condition': condition
                            }
                            
                            self.flows.append(flow)
                            total_flows += 1
                
                else:
                    # Fallback: create a single flow for the entire simulation period
                    if scaled_rate >= 0.1:
                        flow_id += 1
                        
                        # Calculate speed from Google Maps data
                        route_length = od_data.get('route_length_km', 1.0)
                        # Use average duration from time-based rates if available
                        avg_duration = sum(tr.get('duration_hours', 1.0) for tr in od_data.get('time_based_rates', {}).values()) / max(len(od_data.get('time_based_rates', {})), 1)
                        speed_kmh = route_length / avg_duration if avg_duration > 0 else 50
                        
                        # Apply same speed limits as above
                        if speed_kmh > 120:
                            speed_kmh = 60
                        elif speed_kmh < 5:
                            speed_kmh = 30
                            
                        speed_ms = max(speed_kmh / 3.6, 5.0)  # At least 5 m/s
                        
                        flow = {
                            'id': f'flow_{flow_id}',
                            'from_taz': origin_id,
                            'to_taz': dest_id,
                            'begin': 0,
                            'end': SIMULATION_DURATION,
                            'vehsPerHour': round(scaled_rate, 2),
                            'departLane': 'random',
                            'departSpeed': round(speed_ms, 1),  # ACTUAL Google Maps speed!
                            'route_length_km': route_length,
                            'google_speed_kmh': round(speed_kmh, 1)
                        }
                        
                        self.flows.append(flow)
                        total_flows += 1
        
        print(f"Generated {total_flows} flows from {total_pairs} OD pairs")
        print(f"   Skipped {skipped_too_low} pairs with rates below 0.1 veh/h")
        
        if total_flows > 0:
            print(f"   Flows cover {len(set(f['from_taz'] for f in self.flows))} origin TAZ zones")
            print(f"   Flows cover {len(set(f['to_taz'] for f in self.flows))} destination TAZ zones")
        else:
            print(f"   ⚠️  No flows generated - all rates were below minimum threshold")
            print(f"   Try reducing scaling factor from {self.flow_scaling_factor} to {self.flow_scaling_factor//10}")
        
        return total_flows
    
    def create_flow_periods(self, time_rates):
        """Group consecutive hours with similar rates into flow periods"""
        periods = []
        current_period = None
        
        # Sort time keys
        sorted_times = sorted(time_rates.keys())
        
        for time_key in sorted_times:
            hour = int(time_key.split(':')[0])
            rate = time_rates[time_key].get('insertion_rate', 0)
            
            # Convert hour to simulation steps (scaled time)
            hour_offset = hour - SIMULATION_START_TIME
            step_begin = int(hour_offset * (SIMULATION_DURATION / REAL_DURATION_HOURS))
            step_end = int((hour_offset + 1) * (SIMULATION_DURATION / REAL_DURATION_HOURS))
            
            if current_period is None:
                # Start new period
                current_period = {
                    'begin': step_begin,
                    'end': step_end,
                    'rate': rate
                }
            else:
                # Check if rate is similar (within 20%)
                rate_diff = abs(rate - current_period['rate']) / max(current_period['rate'], 1)
                
                if rate_diff < 0.2 and step_begin == current_period['end']:
                    # Extend current period
                    current_period['end'] = step_end
                    current_period['rate'] = (current_period['rate'] + rate) / 2  # Average rate
                else:
                    # End current period and start new one
                    periods.append(current_period)
                    current_period = {
                        'begin': step_begin,
                        'end': step_end,
                        'rate': rate
                    }
        
        # Add the last period
        if current_period:
            periods.append(current_period)
        
        return periods
    
    def write_flows_file(self):
        """Write flow definitions to XML file (always sorted by departure time)"""
        print(f"\nWriting flows to: {OUTPUT_FLOWS_FILE}")
        
        # Ensure output directory exists
        OUTPUT_FLOWS_FILE.parent.mkdir(exist_ok=True)
        
        try:
            # Sort flows by begin time before writing - THIS ENSURES NO SORTING WARNINGS
            print("Sorting flows by departure time...")
            sorted_flows = sorted(self.flows, key=lambda x: float(x['begin']))
            print(f"Flows sorted: {len(sorted_flows)} flows from time {sorted_flows[0]['begin']} to {sorted_flows[-1]['begin']}")
            
            with open(OUTPUT_FLOWS_FILE, 'w', encoding='UTF-8') as f:
                # Write XML header
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ')
                f.write('xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n')
                
                # Write flows in sorted order
                for flow in sorted_flows:
                    f.write(f'    <flow id="{flow["id"]}" ')
                    f.write(f'fromTaz="{flow["from_taz"]}" ')
                    f.write(f'toTaz="{flow["to_taz"]}" ')
                    f.write(f'begin="{flow["begin"]}" ')
                    f.write(f'end="{flow["end"]}" ')
                    f.write(f'vehsPerHour="{flow["vehsPerHour"]}" ')
                    f.write(f'departLane="{flow["departLane"]}" ')
                    f.write(f'departSpeed="{flow["departSpeed"]}" />\n')  # Using ACTUAL Google Maps speed!
                
                # Close XML
                f.write('\n</routes>\n')
            
            print(f"SUCCESS: Wrote {len(sorted_flows)} flows to {OUTPUT_FLOWS_FILE} (sorted by departure time)")
            
        except Exception as e:
            print(f"ERROR: Error writing flows file: {e}")
            sys.exit(1)
    
    def print_summary(self):
        """Print generation summary and statistics"""
        print(f"\n{'='*70}")
        print(f"FLOW GENERATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"Total flows generated: {len(self.flows)}")
        print(f"Real time period: {SIMULATION_START_TIME}:00 - {SIMULATION_END_TIME}:00 ({REAL_DURATION_HOURS} hours)")
        print(f"Simulation steps: 0 - {SIMULATION_DURATION} (18,000 steps)")
        print(f"Time scaling: {TIME_SCALE_FACTOR:.2f} steps per real second")
        
        if self.flows:
            # Flow statistics
            total_rate = sum(flow['vehsPerHour'] for flow in self.flows)
            avg_rate = total_rate / len(self.flows)
            max_rate = max(flow['vehsPerHour'] for flow in self.flows)
            min_rate = min(flow['vehsPerHour'] for flow in self.flows)
            
            print(f"\nFlow rate statistics:")
            print(f"  Total vehicles per hour: {total_rate:,.1f}")
            print(f"  Average flow rate: {avg_rate:.2f} veh/hour")
            print(f"  Maximum flow rate: {max_rate:.2f} veh/hour") 
            print(f"  Minimum flow rate: {min_rate:.2f} veh/hour")
            
            # Estimate total vehicles over simulation
            estimated_vehicles = total_rate * (SIMULATION_DURATION / 3600)
            print(f"  Estimated total vehicles: {estimated_vehicles:,.0f}")
            
            # Coverage statistics
            origin_count = len(set(flow['from_taz'] for flow in self.flows))
            dest_count = len(set(flow['to_taz'] for flow in self.flows))
            print(f"\nCoverage:")
            print(f"  Origin TAZ zones: {origin_count}")
            print(f"  Destination TAZ zones: {dest_count}")
        
        print(f"\nFlows are much more efficient than individual trips!")
        print(f"   SUMO will generate vehicles dynamically during simulation")
        print(f"{'='*70}")

def main():
    print(f"{'='*70}")
    print(f"FLOW GENERATOR (FROM OD MATRIX)")
    print(f"{'='*70}\n")
    
    # Initialize
    generator = FlowGenerator()
    
    # Step 1: Load existing OD matrix
    generator.load_od_matrix()
    
    # Step 2: Generate flows
    num_flows = generator.generate_flows()
    
    if num_flows == 0:
        print("ERROR: No flows generated. Check OD matrix data or scaling factor.")
        sys.exit(1)
    
    # Step 3: Write flows file
    generator.write_flows_file()
    
    # Step 4: Print summary
    generator.print_summary()
    
    print(f"\nSUCCESS: Done! Generated flows from OD matrix.")
    print(f"\nOutput file:")
    print(f"   - Flows file: {OUTPUT_FLOWS_FILE}")
    print(f"\n🚗 Next: Use flows.flows.xml directly in your SUMO simulation!")
    print(f"   No need for duarouter - flows work directly with SUMO!")

if __name__ == '__main__':
    main()