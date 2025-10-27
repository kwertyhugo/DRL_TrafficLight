"""
generate_od_traffic_dummy.py

Dummy version of the traffic generator for testing without Google API calls.
Uses simulated traffic data to test the complete pipeline logic.

Requires:
- traffic.taz.xml (from create_taz_dummy.py)
- taz_info.json (from create_taz_dummy.py)

Usage:
    python generate_od_traffic_dummy.py

Output:
    trips.trips.xml - Raw trip definitions
    demand.rou.xml - Final routed demand with realistic volumes
    od_matrix.json - Origin-Destination matrix for analysis
"""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import subprocess
import random
from datetime import datetime, timedelta
import time

# Configuration
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
TAZ_INFO_FILE = SCRIPT_DIR / 'taz_info.json'
TAZ_FILE = SCRIPT_DIR / 'traffic.taz.xml'
NETWORK_FILE = SCRIPT_DIR / 'map.net.xml'

# Output files
OUTPUT_TRIPS_FILE = SCRIPT_DIR / 'trips.trips.xml'
OUTPUT_DEMAND_FILE = SCRIPT_DIR.parent / 'demand' / 'demand.rou.xml'
OUTPUT_OD_MATRIX_FILE = SCRIPT_DIR / 'od_matrix.json'

# Traffic generation parameters
# Simulation hour range (real-world hours)
SIMULATION_START_TIME = 4  # 4 AM
SIMULATION_END_TIME = 22   # 10 PM
TOTAL_SIMULATION_HOURS = SIMULATION_END_TIME - SIMULATION_START_TIME + 1

# Duration of the compressed simulation in seconds (matches map.sumocfg)
# We'll fit the 4AM-10PM range into 0..SIMULATION_DURATION_SECONDS-1
SIMULATION_DURATION_SECONDS = 10000

# Seconds per (real) hour when compressing the day into the simulation window
SECONDS_PER_HOUR = SIMULATION_DURATION_SECONDS / float(TOTAL_SIMULATION_HOURS)

# By default, sample ALL hours between SIMULATION_START_TIME and SIMULATION_END_TIME
KEY_TRAFFIC_HOURS = list(range(SIMULATION_START_TIME, SIMULATION_END_TIME + 1))

# Distance filtering
MAX_DISTANCE_KM = 10  # Only generate traffic for OD pairs within 10km

class DummyTrafficGenerator:
    def __init__(self):
        self.taz_info = {}
        self.od_matrix = {}
        
    def load_taz_data(self):
        """Load TAZ information from files"""
        print(f"Loading TAZ data...")
        
        # Load TAZ metadata
        if not TAZ_INFO_FILE.exists():
            print(f"ERROR: {TAZ_INFO_FILE} not found. Run create_taz_dummy.py first.")
            sys.exit(1)
        
        with open(TAZ_INFO_FILE, 'r') as f:
            data = json.load(f)
            self.taz_info = data
        
        print(f"Loaded {len(self.taz_info['taz_zones'])} TAZ zones")
        
        # Verify TAZ file exists
        if not TAZ_FILE.exists():
            print(f"ERROR: {TAZ_FILE} not found. Run create_taz_dummy.py first.")
            sys.exit(1)
    
    def calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between two coordinates (Haversine formula)"""
        R = 6371  # Earth radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlng/2) * math.sin(dlng/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def simulate_traffic_conditions(self, origin_id, dest_id, hour):
        """
        Simulate traffic conditions without Google API
        Returns dummy travel time and traffic condition
        """
        # Get origin and destination info
        origin = self.taz_info['taz_zones'][origin_id]
        dest = self.taz_info['taz_zones'][dest_id]
        
        # Calculate straight-line distance
        distance_km = self.calculate_distance(
            origin['lat'], origin['lng'],
            dest['lat'], dest['lng']
        )
        
        # Simulate base travel time (assuming ~30 km/h average in city)
        # Ensure minimum distance to avoid division by zero
        safe_distance_km = max(distance_km, 0.1)  # Minimum 100m
        base_time_hours = safe_distance_km / 30.0
        base_time_minutes = base_time_hours * 60
        
        # Add traffic variation based on hour and random factors
        traffic_multiplier = self.get_traffic_multiplier(hour)
        actual_time_minutes = base_time_minutes * traffic_multiplier
        
        # Determine traffic condition based on delay
        delay_ratio = traffic_multiplier - 1.0
        
        if delay_ratio < 0.2:
            condition = "Normal"
        elif delay_ratio < 0.5:
            condition = "Slow" 
        else:
            condition = "Traffic Jam"
        
        return {
            'distance_km': distance_km,
            'duration_minutes': actual_time_minutes,
            'duration_hours': actual_time_minutes / 60.0,
            'traffic_condition': condition,
            'delay_ratio': delay_ratio
        }
    
    def get_traffic_multiplier(self, hour):
        """Get traffic multiplier based on hour of day"""
        # Peak hours have more traffic
        if hour in [7, 8, 17, 18, 19]:  # Rush hours
            return random.uniform(1.3, 2.2)  # 30-120% slower
        elif hour in [9, 10, 11, 16, 20]:  # Moderate traffic
            return random.uniform(1.1, 1.6)  # 10-60% slower
        elif hour in [12, 13, 14, 15]:  # Lunch/afternoon
            return random.uniform(1.0, 1.4)  # 0-40% slower
        else:  # Off-peak
            return random.uniform(0.9, 1.2)  # Normal to slight delay
    
    def calculate_insertion_rate(self, traffic_data, origin_type, dest_type):
        """
        Calculate realistic insertion rate for dummy testing
        
        For testing purposes, generate reasonable trip numbers:
        - 1-10 trips per hour per OD pair (typical for small zones)
        - Higher rates for popular destinations during peak hours
        """
        distance_km = traffic_data['distance_km']
        duration_hours = traffic_data['duration_hours']
        condition = traffic_data['traffic_condition']
        
        # Base rate: start with small realistic numbers (1-5 trips/hour)
        if condition == "Normal":
            base_rate = random.randint(1, 3)
        elif condition == "Slow":
            base_rate = random.randint(2, 5)  # More trips when slower (people avoid)
        else:  # Traffic Jam
            base_rate = random.randint(1, 2)  # Fewer trips when jammed
        
        # Apply place type attractiveness factors
        origin_factor = self.get_place_attractiveness(origin_type, 'origin')
        dest_factor = self.get_place_attractiveness(dest_type, 'destination')
        
        # Final insertion rate (realistic for testing)
        insertion_rate = int(base_rate * origin_factor * dest_factor)
        
        # Ensure reasonable bounds for testing
        insertion_rate = max(1, min(insertion_rate, 8))  # 1-8 trips per hour max
        
        return insertion_rate
    
    def get_place_attractiveness(self, place_type, role):
        """Get attractiveness factor for different place types"""
        attractiveness = {
            'shopping_mall': {'origin': 0.8, 'destination': 1.5},
            'university': {'origin': 1.2, 'destination': 1.3},
            'hospital': {'origin': 0.6, 'destination': 1.1},
            'school': {'origin': 1.0, 'destination': 1.2},
            'transit_station': {'origin': 1.4, 'destination': 1.4},
            'park': {'origin': 0.5, 'destination': 0.8},
            'point_of_interest': {'origin': 0.7, 'destination': 0.9}
        }
        
        return attractiveness.get(place_type, {}).get(role, 1.0)
    
    def generate_od_matrix(self):
        """Generate Origin-Destination matrix with simulated traffic data"""
        print(f"\n{'='*70}")
        print(f"GENERATING OD MATRIX (DUMMY MODE)")
        print(f"{'='*70}")
        
        taz_zones = self.taz_info['taz_zones']
        zone_ids = list(taz_zones.keys())
        
        total_pairs = len(zone_ids) * (len(zone_ids) - 1)
        print(f"Total possible OD pairs: {total_pairs}")
        
        # Filter by distance
        valid_pairs = []
        for origin_id in zone_ids:
            for dest_id in zone_ids:
                if origin_id == dest_id:
                    continue
                
                origin = taz_zones[origin_id]
                dest = taz_zones[dest_id]
                
                distance = self.calculate_distance(
                    origin['lat'], origin['lng'],
                    dest['lat'], dest['lng']
                )
                
                if distance <= MAX_DISTANCE_KM:
                    valid_pairs.append((origin_id, dest_id, distance))
        
        print(f"Valid OD pairs (≤ {MAX_DISTANCE_KM}km): {len(valid_pairs)}")
        
        # Sample traffic conditions for key hours
        processed_pairs = 0
        
        for origin_id, dest_id, distance in valid_pairs:
            processed_pairs += 1
            origin = taz_zones[origin_id]
            dest = taz_zones[dest_id]
            
            print(f"[{processed_pairs}/{len(valid_pairs)}] "
                  f"{origin['name'][:20]:20} → {dest['name'][:20]:20} "
                  f"({distance:.1f}km)")
            
            # Sample traffic conditions for key hours
            hourly_data = {}
            total_trips = 0
            
            for hour in KEY_TRAFFIC_HOURS:
                # Simulate traffic conditions for the real-world hour
                traffic_data = self.simulate_traffic_conditions(origin_id, dest_id, hour)

                # Calculate insertion rate using the testing formula
                insertion_rate = self.calculate_insertion_rate(
                    traffic_data, origin['type'], dest['type']
                )

                hourly_data[hour] = {
                    'distance_km': traffic_data['distance_km'],
                    'duration_minutes': traffic_data['duration_minutes'],
                    'traffic_condition': traffic_data['traffic_condition'],
                    'insertion_rate': insertion_rate
                }

                total_trips += insertion_rate
            
            # Store in OD matrix
            if origin_id not in self.od_matrix:
                self.od_matrix[origin_id] = {}
            
            self.od_matrix[origin_id][dest_id] = {
                'origin_name': origin['name'],
                'dest_name': dest['name'],
                'distance_km': distance,
                'hourly_data': hourly_data,
                'total_trips': total_trips
            }
            
            print(f"  Trips: {total_trips}, Avg condition: {self.get_avg_condition(hourly_data)}")
        
        print(f"\nCompleted OD matrix generation!")
        print(f"Processed: {processed_pairs} OD pairs")
    
    def get_avg_condition(self, hourly_data):
        """Get average traffic condition description"""
        conditions = [data['traffic_condition'] for data in hourly_data.values()]
        if conditions.count('Traffic Jam') > len(conditions) / 2:
            return "Congested"
        elif conditions.count('Slow') > len(conditions) / 2:
            return "Moderate"
        else:
            return "Normal"
    
    def generate_trips_file(self):
        """Generate SUMO trips file"""
        print(f"\nGenerating trips file: {OUTPUT_TRIPS_FILE}")
        
        root = ET.Element('trips')
        # Remove schema validation to avoid "can't read local schema" warnings
        # SUMO will still process the file correctly without schema validation
        
        trip_id = 0
        total_trips = 0
        
        for origin_id, destinations in self.od_matrix.items():
            for dest_id, od_data in destinations.items():
                # Distribute trips across hours
                for hour, hour_data in od_data['hourly_data'].items():
                    num_trips = hour_data['insertion_rate']
                    
                    # Create trips for this hour
                    for i in range(num_trips):
                        trip_id += 1
                        # Map the real-world hour to a simulation time window in 0..SIMULATION_DURATION_SECONDS-1
                        # Compute which hour index this is (0 = SIMULATION_START_TIME)
                        simulation_hour_index = hour - SIMULATION_START_TIME
                        # Start and end seconds for this hour's window
                        start_sec = int(simulation_hour_index * SECONDS_PER_HOUR)
                        end_sec = int(min((simulation_hour_index + 1) * SECONDS_PER_HOUR - 1, SIMULATION_DURATION_SECONDS - 1))
                        # Pick a random second inside the compressed hour window
                        if start_sec >= end_sec:
                            depart_time = min(max(0, start_sec), SIMULATION_DURATION_SECONDS - 1)
                        else:
                            depart_time = random.randint(start_sec, end_sec)
                        
                        trip = ET.SubElement(root, 'trip')
                        trip.set('id', f'trip_{trip_id}')
                        trip.set('from', random.choice(self.taz_info['taz_zones'][origin_id]['edges']))
                        trip.set('to', random.choice(self.taz_info['taz_zones'][dest_id]['edges']))
                        trip.set('depart', str(depart_time))
                        trip.set('fromTaz', origin_id)
                        trip.set('toTaz', dest_id)
                        
                        total_trips += 1
        
        # Write trips file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(OUTPUT_TRIPS_FILE, encoding='utf-8', xml_declaration=True)
        
        print(f"✅ Generated {total_trips:,} trips")
        return total_trips
    
    def run_duarouter(self):
        """Run SUMO duarouter to generate routes"""
        print(f"\nRunning duarouter...")
        
        # Ensure output directory exists
        OUTPUT_DEMAND_FILE.parent.mkdir(exist_ok=True)
        
        # duarouter command
        cmd = [
            'duarouter',
            '--net-file', str(NETWORK_FILE),
            '--trip-files', str(OUTPUT_TRIPS_FILE),
            '--output-file', str(OUTPUT_DEMAND_FILE),
            '--ignore-errors', 'true',
            '--repair', 'true'
        ]
        
        try:
            print(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
            
            if result.returncode == 0:
                print(f"✅ duarouter completed successfully")
                print(f"Generated: {OUTPUT_DEMAND_FILE}")
                return True
            else:
                print(f"❌ duarouter failed:")
                print(f"Error: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print(f"❌ duarouter not found. Make sure SUMO is installed and in PATH.")
            return False
    
    def save_od_matrix(self):
        """Save OD matrix to JSON file"""
        print(f"\nSaving OD matrix: {OUTPUT_OD_MATRIX_FILE}")
        
        # Add summary statistics
        summary_data = {
            'generated_at': datetime.now().isoformat(),
            'dummy_mode': True,
            'total_od_pairs': len([(o, d) for o in self.od_matrix for d in self.od_matrix[o]]),
            'total_zones': len(self.taz_info['taz_zones']),
            'hours_sampled': KEY_TRAFFIC_HOURS,
            'distance_filter_km': MAX_DISTANCE_KM,
            'od_matrix': self.od_matrix
        }
        
        with open(OUTPUT_OD_MATRIX_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ OD matrix saved")
    
    def print_summary(self):
        """Print generation summary"""
        print(f"\n{'='*70}")
        print(f"DUMMY TRAFFIC GENERATION SUMMARY")
        print(f"{'='*70}")
        
        total_pairs = len([(o, d) for o in self.od_matrix for d in self.od_matrix[o]])
        total_trips = sum(od_data['total_trips'] 
                         for destinations in self.od_matrix.values()
                         for od_data in destinations.values())
        
        print(f"TAZ zones: {len(self.taz_info['taz_zones'])}")
        print(f"OD pairs processed: {total_pairs}")
        print(f"Total trips generated: {total_trips:,}")
        print(f"Hours sampled: {len(KEY_TRAFFIC_HOURS)} ({KEY_TRAFFIC_HOURS})")
        print(f"Distance filter: ≤ {MAX_DISTANCE_KM}km")
        
        print(f"\nFiles generated:")
        print(f"  - {OUTPUT_TRIPS_FILE} (raw trips)")
        print(f"  - {OUTPUT_DEMAND_FILE} (routed demand)")
        print(f"  - {OUTPUT_OD_MATRIX_FILE} (analysis data)")
        
        print(f"\n🎯 Dummy traffic generation completed!")
        print(f"   No API calls needed - pure simulation")
        print(f"   Ready for SUMO simulation testing")

def main():
    print(f"{'='*70}")
    print(f"DUMMY TRAFFIC GENERATOR (NO API CALLS)")
    print(f"{'='*70}\n")
    
    # Initialize
    generator = DummyTrafficGenerator()
    
    # Step 1: Load TAZ data
    generator.load_taz_data()
    
    # Step 2: Generate OD matrix with simulated data
    generator.generate_od_matrix()
    
    # Step 3: Generate trips file
    total_trips = generator.generate_trips_file()
    
    # Step 4: Run duarouter
    router_success = generator.run_duarouter()
    
    # Step 5: Save OD matrix
    generator.save_od_matrix()
    
    # Step 6: Print summary
    generator.print_summary()
    
    if router_success:
        print(f"\nSuccess! You can now test SUMO with:")
        print(f"sumo-gui -c map.sumocfg --route-files demand/demand.rou.xml")
    else:
        print(f"\nGenerated trips file, but duarouter failed.")
        print(f"Check SUMO installation and network file.")

if __name__ == '__main__':
    main()