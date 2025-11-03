"""
generate_od_traffic.py

Generate realistic Origin-Destination traffic demand by:
1. Loading TAZ data from create_taz.py
2. Querying Google Maps for travel times between OD pairs
3. Calculating insertion rates based on travel times and place attractiveness
4. Generating trips with duarouter
5. Creating demand.rou.xml with realistic volumes

Requires:
- traffic.taz.xml (from create_taz.py)
- taz_info.json (from create_taz.py)
- Google Maps API key with Directions API enabled

Usage:
    python generate_od_traffic.py

Output:
    trips.trips.xml - Raw trip definitions
    demand.rou.xml - Final routed demand with realistic volumes
    od_matrix.json - Origin-Destination matrix for analysis
"""

import googlemaps
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
GOOGLE_API_KEY = 'AIzaSyDU6v_Cc1mJglRT7C-rTv6vU3JZhiBH0oM'

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
SIMULATION_START_TIME = 4  # 4 AM
SIMULATION_END_TIME = 22   # 10 PM
SIMULATION_DURATION = (SIMULATION_END_TIME - SIMULATION_START_TIME) * 3600  # 18 hours in seconds

# Sample only key hours to reduce API calls (comment out to sample all hours)
MAX_DISTANCE_KM = 10  # Maximum distance between OD pairs to consider

# Traffic demand based on Google Maps traffic conditions (vehicles/hour)
TRAFFIC_DEMAND = {
    'normal': 50,       # veh/hr
    'slow': 120,        # veh/hr
    'traffic_jam': 200  # veh/hr
}

# Road capacity estimates by road type (vehicles/km)
ROAD_CAPACITY = {
    'highway': 2000,
    'arterial': 1200,
    'collector': 800,
    'local': 400,
    'default': 600  # Default for unknown road types
}

# Time intervals for traffic condition sampling (every hour)
TRAFFIC_SAMPLE_INTERVAL = 3600  # 1 hour in seconds

class ODTrafficGenerator:
    def __init__(self, api_key):
        self.gmaps = googlemaps.Client(key=api_key) if api_key != 'YOUR_API_KEY_HERE' else None
        self.taz_info = {}
        self.od_matrix = {}
        self.trip_list = []
        
    def load_taz_data(self):
        """Load TAZ information from previous step"""
        print(f"Loading TAZ data from: {TAZ_INFO_FILE}")
        
        if not TAZ_INFO_FILE.exists():
            print(f"ERROR: {TAZ_INFO_FILE} not found. Run create_taz.py first.")
            sys.exit(1)
            
        with open(TAZ_INFO_FILE, 'r') as f:
            self.taz_info = json.load(f)
            
        print(f"Loaded {len(self.taz_info)} TAZ zones")
        
        # Print TAZ summary
        type_counts = {}
        for taz_id, info in self.taz_info.items():
            place_type = info['type']
            type_counts[place_type] = type_counts.get(place_type, 0) + 1
            
        print("TAZ zones by type:")
        for place_type, count in sorted(type_counts.items()):
            print(f"  {place_type:20s}: {count:3d}")
    
    def get_traffic_conditions_for_time_range(self, origin_lat, origin_lng, dest_lat, dest_lng):
        """
        Get traffic conditions for different times of day (4 AM to 10 PM)
        Returns dict with time intervals and their traffic conditions
        """
        if not self.gmaps:
            # Fallback: simulate traffic patterns
            return self.simulate_traffic_patterns()
        
        traffic_data = {}
        
        # Sample traffic conditions every hour from 4 AM to 10 PM
        for hour in range(SIMULATION_START_TIME, SIMULATION_END_TIME + 1):
            # Create departure time for this sample (use tomorrow to avoid past times)
            tomorrow = datetime.now() + timedelta(days=1)
            sample_time = tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
            time_key = f"{hour:02d}:00"
            
            try:
                directions = self.gmaps.directions(
                    origin=(origin_lat, origin_lng),
                    destination=(dest_lat, dest_lng),
                    mode="driving",
                    departure_time=sample_time,
                    traffic_model="best_guess"
                )
                
                if directions and len(directions) > 0:
                    route = directions[0]
                    leg = route['legs'][0]
                    
                    # Get normal duration and duration in traffic
                    normal_duration = leg['duration']['value']
                    traffic_duration = leg.get('duration_in_traffic', {}).get('value', normal_duration)
                    
                    # Calculate traffic condition based on delay
                    delay_ratio = traffic_duration / normal_duration
                    
                    if delay_ratio <= 1.2:
                        traffic_condition = 'normal'
                    elif delay_ratio <= 1.8:
                        traffic_condition = 'slow'
                    else:
                        traffic_condition = 'traffic_jam'
                    
                    traffic_data[time_key] = {
                        'condition': traffic_condition,
                        'duration_normal': normal_duration,
                        'duration_traffic': traffic_duration,
                        'delay_ratio': delay_ratio,
                        'distance': leg['distance']['value']  # meters
                    }
                
                # Small delay to avoid API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                print(f"  Warning: Traffic API error for {time_key}: {e}")
                # Use fallback estimation
                distance = self.estimate_distance(origin_lat, origin_lng, dest_lat, dest_lng)
                traffic_data[time_key] = {
                    'condition': 'normal',
                    'duration_normal': int(distance / (30 * 1000 / 3600)),  # 30 km/h
                    'duration_traffic': int(distance / (25 * 1000 / 3600)),  # 25 km/h
                    'delay_ratio': 1.2,
                    'distance': distance
                }
        
        return traffic_data
    
    def simulate_traffic_patterns(self):
        """Simulate realistic traffic patterns when API is not available"""
        traffic_data = {}
        
        for hour in range(SIMULATION_START_TIME, SIMULATION_END_TIME + 1):
            time_key = f"{hour:02d}:00"
            
            # Simulate traffic patterns based on time of day
            if 6 <= hour <= 9 or 17 <= hour <= 19:  # Peak hours
                condition = 'traffic_jam' if random.random() < 0.6 else 'slow'
            elif 10 <= hour <= 16 or 20 <= hour <= 21:  # Moderate hours
                condition = 'slow' if random.random() < 0.4 else 'normal'
            else:  # Off-peak hours
                condition = 'normal'
            
            # Estimate base travel time (will be calculated per OD pair)
            traffic_data[time_key] = {
                'condition': condition,
                'delay_ratio': {'normal': 1.0, 'slow': 1.5, 'traffic_jam': 2.2}[condition]
            }
        
        return traffic_data
    
    def estimate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate straight-line distance between two points in meters"""
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lng2 - lng1)
        
        a = math.sin(delta_phi/2)**2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def get_road_capacity(self, route_length_km):
        """
        Estimate road capacity based on route length and road type
        For now, use default capacity - could be enhanced with road type detection
        """
        return ROAD_CAPACITY['default']
    
    def calculate_insertion_rate(self, traffic_condition, route_length_km=None, trip_duration_hours=None):
        """
        Return vehicles per hour based on traffic condition label from Google Maps.
        SUMO insertion rate should be a demand (veh/hr).

        Args:
            traffic_condition: 'normal', 'slow', or 'traffic_jam' (from Google Maps)
            route_length_km: optional (kept for API compatibility)
            trip_duration_hours: optional (kept for API compatibility)

        Returns:
            float: insertion rate (vehicles per hour)
        """
        # Defensive: normalize the key and validate
        if not isinstance(traffic_condition, str):
            raise TypeError("traffic_condition must be a string")

        key = traffic_condition.lower()
        if key not in TRAFFIC_DEMAND:
            raise ValueError(f"unknown traffic_condition: {traffic_condition!r}")

        veh_per_hour = float(TRAFFIC_DEMAND[key])
        # ensure at least 1 veh/hr to avoid divide-by-zero later
        return max(veh_per_hour, 1.0)
    
    def generate_od_matrix(self):
        """
        Generate Origin-Destination matrix with traffic conditions and insertion rates
        """
        print(f"\n{'='*70}")
        print(f"GENERATING ORIGIN-DESTINATION MATRIX WITH TRAFFIC CONDITIONS")
        print(f"{'='*70}")
        
        taz_list = list(self.taz_info.items())
        total_pairs = len(taz_list) * (len(taz_list) - 1)  # All pairs except self
        completed = 0
        
        print(f"Processing {total_pairs} OD pairs with traffic conditions (4 AM - 10 PM)...")
        
        for origin_id, origin_info in taz_list:
            self.od_matrix[origin_id] = {}
            
            for dest_id, dest_info in taz_list:
                if origin_id == dest_id:
                    continue  # Skip trips to same zone
                
                # Skip very distant pairs to reduce API calls
                distance_km = self.estimate_distance(
                    origin_info['lat'], origin_info['lng'],
                    dest_info['lat'], dest_info['lng']
                ) / 1000
                
                if distance_km > 10:  # Skip pairs more than 10km apart
                    continue
                
                completed += 1
                if completed % 10 == 0:  # More frequent updates due to longer processing
                    print(f"  Progress: {completed}/{total_pairs} ({completed/total_pairs*100:.1f}%)")
                
                print(f"  Processing: {origin_info['name']} → {dest_info['name']}")
                
                # Get traffic conditions for the full time range (4 AM - 10 PM)
                traffic_data = self.get_traffic_conditions_for_time_range(
                    origin_info['lat'], origin_info['lng'],
                    dest_info['lat'], dest_info['lng']
                )
                
                if not traffic_data:
                    continue  # Skip if we can't get traffic data
                
                # Calculate route length (use straight-line distance * 1.3 for city routing)
                straight_distance = self.estimate_distance(
                    origin_info['lat'], origin_info['lng'],
                    dest_info['lat'], dest_info['lng']
                )
                route_length_km = (straight_distance * 1.3) / 1000  # Convert to km
                
                # Calculate insertion rates for each time period
                time_based_rates = {}
                total_insertion_rate = 0
                
                for time_key, traffic_info in traffic_data.items():
                    trip_duration_hours = traffic_info['duration_traffic'] / 3600
                    
                    insertion_rate = self.calculate_insertion_rate(
                        traffic_info['condition'],
                        route_length_km,
                        trip_duration_hours
                    )
                    
                    time_based_rates[time_key] = {
                        'condition': traffic_info['condition'],
                        'insertion_rate': insertion_rate,
                        'duration_hours': trip_duration_hours
                    }
                    
                    total_insertion_rate += insertion_rate
                
                # Average insertion rate across all time periods
                avg_insertion_rate = total_insertion_rate / len(time_based_rates) if time_based_rates else 1.0
                
                # Store in OD matrix
                self.od_matrix[origin_id][dest_id] = {
                    'route_length_km': route_length_km,
                    'avg_insertion_rate_per_hour': avg_insertion_rate,
                    'time_based_rates': time_based_rates,
                    'origin_name': origin_info['name'],
                    'dest_name': dest_info['name'],
                    'origin_type': origin_info['type'],
                    'dest_type': dest_info['type']
                }
        
        print(f"✅ Generated OD matrix with {completed} valid pairs")
        
        # Save OD matrix
        with open(OUTPUT_OD_MATRIX_FILE, 'w') as f:
            json.dump(self.od_matrix, f, indent=2)
        print(f"✅ Saved OD matrix to {OUTPUT_OD_MATRIX_FILE}")
    
    def generate_trips(self):
        """
        Generate individual trips based on time-varying traffic conditions
        """
        print(f"\n{'='*70}")
        print(f"GENERATING TRIPS WITH TIME-BASED TRAFFIC CONDITIONS")
        print(f"{'='*70}")
        
        trip_id = 0
        total_trips = 0
        
        # Time periods (hourly intervals from 4 AM to 10 PM)
        time_periods = []
        for hour in range(SIMULATION_START_TIME, SIMULATION_END_TIME + 1):
            time_periods.append(f"{hour:02d}:00")
        
        print(f"Generating trips across {len(time_periods)} hourly periods...")
        
        for origin_id, destinations in self.od_matrix.items():
            for dest_id, od_data in destinations.items():
                time_based_rates = od_data.get('time_based_rates', {})
                
                for time_key in time_periods:
                    if time_key not in time_based_rates:
                        continue
                    
                    rate_data = time_based_rates[time_key]
                    insertion_rate = rate_data['insertion_rate']
                    traffic_condition = rate_data['condition']
                    
                    # Calculate trips for this 1-hour period
                    period_duration_hours = 1.0  # 1 hour
                    expected_trips = insertion_rate * period_duration_hours
                    
                    # Use Poisson distribution for realistic arrival patterns
                    num_trips = max(0, int(random.poisson(expected_trips)))
                    
                    if num_trips == 0:
                        continue
                    
                    # Calculate time period bounds in seconds since 4 AM
                    hour, minute = map(int, time_key.split(':'))
                    period_start = (hour - SIMULATION_START_TIME) * 3600 + minute * 60
                    period_end = period_start + TRAFFIC_SAMPLE_INTERVAL  # 1 hour
                    
                    # Generate departure times within this period
                    departure_times = sorted([
                        random.uniform(period_start, min(period_end, SIMULATION_DURATION))
                        for _ in range(num_trips)
                    ])
                    
                    # Create trips
                    for depart_time in departure_times:
                        self.trip_list.append({
                            'id': f"trip_{trip_id}",
                            'origin': origin_id,
                            'destination': dest_id,
                            'depart': f"{depart_time:.2f}",
                            'time_period': time_key,
                            'traffic_condition': traffic_condition,
                            'insertion_rate': insertion_rate,
                            'route_length_km': od_data['route_length_km']
                        })
                        trip_id += 1
                        total_trips += 1
        
        print(f"✅ Generated {total_trips} trips from {len(self.od_matrix)} OD pairs")
        
        # Sort trips by departure time
        self.trip_list.sort(key=lambda x: float(x['depart']))
        
        # Print traffic condition breakdown
        condition_counts = {}
        for trip in self.trip_list:
            condition = trip['traffic_condition']
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        print("Trips by traffic condition:")
        for condition, count in condition_counts.items():
            percentage = (count / total_trips) * 100
            print(f"  {condition:12s}: {count:5d} trips ({percentage:4.1f}%)")
        
        return total_trips
    
    def write_trips_file(self):
        """Write trip definitions to XML file for duarouter"""
        print(f"\nWriting trips to: {OUTPUT_TRIPS_FILE}")
        
        root = ET.Element('trips')
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/trips_file.xsd')
        
        for trip in self.trip_list:
            trip_elem = ET.SubElement(root, 'trip')
            trip_elem.set('id', trip['id'])
            trip_elem.set('depart', trip['depart'])
            trip_elem.set('fromTaz', trip['origin'])
            trip_elem.set('toTaz', trip['destination'])
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space='    ')
        tree.write(OUTPUT_TRIPS_FILE, encoding='UTF-8', xml_declaration=True)
        
        print(f"✅ Wrote {len(self.trip_list)} trips to {OUTPUT_TRIPS_FILE}")
    
    def run_duarouter(self):
        """Run SUMO duarouter to convert trips to routes"""
        print(f"\n{'='*70}")
        print(f"RUNNING DUAROUTER")
        print(f"{'='*70}")
        
        # Ensure output directory exists
        OUTPUT_DEMAND_FILE.parent.mkdir(exist_ok=True)
        
        duarouter_cmd = [
            'duarouter',
            '--net-file', str(NETWORK_FILE),
            '--trip-files', str(OUTPUT_TRIPS_FILE),
            '--additional-files', str(TAZ_FILE),
            '--output-file', str(OUTPUT_DEMAND_FILE),
            '--begin', '0',
            '--end', str(SIMULATION_DURATION),
            '--no-warnings', 'true',
            '--ignore-errors', 'true',
            '--no-step-log', 'true'
        ]
        
        print(f"Running: {' '.join(duarouter_cmd)}")
        
        try:
            result = subprocess.run(
                duarouter_cmd,
                capture_output=True,
                text=True,
                cwd=SCRIPT_DIR
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully generated routes: {OUTPUT_DEMAND_FILE}")
                if result.stdout:
                    print("Duarouter output:")
                    print(result.stdout)
            else:
                print(f"❌ Duarouter failed with return code {result.returncode}")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
                
        except FileNotFoundError:
            print(f"❌ duarouter not found. Please ensure SUMO is installed and in PATH.")
            return False
        except Exception as e:
            print(f"❌ Error running duarouter: {e}")
            return False
            
        return True
    
    def print_summary(self):
        """Print generation summary and statistics"""
        print(f"\n{'='*70}")
        print(f"TRAFFIC-BASED GENERATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"TAZ zones: {len(self.taz_info)}")
        print(f"OD pairs: {sum(len(dests) for dests in self.od_matrix.values())}")
        print(f"Total trips: {len(self.trip_list)}")
        print(f"Simulation period: {SIMULATION_START_TIME}:00 - {SIMULATION_END_TIME}:00 ({SIMULATION_DURATION/3600:.1f} hours)")
        
        # Trip statistics
        if self.trip_list:
            departure_times = [float(trip['depart']) for trip in self.trip_list]
            print(f"First departure: {min(departure_times)/3600:.1f}h after 4 AM")
            print(f"Last departure: {max(departure_times)/3600:.1f}h after 4 AM")
            print(f"Average rate: {len(self.trip_list) / (SIMULATION_DURATION/3600):.1f} trips/hour")
        
        # Traffic condition statistics
        if self.trip_list:
            condition_stats = {}
            for trip in self.trip_list:
                condition = trip['traffic_condition']
                if condition not in condition_stats:
                    condition_stats[condition] = {'count': 0, 'total_rate': 0}
                condition_stats[condition]['count'] += 1
                condition_stats[condition]['total_rate'] += trip['insertion_rate']
            
            print(f"\nTraffic condition analysis:")
            for condition in ['normal', 'slow', 'traffic_jam']:
                if condition in condition_stats:
                    stats = condition_stats[condition]
                    avg_rate = stats['total_rate'] / stats['count']
                    percentage = (stats['count'] / len(self.trip_list)) * 100
                    print(f"  {condition.upper():12s}: {stats['count']:5d} trips ({percentage:4.1f}%) - Avg rate: {avg_rate:.1f} trips/h")
        
        # Top OD pairs by average insertion rate
        print(f"\nTop 10 OD pairs by average insertion rate:")
        od_pairs = []
        for origin_id, destinations in self.od_matrix.items():
            for dest_id, od_data in destinations.items():
                od_pairs.append((
                    f"{od_data['origin_name']} → {od_data['dest_name']}",
                    od_data['avg_insertion_rate_per_hour'],
                    od_data['route_length_km']
                ))
        
        od_pairs.sort(key=lambda x: x[1], reverse=True)
        for i, (pair_name, rate, distance) in enumerate(od_pairs[:10]):
            print(f"  {i+1:2d}. {pair_name[:45]:45s} {rate:6.1f} trips/h ({distance:.1f} km)")
        
        print(f"{'='*70}")

def main():
    print(f"{'='*70}")
    print(f"ORIGIN-DESTINATION TRAFFIC GENERATOR")
    print(f"{'='*70}\n")
    
    # Initialize
    generator = ODTrafficGenerator(GOOGLE_API_KEY)
    
    # Step 1: Load TAZ data
    generator.load_taz_data()
    
    # Step 2: Generate OD matrix with Google Maps travel times
    generator.generate_od_matrix()
    
    # Step 3: Generate individual trips
    num_trips = generator.generate_trips()
    
    if num_trips == 0:
        print("ERROR: No trips generated. Check TAZ data and OD matrix.")
        sys.exit(1)
    
    # Step 4: Write trips file
    generator.write_trips_file()
    
    # Step 5: Run duarouter to create final demand
    success = generator.run_duarouter()
    
    if not success:
        print("ERROR: Failed to generate routes. Check SUMO installation.")
        sys.exit(1)
    
    # Step 6: Print summary
    generator.print_summary()
    
    print(f"\n✅ Done! Generated traffic-based demand using your formula:")
    print(f"   Formula: (traffic_density × road_capacity × route_length) ÷ trip_duration")
    print(f"   Time range: {SIMULATION_START_TIME}:00 - {SIMULATION_END_TIME}:00")
    print(f"   Traffic conditions: Normal, Slow, Traffic Jam (from Google Maps)")
    print(f"\n📁 Output files:")
    print(f"   - Trips file: {OUTPUT_TRIPS_FILE}")
    print(f"   - Routes file: {OUTPUT_DEMAND_FILE}")
    print(f"   - OD matrix: {OUTPUT_OD_MATRIX_FILE}")
    print(f"\n🚗 Next: Run your SUMO simulation with the new traffic-realistic demand.rou.xml!")

if __name__ == '__main__':
    main()