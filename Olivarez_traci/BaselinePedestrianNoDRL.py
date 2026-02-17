import os
import sys 
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo',
    '-c', r'Olivarez_traci\baselinePed.sumocfg',
    '--step-length', '0.1',
    '--delay', '20',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_NoDRL\BP_NoDRL_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_NoDRL\BP_NoDRL_trips.xml'
]

# Metrics Initialization
detector_count = 7
stepLength = 0.1
step_counter = 0

jam_length_total = 0
throughput_total = 0
metric_observation_count = 0

# Detector IDs
detector_ids = ["e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10"]

def _subscribe_all_detectors():
    global detector_ids
    # VAR_INTERVAL_NUMBER is required for throughput
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribe(det_id, vehicle_vars)

traci.start(Sumo_config)
_subscribe_all_detectors()

print("Baseline Simulation Started - Sampling metrics every 60 seconds...")

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    step_counter += 1
    
    # Track metrics every 60 seconds (600 steps at 0.1s step-length)
    if step_counter % int(60 / stepLength) == 0:
        minute_jam = 0
        minute_throughput = 0
        metric_observation_count += 1
        
        for det_id in detector_ids:
            detector_stats = traci.lanearea.getSubscriptionResults(det_id)
            if detector_stats:
                minute_jam += detector_stats.get(traci.constants.JAM_LENGTH_METERS, 0)
                minute_throughput += detector_stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)
            else:
                print(f"Warning: No data for detector {det_id} at step {step_counter}")
        
        # Calculate system-wide average jam length for this minute
        jam_length_total += (minute_jam / detector_count)
        # Sum total vehicles that passed all detectors in this minute
        throughput_total += minute_throughput

traci.close()

# --- Final Metrics Calculation ---
if metric_observation_count > 0:
    jam_length_average = jam_length_total / metric_observation_count
    # Since we sample every 60s, the average of our throughput totals IS the vehicles/minute
    throughput_average = throughput_total / metric_observation_count
else:
    jam_length_average = 0
    throughput_average = 0

print("\n" + "=" * 40)
print("FINAL BASELINE PERFORMANCE METRICS")
print("=" * 40)
print(f"Queue Length (Average): {jam_length_average:.2f} meters")
print(f"Throughput (Average):   {throughput_average:.2f} vehicles/minute")
print(f"Total Observations:     {metric_observation_count} (minutes)")
print(f"Total Simulation Steps: {step_counter}")
print("=" * 40)