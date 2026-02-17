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
    '-c', r'Olivarez_traci\SignalizedPed.sumocfg',
    '--route-files', r'Olivarez_traci\demand\flows_normal_traffic.rou.xml',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_NoDRL\SP_NoDRL_stats_trafficjam.xml',
    '--tripinfo-output', r'Olivarez_traci\output_NoDRL\SP_NoDRL_trips_trafficjam.xml'
]

# === SIMULATION VARIABLES ===
stepLength = 0.1
detector_count = 7

# === METRICS TRACKING ===
throughput_average = 0
throughput_total = 0
jam_length_average = 0
jam_length_total = 0
metric_observation_count = 0

# === DETECTOR SUBSCRIPTION ===
detector_ids = ["e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10"]

def _subscribe_all_detectors():
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribe(det_id, vehicle_vars)

# === START SUMO ===
traci.start(Sumo_config)
_subscribe_all_detectors()

print("\n" + "=" * 70)
print("NO-DRL BASELINE - TRAFFIC JAM SCENARIO")
print("=" * 70)
print("Configuration:")
print("  - Traffic Flow: flows_traffic_jam.rou.xml")
print("  - Fixed signal timing (no adaptation)")
print("=" * 70 + "\n")

# === SIMULATION LOOP ===
sim_step_count = 0
max_sim_steps = 50000

while traci.simulation.getMinExpectedNumber() > 0 and sim_step_count < max_sim_steps:
    
    # === PERIODIC METRICS TRACKING (every 60 seconds) ===
    if sim_step_count % int(60 / stepLength) == 0:
        jam_length = 0
        throughput = 0
        metric_observation_count += 1
        
        for det_id in detector_ids:
            detector_stats = traci.lanearea.getSubscriptionResults(det_id)
            
            if not detector_stats:
                print("Lane Data Error: Undetected")
                break
            
            jam_length += detector_stats.get(traci.constants.JAM_LENGTH_METERS, 0)
            throughput += detector_stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)
        
        jam_length /= detector_count
        jam_length_total += jam_length
        throughput_total += throughput
    
    traci.simulationStep()
    sim_step_count += 1
    
    if sim_step_count >= max_sim_steps:
        print(f"\nReached simulation step limit ({max_sim_steps}). Stopping.")
        break

# === CALCULATE AVERAGES ===
jam_length_average = jam_length_total / metric_observation_count if metric_observation_count > 0 else 0
throughput_average = throughput_total / metric_observation_count if metric_observation_count > 0 else 0

# === RESULTS ===
print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
print(f"Total simulation steps executed: {sim_step_count}")
print(f"\nPerformance Metrics:")
print(f"  Average Jam Length: {jam_length_average:.2f} meters")
print(f"  Average Throughput: {throughput_average:.2f} vehicles/minute")
print(f"  Total Observations: {metric_observation_count}")
print(f"\nOutput files saved:")
print(f"  - Statistics: Olivarez_traci/output_NoDRL/SP_NoDRL_stats_trafficjam.xml")
print(f"  - Trip Info: Olivarez_traci/output_NoDRL/SP_NoDRL_trips_trafficjam.xml")
print("=" * 70)

traci.close()