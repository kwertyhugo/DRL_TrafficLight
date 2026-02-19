import os
import sys
import traci
import csv
import numpy as np

# ==========================================
# PATH SETUP
# ==========================================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Create output directory
os.makedirs('./Balibago_traci/output_DDPG', exist_ok=True)

# ==========================================
# CONFIGURATION
# ==========================================
Sumo_config = [
    'sumo', # Change to 'sumo-gui' to watch the baseline run
    '-c', 'Balibago_traci/signalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--statistic-output', r'Balibago_traci/output_DDPG/baseline_slow_stats.xml',
    '--tripinfo-output', r'Balibago_traci/output_DDPG/baseline_slow_trips.xml'    
]

# Intersection IDs from your DDPG code
NORTH_TL = "4902876117"
SOUTH_TL = "12188714"

# Fixed Timings (matching your DDPG 'base' values)
NORTH_TIMINGS = {0: 45, 2: 130, 4: 30, 6: 90}
SOUTH_TIMINGS = {0: 25, 2: 30, 4: 40, 6: 45}
YELLOW_TIME = 5.0

# Detectors for metrics
detector_ids = [f"e2_{i}" for i in range(13)]

# ==========================================
# SIMULATION VARIABLES
# ==========================================
step_counter = 0
stepLength = 0.1

northCurrentPhase = 0
northTimer = NORTH_TIMINGS[0]

southCurrentPhase = 0
southTimer = SOUTH_TIMINGS[0]

# Metrics
jam_length_total = 0
throughput_total = 0
metric_observation_count = 0

# ==========================================
# MAIN LOOP
# ==========================================
traci.start(Sumo_config)

# Subscribe to lanearea detectors for metrics
for det in detector_ids:
    traci.lanearea.subscribe(det, [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER])

print("Running Balibago Static Baseline (Fixed Timings)...")

try:
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step_counter += 1
        
        # Decrement timers
        northTimer -= stepLength
        southTimer -= stepLength

        # --- North Logic ---
        if northTimer <= 0:
            northCurrentPhase = (northCurrentPhase + 1) % 8
            traci.trafficlight.setPhase(NORTH_TL, northCurrentPhase)
            
            if northCurrentPhase % 2 == 1: # Yellow
                northTimer = YELLOW_TIME
            else: # Green
                northTimer = NORTH_TIMINGS.get(northCurrentPhase, 30)
            traci.trafficlight.setPhaseDuration(NORTH_TL, northTimer)

        # --- South Logic ---
        if southTimer <= 0:
            southCurrentPhase = (southCurrentPhase + 1) % 8
            traci.trafficlight.setPhase(SOUTH_TL, southCurrentPhase)
            
            if southCurrentPhase % 2 == 1: # Yellow
                southTimer = YELLOW_TIME
            else: # Green
                southTimer = SOUTH_TIMINGS.get(southCurrentPhase, 30)
            traci.trafficlight.setPhaseDuration(SOUTH_TL, southTimer)

        # --- Metrics Collection (Every 60s) ---
        if step_counter % 600 == 0:
            jam_length = 0
            throughput = 0
            metric_observation_count += 1
            
            for det_id in detector_ids:
                stats = traci.lanearea.getSubscriptionResults(det_id)
                if stats:
                    jam_length += stats.get(traci.constants.JAM_LENGTH_METERS, 0)
                    throughput += stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)
            
            jam_length_total += (jam_length / len(detector_ids))
            throughput_total += throughput

except Exception as e:
    print(f"Simulation Error: {e}")

finally:
    traci.close()
    
    # Final Output
    avg_jam = jam_length_total / metric_observation_count if metric_observation_count > 0 else 0
    avg_tp = throughput_total / metric_observation_count if metric_observation_count > 0 else 0
    
    print("\n" + "="*30)
    print("STATIC BASELINE COMPLETE")
    print(f"Avg Queue Length: {avg_jam:.2f} m")
    print(f"Avg Throughput: {avg_tp:.2f} vehicles")
    print("="*30)

    # Save to CSV for side-by-side comparison
    with open('./Balibago_traci/output_DDPG/baseline_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Fixed_Baseline_Value'])
        writer.writerow(['Avg_Queue_Length', avg_jam])
        writer.writerow(['Avg_Throughput', avg_tp])