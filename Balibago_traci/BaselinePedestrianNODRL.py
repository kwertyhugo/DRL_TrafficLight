import os
import sys
import traci
import numpy as np
import csv
from keras.utils import to_categorical
from keras.models import load_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === OUTPUT DIRECTORY ===
os.makedirs('./Balibago_traci/output_A2C_baseline', exist_ok=True)

# === LOAD TRAINED MODELS ===
print("\n" + "=" * 70)
print("Loading Trained Baseline A2C Models for Testing")
print("=" * 70)

north_model_path = './Balibago_traci/models_A2C_baseline/North_A2CAgent_Baseline.keras'
south_model_path = './Balibago_traci/models_A2C_baseline/South_A2CAgent_Baseline.keras'

if not os.path.exists(north_model_path):
    sys.exit(f"ERROR: North model not found at {north_model_path}")
if not os.path.exists(south_model_path):
    sys.exit(f"ERROR: South model not found at {south_model_path}")

north_model = load_model(north_model_path)
south_model = load_model(south_model_path)
print(f"  ✓ Loaded North Agent from {north_model_path}")
print(f"  ✓ Loaded South Agent from {south_model_path}")
print("=" * 70)

# === SUMO ENVIRONMENT ===
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# === SCENARIOS ===
scenarios = [
    {
        'name'      : 'normal',
        'route_file': 'Balibago_traci/demand/flows_normal_traffic.rou.xml',
        'stats'     : 'Balibago_traci/output_A2C_baseline/test_normal_stats.xml',
        'trips'     : 'Balibago_traci/output_A2C_baseline/test_normal_trips.xml',
        'metrics'   : 'Balibago_traci/output_A2C_baseline/test_normal_metrics.csv',
        'summary'   : 'Balibago_traci/output_A2C_baseline/test_normal_summary.txt',
    },
    {
        'name'      : 'slow',
        'route_file': 'Balibago_traci/demand/flows_slow_traffic.rou.xml',
        'stats'     : 'Balibago_traci/output_A2C_baseline/test_slow_stats.xml',
        'trips'     : 'Balibago_traci/output_A2C_baseline/test_slow_trips.xml',
        'metrics'   : 'Balibago_traci/output_A2C_baseline/test_slow_metrics.csv',
        'summary'   : 'Balibago_traci/output_A2C_baseline/test_slow_summary.txt',
    },
    {
        'name'      : 'jam',
        'route_file': 'Balibago_traci/demand/flows_jam_traffic.rou.xml',
        'stats'     : 'Balibago_traci/output_A2C_baseline/test_jam_stats.xml',
        'trips'     : 'Balibago_traci/output_A2C_baseline/test_jam_trips.xml',
        'metrics'   : 'Balibago_traci/output_A2C_baseline/test_jam_metrics.csv',
        'summary'   : 'Balibago_traci/output_A2C_baseline/test_jam_summary.txt',
    },
]

# --- SIMULATION VARIABLES ---
stepLength = 0.1
actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)
max_sim_steps = 576000

detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4",
    "e2_5", "e2_6", "e2_7", "e2_8", "e2_9",
    "e2_10", "e2_11", "e2_12"
]
detector_count = len(detector_ids)
north_detector_ids = detector_ids[:8]
south_detector_ids = detector_ids[8:]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _subscribe_all_detectors():
    vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(
            det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_context_vars)
        traci.lanearea.subscribe(det_id, vehicle_vars)

def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
    if not vehicle_data:
        return 0
    weights = {"car": 1.0, "jeep": 1.5, "bus": 2.2,
               "truck": 2.5, "motorcycle": 0.3, "tricycle": 0.5}
    for data in vehicle_data.values():
        v_type   = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        sumWait += waitTime * weights.get(v_type, 1.0)
    return sumWait

def _northIntersection_queue():
    queues = [_weighted_waits(f"e2_{i}") for i in range(8)]
    return queues

def _southIntersection_queue():
    queues = [_weighted_waits(f"e2_{i}") for i in range(8, 13)]
    return queues

def predict_action(model, state):
    """Get action from trained model (greedy - no exploration)"""
    state_batch = np.expand_dims(state, axis=0)
    action_probs = model.predict(state_batch, verbose=0)[0]
    return np.argmax(action_probs)

def save_metrics_csv(filename, metrics_list):
    """Save test metrics to CSV"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time_min', 'Avg_Jam_Length_m', 'Throughput_veh_per_min', 
                        'North_Queue', 'South_Queue', 'Total_Queue',
                        'North_Jam_Length_m', 'South_Jam_Length_m'])
        writer.writerows(metrics_list)

# ============================================================
# RUN SCENARIO
# ============================================================

def run_scenario(scenario):
    """Run a single scenario and return its results."""
    
    print("\n" + "=" * 70)
    print(f"BASELINE A2C - {scenario['name'].upper()} TRAFFIC")
    print("=" * 70)
    print("Configuration:")
    print("  - Two intersections: North (4902876117) | South (12188714)")
    print("  - Detectors: e2_0–e2_7 (North) | e2_8–e2_12 (South)")
    print("  - North: 8-phase cycle | South: 6-phase cycle")
    print("  - Baseline network (no pedestrian signals)")
    print("=" * 70 + "\n")
    
    Sumo_config = [
        'sumo',
        '-c', 'Balibago_traci/baselinePed.sumocfg',
        '--route-files', scenario['route_file'],
        '--step-length', '0.1',
        '--delay', '0',
        '--lateral-resolution', '0.1',
        '--statistic-output', scenario['stats'],
        '--tripinfo-output', scenario['trips']
    ]

    northCurrentPhase = 0
    northCurrentPhaseDuration = 30
    southCurrentPhase = 0
    southCurrentPhaseDuration = 30

    step_counter = 0
    metric_observation_count = 0
    throughput_total = 0
    jam_length_total = 0
    total_queue_north = 0
    total_queue_south = 0
    north_jam_length_total = 0
    south_jam_length_total = 0
    north_throughput_total = 0
    south_throughput_total = 0

    metrics_timeline = []

    traci.start(Sumo_config)
    _subscribe_all_detectors()

    print(f"  Running test for {scenario['name']}_traffic...")

    while traci.simulation.getMinExpectedNumber() > 0 and step_counter < max_sim_steps:
        step_counter += 1
        northCurrentPhaseDuration -= stepLength
        southCurrentPhaseDuration -= stepLength
        
        north_decision_needed = (northCurrentPhaseDuration <= 0) and (northCurrentPhase % 2 == 0)
        south_decision_needed = (southCurrentPhaseDuration <= 0) and (southCurrentPhase % 2 == 0)
        
        next_action_N_idx = None
        next_action_S_idx = None
        
        if north_decision_needed:
            queue = np.array(_northIntersection_queue())
            norm_q_north = queue / 2000.0
            n_phase_oh = to_categorical(northCurrentPhase // 2, num_classes=4).flatten()
            obs_north = np.concatenate([norm_q_north, n_phase_oh]).astype(np.float32)
            next_action_N_idx = predict_action(north_model, obs_north)
        elif northCurrentPhaseDuration <= 0:
            next_action_N_idx = 5
        
        if south_decision_needed:
            queue = np.array(_southIntersection_queue())
            norm_q_south = queue / 1000.0
            s_phase_oh = to_categorical(southCurrentPhase // 2, num_classes=3).flatten()
            obs_south = np.concatenate([norm_q_south, s_phase_oh]).astype(np.float32)
            next_action_S_idx = predict_action(south_model, obs_south)
        elif southCurrentPhaseDuration <= 0:
            next_action_S_idx = 5
        
        # Apply North (8-phase cycle)
        if northCurrentPhaseDuration <= 0:
            northCurrentPhase = (northCurrentPhase + 1) % 8
            traci.trafficlight.setPhase("4902876117", northCurrentPhase)
            
            if northCurrentPhase % 2 == 1:
                northCurrentPhaseDuration = 5
            else:
                duration_adj = actionSpace[next_action_N_idx]
                base = {0: 45, 2: 130, 4: 30, 6: 90}.get(northCurrentPhase, 30)
                northCurrentPhaseDuration = max(5, min(180, base + duration_adj))
            
            traci.trafficlight.setPhaseDuration("4902876117", northCurrentPhaseDuration)
        
        # Apply South (6-phase cycle)
        if southCurrentPhaseDuration <= 0:
            southCurrentPhase = (southCurrentPhase + 1) % 6
            traci.trafficlight.setPhase("12188714", southCurrentPhase)
            
            if southCurrentPhase % 2 == 1:
                southCurrentPhaseDuration = 5
            else:
                duration_adj = actionSpace[next_action_S_idx]
                base = {0: 30, 2: 30, 4: 45}.get(southCurrentPhase, 30)
                southCurrentPhaseDuration = max(5, min(180, base + duration_adj))
            
            traci.trafficlight.setPhaseDuration("12188714", southCurrentPhaseDuration)
        
        TRACK_INTERVAL_STEPS = int(60 / stepLength)
        if step_counter % TRACK_INTERVAL_STEPS == 0:
            jam_length = 0
            throughput = 0
            north_jam_length = 0
            south_jam_length = 0
            north_throughput = 0
            south_throughput = 0
            metric_observation_count += 1
            
            for det_id in detector_ids:
                detector_stats = traci.lanearea.getSubscriptionResults(det_id)
                if not detector_stats:
                    continue
                
                det_jam = detector_stats.get(traci.constants.JAM_LENGTH_METERS, 0)
                det_throughput = detector_stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)
                
                jam_length += det_jam
                throughput += det_throughput
                
                if det_id in north_detector_ids:
                    north_jam_length += det_jam
                    north_throughput += det_throughput
                else:
                    south_jam_length += det_jam
                    south_throughput += det_throughput
            
            jam_length /= detector_count
            jam_length_total += jam_length
            throughput_total += throughput
            
            north_jam_length /= 8
            south_jam_length /= 5
            north_jam_length_total += north_jam_length
            south_jam_length_total += south_jam_length
            north_throughput_total += north_throughput
            south_throughput_total += south_throughput
            
            north_queue = sum(_northIntersection_queue())
            south_queue = sum(_southIntersection_queue())
            total_queue_north += north_queue
            total_queue_south += south_queue
            
            time_min = step_counter * stepLength / 60
            metrics_timeline.append([
                f"{time_min:.1f}",
                f"{jam_length:.2f}",
                f"{throughput:.2f}",
                f"{north_queue:.2f}",
                f"{south_queue:.2f}",
                f"{north_queue + south_queue:.2f}",
                f"{north_jam_length:.2f}",
                f"{south_jam_length:.2f}"
            ])
        
        traci.simulationStep()
        
        if step_counter >= max_sim_steps:
            print(f"\nReached simulation step limit ({max_sim_steps}). Stopping.")
            break

    traci.close()

    # Calculate averages
    obs = metric_observation_count if metric_observation_count > 0 else 1
    avg_jam = jam_length_total / obs
    avg_throughput = throughput_total / obs
    avg_queue_north = total_queue_north / obs
    avg_queue_south = total_queue_south / obs
    avg_north_jam = north_jam_length_total / obs
    avg_south_jam = south_jam_length_total / obs
    avg_north_throughput = north_throughput_total / obs
    avg_south_throughput = south_throughput_total / obs
    
    results = {
        'scenario': scenario['name'],
        'sim_steps': step_counter,
        'observations': metric_observation_count,
        'jam_avg': avg_jam,
        'throughput_avg': avg_throughput,
        'north_jam_avg': avg_north_jam,
        'north_throughput_avg': avg_north_throughput,
        'south_jam_avg': avg_south_jam,
        'south_throughput_avg': avg_south_throughput,
    }
    
    # Print scenario results
    print("\n" + "=" * 70)
    print(f"SCENARIO COMPLETE - {scenario['name'].upper()} TRAFFIC")
    print("=" * 70)
    print(f"Total simulation steps executed : {step_counter}")
    print(f"Total metric observations       : {metric_observation_count}")
    print(f"\n{'─'*70}")
    print("Overall Network Performance")
    print(f"{'─'*70}")
    print(f"  Average Jam Length  : {avg_jam:.2f} m")
    print(f"  Average Throughput  : {avg_throughput:.2f} vehicles/interval")
    print(f"\n{'─'*70}")
    print("North Intersection  (e2_0 – e2_7 | 8-phase)")
    print(f"{'─'*70}")
    print(f"  Average Jam Length  : {avg_north_jam:.2f} m")
    print(f"  Average Throughput  : {avg_north_throughput:.2f} vehicles/interval")
    print(f"\n{'─'*70}")
    print("South Intersection  (e2_8 – e2_12 | 6-phase)")
    print(f"{'─'*70}")
    print(f"  Average Jam Length  : {avg_south_jam:.2f} m")
    print(f"  Average Throughput  : {avg_south_throughput:.2f} vehicles/interval")
    
    # Save metrics
    save_metrics_csv(scenario['metrics'], metrics_timeline)
    print(f"\n  ✓ Saved metrics to {scenario['metrics']}")
    
    # Save summary
    with open(scenario['summary'], 'w') as f:
        f.write(f"Test Results - {scenario['name']}_traffic (Baseline A2C)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Average Jam Length (Overall)    : {avg_jam:.2f} m\n")
        f.write(f"Average Jam Length (North)      : {avg_north_jam:.2f} m\n")
        f.write(f"Average Jam Length (South)      : {avg_south_jam:.2f} m\n")
        f.write(f"Average Throughput (Overall)    : {avg_throughput:.2f} veh/interval\n")
        f.write(f"Average Throughput (North)      : {avg_north_throughput:.2f} veh/interval\n")
        f.write(f"Average Throughput (South)      : {avg_south_throughput:.2f} veh/interval\n\n")
        f.write(f"Average North Queue             : {avg_queue_north:.2f}\n")
        f.write(f"Average South Queue             : {avg_queue_south:.2f}\n")
        f.write(f"Average Total Queue             : {avg_queue_north + avg_queue_south:.2f}\n")
    print(f"  ✓ Saved summary to {scenario['summary']}")
    
    print(f"\n{'─'*70}")
    print("Output files saved:")
    print(f"  Statistics : {scenario['stats']}")
    print(f"  Trip Info  : {scenario['trips']}")
    print("=" * 70)
    
    return results

# === RUN ALL SCENARIOS ===
all_results = []
for scenario in scenarios:
    result = run_scenario(scenario)
    all_results.append(result)

# === FINAL SUMMARY ACROSS ALL SCENARIOS ===
print("\n" + "=" * 70)
print("ALL SCENARIOS COMPLETE - BASELINE A2C SUMMARY")
print("=" * 70)
print(f"{'Scenario':<10} {'Jam (m)':<12} {'Throughput':<14} {'N Jam (m)':<12} {'S Jam (m)':<12}")
print(f"{'─'*60}")
for r in all_results:
    print(f"{r['scenario']:<10} {r['jam_avg']:<12.2f} {r['throughput_avg']:<14.2f} "
        f"{r['north_jam_avg']:<12.2f} {r['south_jam_avg']:<12.2f}")
print("=" * 70)