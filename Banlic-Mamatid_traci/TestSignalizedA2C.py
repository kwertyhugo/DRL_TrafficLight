import os
import sys
import traci
import numpy as np
import csv
from keras.utils import to_categorical
from keras.models import load_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === OUTPUT DIRECTORY ===
os.makedirs('./Banlic-Mamatid_traci/output_A2C', exist_ok=True)

# === LOAD TRAINED MODEL ===
print("\n" + "=" * 70)
print("Loading Trained A2C Model for Testing - Banlic-Mamatid Network")
print("=" * 70)

main_model_path = './Banlic-Mamatid_traci/models_A2C/Main_A2CAgent_Signalized.keras'

if not os.path.exists(main_model_path):
    sys.exit(f"ERROR: Main model not found at {main_model_path}")

main_model = load_model(main_model_path)
print(f"  ✓ Loaded Main Agent from {main_model_path}")
print("=" * 70)

# === SUMO ENVIRONMENT ===
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# --- SIMULATION VARIABLES ---
stepLength  = 0.1
actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)

detector_ids   = ["e2_0", "e2_1", "e2_2", "e2_3", "e2_4", "e2_5"]
detector_count = len(detector_ids)

JUNCTION_1 = "253768576"
JUNCTION_2 = "253499548"

BASE_DURATIONS = {0: 30, 2: 30, 4: 45, 6: 60, 8: 25}
TOTAL_PHASES   = 10

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _subscribe_all_detectors():
    vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    vehicle_vars         = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(
            det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_context_vars)
        traci.lanearea.subscribe(det_id, vehicle_vars)

def _weighted_waits(detector_id):
    sumWait      = 0
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

def _intersection_queue():
    return [_weighted_waits(det_id) for det_id in detector_ids]

def predict_action(model, state):
    """Get greedy action from trained model (no exploration)."""
    state_batch  = np.expand_dims(state, axis=0)
    action_probs = model.predict(state_batch, verbose=0)[0]
    return np.argmax(action_probs)

def _trafficLight_phase(action_index, currentPhase, currentPhaseDuration):
    """
    Advance phase for both junctions — mirrors SignalizedPedestrianA2C.py exactly.
    Phase increments INSIDE this function, then duration is applied to the NEW phase.
    """
    currentPhase = (currentPhase + 1) % TOTAL_PHASES

    traci.trafficlight.setPhase(JUNCTION_1, currentPhase)
    traci.trafficlight.setPhase(JUNCTION_2, currentPhase)

    if currentPhase % 2 == 1:      # yellow/transition
        currentPhaseDuration = 5
    else:                           # green — apply action
        duration_adj         = actionSpace[action_index]
        base                 = BASE_DURATIONS.get(currentPhase, 30)
        currentPhaseDuration = max(5, min(180, base + duration_adj))

    traci.trafficlight.setPhaseDuration(JUNCTION_1, currentPhaseDuration)
    traci.trafficlight.setPhaseDuration(JUNCTION_2, currentPhaseDuration)

    return currentPhase, currentPhaseDuration

def save_metrics_csv(filename, metrics_list):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time_min', 'Avg_Jam_Length_m', 'Throughput_veh_per_min',
                         'Total_Queue', 'Phase'])
        writer.writerows(metrics_list)

# ============================================================
# SCENARIO RUNNER
# ============================================================

def run_test_scenario(scenario_name, route_file):
    """Run a single test scenario and return results dict."""

    print("\n" + "=" * 70)
    print(f"Testing Scenario: {scenario_name.upper()}")
    print("=" * 70)

    scenario_prefix = scenario_name.lower().replace(" ", "_").replace("/", "_")

    Sumo_config = [
        'sumo',
        '-c', 'Banlic-Mamatid_traci/signalizedPed.sumocfg',
        '--route-files', route_file,
        '--step-length', '0.1',
        '--delay', '0',
        '--lateral-resolution', '0.1',
        '--statistic-output', f'Banlic-Mamatid_traci/output_A2C/test_{scenario_prefix}_stats.xml',
        '--tripinfo-output',  f'Banlic-Mamatid_traci/output_A2C/test_{scenario_prefix}_trips.xml'
    ]

    # Reset per-scenario state
    currentPhase         = 0
    currentPhaseDuration = BASE_DURATIONS[0]

    step_counter             = 0
    metric_observation_count = 0
    throughput_total         = 0
    jam_length_total         = 0
    total_queue              = 0

    prevAction       = None
    metrics_timeline = []

    TRACK_INTERVAL_STEPS = int(60 / stepLength)

    traci.start(Sumo_config, port=8814)
    _subscribe_all_detectors()

    print(f"  Running simulation for {scenario_name}...")

    while traci.simulation.getMinExpectedNumber() > 0 and step_counter < 576000:
        step_counter         += 1
        currentPhaseDuration -= stepLength

        # ── DECISION: only on green phases ────────────────────────────
        if currentPhaseDuration <= 0:
            if currentPhase % 2 == 0:   # green phase — ask the model
                queue      = np.array(_intersection_queue())
                norm_queue = queue / 1000.0
                phase_oh   = to_categorical(currentPhase // 2, num_classes=5).flatten()
                obs        = np.concatenate([norm_queue, phase_oh]).astype(np.float32)
                actionIndex = predict_action(main_model, obs)
                prevAction  = actionIndex
            else:                        # yellow — carry over last green action
                actionIndex = prevAction if prevAction is not None else 5

            # Apply phase transition (increments phase internally)
            currentPhase, currentPhaseDuration = _trafficLight_phase(
                actionIndex, currentPhase, currentPhaseDuration)

        # ── METRICS every 60 s ────────────────────────────────────────
        if step_counter % TRACK_INTERVAL_STEPS == 0:
            jam_length = 0
            throughput = 0
            metric_observation_count += 1

            for det_id in detector_ids:
                stats = traci.lanearea.getSubscriptionResults(det_id)
                if not stats:
                    continue
                jam_length += stats.get(traci.constants.JAM_LENGTH_METERS, 0)
                throughput += stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)

            jam_length       /= detector_count
            jam_length_total += jam_length
            throughput_total += throughput

            current_queue = sum(_intersection_queue())
            total_queue  += current_queue

            time_min = step_counter * stepLength / 60
            metrics_timeline.append([
                f"{time_min:.1f}",
                f"{jam_length:.2f}",
                f"{throughput:.2f}",
                f"{current_queue:.2f}",
                f"{currentPhase}"
            ])

        traci.simulationStep()

    traci.close()

    # ── RESULTS ───────────────────────────────────────────────────────
    results = {}
    if metric_observation_count > 0:
        results['avg_jam']        = jam_length_total / metric_observation_count
        results['avg_throughput'] = throughput_total / metric_observation_count
        results['avg_queue']      = total_queue      / metric_observation_count

        print(f"\n  Results for {scenario_name}:")
        print(f"    Average Jam Length              : {results['avg_jam']:.2f} m")
        print(f"    Average Throughput              : {results['avg_throughput']:.2f} veh/min")
        print(f"    Average Queue (Weighted Wait)   : {results['avg_queue']:.2f}")

        csv_file = f'./Banlic-Mamatid_traci/output_A2C/test_{scenario_prefix}_metrics.csv'
        save_metrics_csv(csv_file, metrics_timeline)
        print(f"    ✓ Saved metrics  → {csv_file}")

        summary_file = f'./Banlic-Mamatid_traci/output_A2C/test_{scenario_prefix}_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Test Results - {scenario_name}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Average Jam Length              : {results['avg_jam']:.2f} m\n")
            f.write(f"Average Throughput              : {results['avg_throughput']:.2f} veh/min\n")
            f.write(f"Average Queue (Weighted Wait)   : {results['avg_queue']:.2f}\n")
        print(f"    ✓ Saved summary  → {summary_file}")

    return results

# ============================================================
# RUN ALL THREE SCENARIOS
# ============================================================

print("\n" + "=" * 70)
print("SIGNALIZED A2C MODEL — COMPREHENSIVE TESTING")
print("Banlic-Mamatid Network")
print("Running Normal → Slow → Jam/Heavy sequentially...")
print("=" * 70)

test_scenarios = [
    {
        'name'      : 'Normal Traffic',
        'route_file': 'Banlic-Mamatid_traci/demand/flows_normal_traffic.rou.xml'
    },
    {
        'name'      : 'Slow Traffic',
        'route_file': 'Banlic-Mamatid_traci/demand/flows_slow_traffic.rou.xml'
    },
    {
        'name'      : 'Jam/Heavy Traffic',
        'route_file': 'Banlic-Mamatid_traci/demand/flows_traffic_jam.rou.xml'
    },
]

all_results = {}

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n[{i}/{len(test_scenarios)}] Starting {scenario['name']}...")
    all_results[scenario['name']] = run_test_scenario(
        scenario['name'], scenario['route_file'])
    print(f"  ✓ {scenario['name']} completed!")

# ============================================================
# COMPREHENSIVE SUMMARY
# ============================================================

metrics_def = [
    ('Avg Jam Length (m)',      'avg_jam'),
    ('Avg Throughput (veh/min)', 'avg_throughput'),
    ('Avg Queue (Weighted)',    'avg_queue'),
]

print("\n" + "=" * 70)
print("COMPREHENSIVE TEST SUMMARY — SIGNALIZED A2C MODEL")
print("Banlic-Mamatid Network")
print("=" * 70)
print(f"{'Metric':<35} {'Normal':<15} {'Slow':<15} {'Jam/Heavy':<15}")
print("-" * 80)

for metric_name, metric_key in metrics_def:
    n = all_results.get('Normal Traffic',   {}).get(metric_key, 0)
    s = all_results.get('Slow Traffic',     {}).get(metric_key, 0)
    j = all_results.get('Jam/Heavy Traffic',{}).get(metric_key, 0)
    print(f"{metric_name:<35} {n:<15.2f} {s:<15.2f} {j:<15.2f}")

summary_file = './Banlic-Mamatid_traci/output_A2C/test_comprehensive_summary.txt'
with open(summary_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("SIGNALIZED A2C MODEL — COMPREHENSIVE TEST RESULTS\n")
    f.write("Banlic-Mamatid Network\n")
    f.write("=" * 70 + "\n\n")

    for scenario_name, results in all_results.items():
        f.write(f"\n{scenario_name}:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Average Jam Length              : {results.get('avg_jam', 0):.2f} m\n")
        f.write(f"  Average Throughput              : {results.get('avg_throughput', 0):.2f} veh/min\n")
        f.write(f"  Average Queue (Weighted Wait)   : {results.get('avg_queue', 0):.2f}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("COMPARISON ACROSS SCENARIOS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"{'Metric':<35} {'Normal':<15} {'Slow':<15} {'Jam/Heavy':<15}\n")
    f.write("-" * 80 + "\n")
    for metric_name, metric_key in metrics_def:
        n = all_results.get('Normal Traffic',   {}).get(metric_key, 0)
        s = all_results.get('Slow Traffic',     {}).get(metric_key, 0)
        j = all_results.get('Jam/Heavy Traffic',{}).get(metric_key, 0)
        f.write(f"{metric_name:<35} {n:<15.2f} {s:<15.2f} {j:<15.2f}\n")

print(f"\n✓ Comprehensive summary saved → {summary_file}")
print("\n" + "=" * 70)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nOutput files generated per scenario:")
print("  • test_<scenario>_metrics.csv")
print("  • test_<scenario>_summary.txt")
print("  • test_<scenario>_stats.xml")
print("  • test_<scenario>_trips.xml")
print("  • test_comprehensive_summary.txt")
print("\nLocation: ./Banlic-Mamatid_traci/output_A2C/")
print("=" * 70)