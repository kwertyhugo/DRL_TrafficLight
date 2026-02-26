import os
import sys
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# === OUTPUT DIRECTORIES ===
os.makedirs('./Banlic-Mamatid_traci/output_NoDRL', exist_ok=True)

# === SCENARIOS ===
scenarios = [
    {
        'name'      : 'normal',
        'route_file': 'Banlic-Mamatid_traci/demand/flows_normal_traffic.rou.xml',
        'stats'     : 'Banlic-Mamatid_traci/output_NoDRL/SD_NoDRL_normal_stats.xml',
        'trips'     : 'Banlic-Mamatid_traci/output_NoDRL/SD_NoDRL_normal_trips.xml',
    },
    {
        'name'      : 'slow',
        'route_file': 'Banlic-Mamatid_traci/demand/flows_slow_traffic.rou.xml',
        'stats'     : 'Banlic-Mamatid_traci/output_NoDRL/SD_NoDRL_slow_stats.xml',
        'trips'     : 'Banlic-Mamatid_traci/output_NoDRL/SD_NoDRL_slow_trips.xml',
    },
    {
        'name'      : 'jam',
        'route_file': 'Banlic-Mamatid_traci/demand/flows_traffic_jam.rou.xml',
        'stats'     : 'Banlic-Mamatid_traci/output_NoDRL/SD_NoDRL_jam_stats.xml',
        'trips'     : 'Banlic-Mamatid_traci/output_NoDRL/SD_NoDRL_jam_trips.xml',
    },
]

# === SIMULATION CONSTANTS ===
stepLength     = 0.1
detector_count = 6
max_sim_steps  = 576000   # matches A2C loop limit

# === DETECTOR IDs ===
# Based on A2C file: 6 detectors for Banlic-Mamatid network
detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4", "e2_5"
]

# === FIXED SIGNAL TIMING ===
# Banlic-Mamatid has TWO junctions with 10-phase cycle (0-9)
# 5 green phases: 0, 2, 4, 6, 8
# 5 yellow/transition phases: 1, 3, 5, 7, 9

JUNCTION_1_ID = "253768576"  # Main junction from A2C
JUNCTION_2_ID = "253499548"  # Secondary junction from A2C

# Base green durations for each green phase (from A2C code)
GREEN_DURATIONS = {0: 30, 2: 30, 4: 45, 6: 60, 8: 25}
YELLOW_DURATION = 5
TOTAL_PHASES    = 10


def _subscribe_all_detectors():
    """Subscribe to detector data for metrics collection."""
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribe(det_id, vehicle_vars)


def _apply_fixed_timing(currentPhase, currentPhaseDuration):
    """
    Advance both traffic lights using their fixed phase durations.
    Both junctions operate in sync with the same timing.
    """
    currentPhaseDuration -= stepLength
    
    if currentPhaseDuration <= 0:
        # Move to next phase in the 10-phase cycle
        currentPhase = (currentPhase + 1) % TOTAL_PHASES
        
        # Set phase for both junctions
        traci.trafficlight.setPhase(JUNCTION_1_ID, currentPhase)
        traci.trafficlight.setPhase(JUNCTION_2_ID, currentPhase)
        
        # Determine duration based on phase type
        if currentPhase % 2 == 1:  # Yellow/transition phases (1, 3, 5, 7, 9)
            currentPhaseDuration = YELLOW_DURATION
        else:  # Green phases (0, 2, 4, 6, 8)
            currentPhaseDuration = GREEN_DURATIONS.get(currentPhase, 30)
        
        # Set duration for both junctions
        traci.trafficlight.setPhaseDuration(JUNCTION_1_ID, currentPhaseDuration)
        traci.trafficlight.setPhaseDuration(JUNCTION_2_ID, currentPhaseDuration)

    return currentPhase, currentPhaseDuration


def run_scenario(scenario):
    """Run a single scenario and return its results."""

    Sumo_config = [
        'sumo',
        '-c', 'Banlic-Mamatid_traci/signalizedPed.sumocfg',
        '--route-files', scenario['route_file'],
        '--step-length', '0.1',
        '--delay', '0',
        '--lateral-resolution', '0.1',
        '--statistic-output', scenario['stats'],
        '--tripinfo-output',  scenario['trips'],
    ]

    # Reset metrics
    throughput_total         = 0
    jam_length_total         = 0
    metric_observation_count = 0

    # Reset phase tracking
    currentPhase         = 0
    currentPhaseDuration = GREEN_DURATIONS[0]

    traci.start(Sumo_config)
    _subscribe_all_detectors()

    print("\n" + "=" * 70)
    print(f"NO-DRL BASELINE - BANLIC-MAMATID NETWORK  [ {scenario['name'].upper()} TRAFFIC ]")
    print("=" * 70)
    print("Configuration:")
    print("  - Two intersections: 253768576 (Main) | 253499548 (Secondary)")
    print("  - Detectors: e2_0 – e2_5 (6 detectors)")
    print("  - Fixed signal timing (no adaptation)")
    print("  - 10-phase cycle with 5 green phases")
    print("  - Green durations: 30 / 30 / 45 / 60 / 25 seconds")
    print("  - Yellow duration: 5 seconds")
    print("=" * 70 + "\n")

    sim_step_count = 0

    while traci.simulation.getMinExpectedNumber() > 0 and sim_step_count < max_sim_steps:

        currentPhase, currentPhaseDuration = _apply_fixed_timing(
            currentPhase, currentPhaseDuration
        )

        # Periodic metrics (every 60 seconds)
        if sim_step_count % int(60 / stepLength) == 0:
            jam_length       = 0
            throughput       = 0
            metric_observation_count += 1

            for det_id in detector_ids:
                detector_stats = traci.lanearea.getSubscriptionResults(det_id)

                if not detector_stats:
                    print(f"Lane Data Error: Undetected ({det_id})")
                    continue

                jl = detector_stats.get(traci.constants.JAM_LENGTH_METERS, 0)
                tp = detector_stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)
                jam_length += jl
                throughput += tp

            jam_length /= detector_count
            jam_length_total += jam_length
            throughput_total += throughput

        traci.simulationStep()
        sim_step_count += 1

        if sim_step_count >= max_sim_steps:
            print(f"\nReached simulation step limit ({max_sim_steps}). Stopping.")
            break

    traci.close()

    # Calculate averages
    obs = metric_observation_count if metric_observation_count > 0 else 1
    results = {
        'scenario'      : scenario['name'],
        'sim_steps'     : sim_step_count,
        'observations'  : metric_observation_count,
        'jam_avg'       : jam_length_total   / obs,
        'throughput_avg': throughput_total   / obs,
    }

    # Print scenario results
    print("\n" + "=" * 70)
    print(f"SCENARIO COMPLETE - {scenario['name'].upper()} TRAFFIC")
    print("=" * 70)
    print(f"Total simulation steps executed : {sim_step_count}")
    print(f"Total metric observations       : {metric_observation_count}")
    print(f"\n{'─'*70}")
    print("Network Performance")
    print(f"{'─'*70}")
    print(f"  Average Jam Length  : {results['jam_avg']:.2f} m")
    print(f"  Average Throughput  : {results['throughput_avg']:.2f} vehicles/interval")
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
print("ALL SCENARIOS COMPLETE - SUMMARY")
print("=" * 70)
print(f"{'Scenario':<10} {'Jam (m)':<12} {'Throughput':<14}")
print(f"{'─'*40}")
for r in all_results:
    print(f"{r['scenario']:<10} {r['jam_avg']:<12.2f} {r['throughput_avg']:<14.2f}")
print("=" * 70)