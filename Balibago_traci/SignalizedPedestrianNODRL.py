import os
import sys
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# === OUTPUT DIRECTORIES ===
os.makedirs('./Balibago_traci/output_NoDRL', exist_ok=True)

# === SCENARIOS ===
scenarios = [
    {
        'name'      : 'normal',
        'route_file': 'Balibago_traci/demand/flows_normal_traffic.rou.xml',
        'stats'     : 'Balibago_traci/output_NoDRL/SD_NoDRL_normal_stats.xml',
        'trips'     : 'Balibago_traci/output_NoDRL/SD_NoDRL_normal_trips.xml',
    },
    {
        'name'      : 'slow',
        'route_file': 'Balibago_traci/demand/flows_slow_traffic.rou.xml',
        'stats'     : 'Balibago_traci/output_NoDRL/SD_NoDRL_slow_stats.xml',
        'trips'     : 'Balibago_traci/output_NoDRL/SD_NoDRL_slow_trips.xml',
    },
    {
        'name'      : 'jam',
        'route_file': 'Balibago_traci/demand/flows_jam_traffic.rou.xml',
        'stats'     : 'Balibago_traci/output_NoDRL/SD_NoDRL_jam_stats.xml',
        'trips'     : 'Balibago_traci/output_NoDRL/SD_NoDRL_jam_trips.xml',
    },
]

# === SIMULATION CONSTANTS ===
stepLength     = 0.1
detector_count = 13
max_sim_steps  = 576000   # matches A2C loop limit

# === DETECTOR IDs ===
detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4",
    "e2_5", "e2_6", "e2_7", "e2_8", "e2_9",
    "e2_10", "e2_11", "e2_12"
]
north_detector_ids = detector_ids[:8]   # e2_0 – e2_7
south_detector_ids = detector_ids[8:]   # e2_8 – e2_12

# === FIXED SIGNAL TIMING ===
NORTH_TL_ID = "4902876117"
NORTH_GREEN_DURATIONS = {0: 45, 2: 130, 4: 30, 6: 90}
NORTH_YELLOW_DURATION = 5
NORTH_TOTAL_PHASES    = 8

SOUTH_TL_ID = "12188714"
SOUTH_GREEN_DURATIONS = {0: 25, 2: 30, 4: 40, 6: 45}
SOUTH_YELLOW_DURATION = 5
SOUTH_TOTAL_PHASES    = 8


def _subscribe_all_detectors():
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribe(det_id, vehicle_vars)


def _apply_fixed_timing(northCurrentPhase, northCurrentPhaseDuration,
                        southCurrentPhase, southCurrentPhaseDuration):
    """Advance both traffic lights using their fixed phase durations."""

    # --- North ---
    northCurrentPhaseDuration -= stepLength
    if northCurrentPhaseDuration <= 0:
        northCurrentPhase = (northCurrentPhase + 1) % NORTH_TOTAL_PHASES
        traci.trafficlight.setPhase(NORTH_TL_ID, northCurrentPhase)
        if northCurrentPhase % 2 == 1:
            northCurrentPhaseDuration = NORTH_YELLOW_DURATION
        else:
            northCurrentPhaseDuration = NORTH_GREEN_DURATIONS.get(northCurrentPhase, 30)
        traci.trafficlight.setPhaseDuration(NORTH_TL_ID, northCurrentPhaseDuration)

    # --- South ---
    southCurrentPhaseDuration -= stepLength
    if southCurrentPhaseDuration <= 0:
        southCurrentPhase = (southCurrentPhase + 1) % SOUTH_TOTAL_PHASES
        traci.trafficlight.setPhase(SOUTH_TL_ID, southCurrentPhase)
        if southCurrentPhase % 2 == 1:
            southCurrentPhaseDuration = SOUTH_YELLOW_DURATION
        else:
            southCurrentPhaseDuration = SOUTH_GREEN_DURATIONS.get(southCurrentPhase, 30)
        traci.trafficlight.setPhaseDuration(SOUTH_TL_ID, southCurrentPhaseDuration)

    return northCurrentPhase, northCurrentPhaseDuration, southCurrentPhase, southCurrentPhaseDuration


def run_scenario(scenario):
    """Run a single scenario and return its results."""

    Sumo_config = [
        'sumo',
        '-c', 'Balibago_traci/signalizedPed.sumocfg',
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
    north_jam_total          = 0
    north_throughput_total   = 0
    south_jam_total          = 0
    south_throughput_total   = 0

    # Reset phase tracking
    northCurrentPhase         = 0
    northCurrentPhaseDuration = NORTH_GREEN_DURATIONS[0]
    southCurrentPhase         = 0
    southCurrentPhaseDuration = SOUTH_GREEN_DURATIONS[0]

    traci.start(Sumo_config)
    _subscribe_all_detectors()

    print("\n" + "=" * 70)
    print(f"NO-DRL BASELINE - BALIBAGO NETWORK  [ {scenario['name'].upper()} TRAFFIC ]")
    print("=" * 70)
    print("Configuration:")
    print("  - Two intersections: North (4902876117) | South (12188714)")
    print("  - Detectors: e2_0–e2_7 (North) | e2_8–e2_12 (South)")
    print("  - Fixed signal timing (no adaptation)")
    print("  - North cycle: 45 / 130 / 30 / 90 s green phases")
    print("  - South cycle: 25 / 30  / 40 / 45 s green phases")
    print("=" * 70 + "\n")

    sim_step_count = 0

    while traci.simulation.getMinExpectedNumber() > 0 and sim_step_count < max_sim_steps:

        northCurrentPhase, northCurrentPhaseDuration, \
        southCurrentPhase, southCurrentPhaseDuration = _apply_fixed_timing(
            northCurrentPhase, northCurrentPhaseDuration,
            southCurrentPhase, southCurrentPhaseDuration
        )

        # Periodic metrics (every 60 seconds)
        if sim_step_count % int(60 / stepLength) == 0:
            jam_length       = 0
            throughput       = 0
            north_jam        = 0
            north_throughput = 0
            south_jam        = 0
            south_throughput = 0
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

                if det_id in north_detector_ids:
                    north_jam        += jl
                    north_throughput += tp
                else:
                    south_jam        += jl
                    south_throughput += tp

            jam_length /= detector_count
            jam_length_total       += jam_length
            throughput_total       += throughput
            north_jam              /= len(north_detector_ids)
            south_jam              /= len(south_detector_ids)
            north_jam_total        += north_jam
            north_throughput_total += north_throughput
            south_jam_total        += south_jam
            south_throughput_total += south_throughput

        traci.simulationStep()
        sim_step_count += 1

        if sim_step_count >= max_sim_steps:
            print(f"\nReached simulation step limit ({max_sim_steps}). Stopping.")
            break

    traci.close()

    # Calculate averages
    obs = metric_observation_count if metric_observation_count > 0 else 1
    results = {
        'scenario'            : scenario['name'],
        'sim_steps'           : sim_step_count,
        'observations'        : metric_observation_count,
        'jam_avg'             : jam_length_total        / obs,
        'throughput_avg'      : throughput_total        / obs,
        'north_jam_avg'       : north_jam_total         / obs,
        'north_throughput_avg': north_throughput_total  / obs,
        'south_jam_avg'       : south_jam_total         / obs,
        'south_throughput_avg': south_throughput_total  / obs,
    }

    # Print scenario results
    print("\n" + "=" * 70)
    print(f"SCENARIO COMPLETE - {scenario['name'].upper()} TRAFFIC")
    print("=" * 70)
    print(f"Total simulation steps executed : {sim_step_count}")
    print(f"Total metric observations       : {metric_observation_count}")
    print(f"\n{'─'*70}")
    print("Overall Network Performance")
    print(f"{'─'*70}")
    print(f"  Average Jam Length  : {results['jam_avg']:.2f} m")
    print(f"  Average Throughput  : {results['throughput_avg']:.2f} vehicles/interval")
    print(f"\n{'─'*70}")
    print("North Intersection  (e2_0 – e2_7 | TL: 4902876117)")
    print(f"{'─'*70}")
    print(f"  Average Jam Length  : {results['north_jam_avg']:.2f} m")
    print(f"  Average Throughput  : {results['north_throughput_avg']:.2f} vehicles/interval")
    print(f"\n{'─'*70}")
    print("South Intersection  (e2_8 – e2_12 | TL: 12188714)")
    print(f"{'─'*70}")
    print(f"  Average Jam Length  : {results['south_jam_avg']:.2f} m")
    print(f"  Average Throughput  : {results['south_throughput_avg']:.2f} vehicles/interval")
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
print(f"{'Scenario':<10} {'Jam (m)':<12} {'Throughput':<14} {'N Jam (m)':<12} {'S Jam (m)':<12}")
print(f"{'─'*60}")
for r in all_results:
    print(f"{r['scenario']:<10} {r['jam_avg']:<12.2f} {r['throughput_avg']:<14.2f} "
        f"{r['north_jam_avg']:<12.2f} {r['south_jam_avg']:<12.2f}")
print("=" * 70)