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

Sumo_config = [
    'sumo',
    '-c', 'Balibago_traci/signalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Balibago_traci/output_NoDRL/SD_NoDRL_jam_stats.xml',
    '--tripinfo-output', r'Balibago_traci/output_NoDRL/SD_NoDRL_jam_trips.xml'
]

# === SIMULATION VARIABLES ===
stepLength    = 0.1
detector_count = 13

# === METRICS TRACKING ===
throughput_average       = 0
throughput_total         = 0
jam_length_average       = 0
jam_length_total         = 0
metric_observation_count = 0

# North & South separate metric tracking (mirrors A2C output structure)
north_jam_total       = 0
north_throughput_total = 0
south_jam_total       = 0
south_throughput_total = 0

# === DETECTOR IDs ===
# North intersection: e2_0 – e2_7   (matches A2C NorthAgent)
# South intersection: e2_8 – e2_12  (matches A2C SouthAgent)
detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4",
    "e2_5", "e2_6", "e2_7", "e2_8", "e2_9",
    "e2_10", "e2_11", "e2_12"
]

north_detector_ids = detector_ids[:8]   # e2_0 – e2_7
south_detector_ids = detector_ids[8:]   # e2_8 – e2_12

# === FIXED SIGNAL TIMING ===
# North intersection (TL ID: 4902876117) — 8-phase cycle
# Phase 0: green 45s | Phase 1: yellow 5s
# Phase 2: green 130s | Phase 3: yellow 5s
# Phase 4: green 30s  | Phase 5: yellow 5s
# Phase 6: green 90s  | Phase 7: yellow 5s
NORTH_TL_ID = "4902876117"
NORTH_GREEN_DURATIONS = {0: 45, 2: 130, 4: 30, 6: 90}
NORTH_YELLOW_DURATION = 5
NORTH_TOTAL_PHASES    = 8

# South intersection (TL ID: 12188714) — 8-phase cycle
# Phase 0: green 25s | Phase 1: yellow 5s
# Phase 2: green 30s | Phase 3: yellow 5s
# Phase 4: green 40s | Phase 5: yellow 5s
# Phase 6: green 45s | Phase 7: yellow 5s
SOUTH_TL_ID = "12188714"
SOUTH_GREEN_DURATIONS = {0: 25, 2: 30, 4: 40, 6: 45}
SOUTH_YELLOW_DURATION = 5
SOUTH_TOTAL_PHASES    = 8

# === PHASE TRACKING ===
northCurrentPhase         = 0
northCurrentPhaseDuration = NORTH_GREEN_DURATIONS[0]
southCurrentPhase         = 0
southCurrentPhaseDuration = SOUTH_GREEN_DURATIONS[0]

def _subscribe_all_detectors():
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribe(det_id, vehicle_vars)

def _apply_fixed_timing():
    """Advance both traffic lights using their fixed phase durations."""
    global northCurrentPhase, northCurrentPhaseDuration
    global southCurrentPhase, southCurrentPhaseDuration

    # --- North ---
    northCurrentPhaseDuration -= stepLength
    if northCurrentPhaseDuration <= 0:
        northCurrentPhase = (northCurrentPhase + 1) % NORTH_TOTAL_PHASES
        traci.trafficlight.setPhase(NORTH_TL_ID, northCurrentPhase)

        if northCurrentPhase % 2 == 1:   # yellow/transition phase
            northCurrentPhaseDuration = NORTH_YELLOW_DURATION
        else:                             # green phase
            northCurrentPhaseDuration = NORTH_GREEN_DURATIONS.get(northCurrentPhase, 30)

        traci.trafficlight.setPhaseDuration(NORTH_TL_ID, northCurrentPhaseDuration)

    # --- South ---
    southCurrentPhaseDuration -= stepLength
    if southCurrentPhaseDuration <= 0:
        southCurrentPhase = (southCurrentPhase + 1) % SOUTH_TOTAL_PHASES
        traci.trafficlight.setPhase(SOUTH_TL_ID, southCurrentPhase)

        if southCurrentPhase % 2 == 1:   # yellow/transition phase
            southCurrentPhaseDuration = SOUTH_YELLOW_DURATION
        else:                             # green phase
            southCurrentPhaseDuration = SOUTH_GREEN_DURATIONS.get(southCurrentPhase, 30)

        traci.trafficlight.setPhaseDuration(SOUTH_TL_ID, southCurrentPhaseDuration)

# === START SUMO ===
traci.start(Sumo_config)
_subscribe_all_detectors()

print("\n" + "=" * 70)
print("NO-DRL BASELINE - BALIBAGO NETWORK")
print("=" * 70)
print("Configuration:")
print("  - Two intersections: North (4902876117) | South (12188714)")
print("  - Detectors: e2_0–e2_7 (North) | e2_8–e2_12 (South)")
print("  - Fixed signal timing (no adaptation)")
print("  - North cycle: 45 / 130 / 30 / 90 s green phases")
print("  - South cycle: 25 / 30  / 40 / 45 s green phases")
print("=" * 70 + "\n")

# === SIMULATION LOOP ===
sim_step_count = 0
max_sim_steps  = 576000   # matches A2C loop limit (step_counter < 576000)

while traci.simulation.getMinExpectedNumber() > 0 and sim_step_count < max_sim_steps:

    # Apply fixed signal timing every step
    _apply_fixed_timing()

    # === PERIODIC METRICS TRACKING (every 60 seconds) ===
    if sim_step_count % int(60 / stepLength) == 0:
        jam_length        = 0
        throughput        = 0
        north_jam         = 0
        north_throughput  = 0
        south_jam         = 0
        south_throughput  = 0
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

        # Average jam length across all detectors
        jam_length /= detector_count
        jam_length_total  += jam_length
        throughput_total  += throughput

        # Per-intersection averages
        north_jam        /= len(north_detector_ids)
        south_jam        /= len(south_detector_ids)
        north_jam_total       += north_jam
        north_throughput_total += north_throughput
        south_jam_total       += south_jam
        south_throughput_total += south_throughput

    traci.simulationStep()
    sim_step_count += 1

    if sim_step_count >= max_sim_steps:
        print(f"\nReached simulation step limit ({max_sim_steps}). Stopping.")
        break

# === CALCULATE AVERAGES ===
jam_length_average = jam_length_total / metric_observation_count if metric_observation_count > 0 else 0
throughput_average = throughput_total / metric_observation_count if metric_observation_count > 0 else 0

north_jam_avg        = north_jam_total        / metric_observation_count if metric_observation_count > 0 else 0
north_throughput_avg = north_throughput_total / metric_observation_count if metric_observation_count > 0 else 0
south_jam_avg        = south_jam_total        / metric_observation_count if metric_observation_count > 0 else 0
south_throughput_avg = south_throughput_total / metric_observation_count if metric_observation_count > 0 else 0

# === RESULTS ===
print("\n" + "=" * 70)
print("TEST COMPLETE - NO-DRL BASELINE (BALIBAGO)")
print("=" * 70)
print(f"Total simulation steps executed : {sim_step_count}")
print(f"Total metric observations       : {metric_observation_count}")

print(f"\n{'─'*70}")
print("Overall Network Performance")
print(f"{'─'*70}")
print(f"  Average Jam Length  : {jam_length_average:.2f} m")
print(f"  Average Throughput  : {throughput_average:.2f} vehicles/interval")

print(f"\n{'─'*70}")
print("North Intersection  (e2_0 – e2_7 | TL: 4902876117)")
print(f"{'─'*70}")
print(f"  Average Jam Length  : {north_jam_avg:.2f} m")
print(f"  Average Throughput  : {north_throughput_avg:.2f} vehicles/interval")

print(f"\n{'─'*70}")
print("South Intersection  (e2_8 – e2_12 | TL: 12188714)")
print(f"{'─'*70}")
print(f"  Average Jam Length  : {south_jam_avg:.2f} m")
print(f"  Average Throughput  : {south_throughput_avg:.2f} vehicles/interval")

print(f"\n{'─'*70}")
print("Output files saved:")
print(f"  Statistics : Balibago_traci/output_NoDRL/SD_NoDRL_stats.xml")
print(f"  Trip Info  : Balibago_traci/output_NoDRL/SD_NoDRL_trips.xml")
print("=" * 70)

traci.close()