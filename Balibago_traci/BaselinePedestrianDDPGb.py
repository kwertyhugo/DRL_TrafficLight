import os
import sys
import traci
import numpy as np
import csv
from keras.utils import to_categorical

# ==========================================
# PATH SETUP
# ==========================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.makedirs('./Balibago_traci/output_DDPG', exist_ok=True)

try:
    from models.DDPG import DDPGAgent as ddpg
except ImportError:
    sys.exit("Error: Could not find DDPGAgent in models/DDPG.py")

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_MODE = 0         # 1 = Train, 0 = Test
BATCH_SIZE = 64
TRAIN_FREQUENCY = 100
GAMMA = 0.99
TAU = 0.001
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
NOISE_DECAY = 0.9999
MIN_NOISE_STD = 0.01
ACTION_SCALE = 25.0
reward_scale = 10.0

action_low = np.array([-1.0], dtype=np.float32)
action_high = np.array([1.0], dtype=np.float32)

# ==========================================
# AGENT INITIALIZATION
# ==========================================
# North: 8 Detectors + 1 Ped + 4 Phase One-Hot = 13 State Size
NorthAgent = ddpg(
    state_size=13,
    action_size=1,
    action_low=action_low,
    action_high=action_high,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    name='Baseline_North_DDPGAgent'
)

# South: 5 Detectors + 1 Ped + 4 Phase One-Hot = 10 State Size
SouthAgent = ddpg(
    state_size=10,
    action_size=1,
    action_low=action_low,
    action_high=action_high,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    name='Baseline_South_DDPGAgent'
)

# ==========================================
# LOAD EXISTING MODEL & HISTORY
# ==========================================
model_path = './Balibago_traci/output_DDPG/'

if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
    print("\n" + "=" * 60)
    print("Found saved history. Attempting to load...")
    try:
        NorthAgent.load()
        SouthAgent.load()
        print(" - Model weights loaded successfully.")
    except Exception as e:
        print(f" - Warning: Could not load model weights. Starting fresh. Error: {e}")

    if TRAIN_MODE == 1:
        try:
            NorthAgent.load_replay_buffer()
            SouthAgent.load_replay_buffer()
            print(" - Replay buffers loaded.")
        except Exception as e:
            print(f" - Warning: Could not load replay buffers. Error: {e}")
    print("=" * 60)
else:
    print("No history found. Starting fresh.")

# ==========================================
# SUMO SETUP
# ==========================================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo',  # Use 'sumo-gui' to see the simulation
    '-c', 'Balibago_traci/signalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Balibago_traci/output_DDPG/jam_baseline_stats.xml',
    '--tripinfo-output', r'Balibago_traci/output_DDPG/jam_baseline_trips.xml'
]

# ==========================================
# SIMULATION VARIABLES
# ==========================================
stepLength = 0.1
step_counter = 0
MAX_STEPS = 150000  # Safety limit

northCurrentPhase = 0
northCurrentPhaseDuration = 45.0   # Starting duration = base of phase 0
southCurrentPhase = 0
southCurrentPhaseDuration = 25.0   # Starting duration = base of phase 0

# Fixed base durations (used as reference; agent adjusts green phases)
NORTH_BASE_DURATIONS = {0: 45.0, 2: 130.0, 4: 30.0, 6: 90.0}
SOUTH_BASE_DURATIONS = {0: 25.0, 2: 30.0,  4: 40.0, 6: 45.0}
YELLOW_DURATION = 5.0

detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4", "e2_5", "e2_6", "e2_7",
    "e2_8", "e2_9", "e2_10", "e2_11", "e2_12"
]
detector_count = len(detector_ids)

# Metrics
throughput_total = 0
jam_length_total = 0
metric_observation_count = 0

# Training history
reward_history_N, actor_loss_N, critic_loss_N = [], [], []
total_reward_N = 0

reward_history_S, actor_loss_S, critic_loss_S = [], [], []
total_reward_S = 0

northPrevState = None
northPrevAction = None
southPrevState = None
southPrevAction = None

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id,
        traci.constants.CMD_GET_PERSON_VARIABLE,
        10.0,
        [traci.constants.VAR_WAITING_TIME]
    )

def _subscribe_all_detectors():
    vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_context_vars)
        traci.lanearea.subscribe(det_id, vehicle_vars)

def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
    if not vehicle_data:
        return 0
    weights = {"car": 1.0, "jeep": 1.5, "bus": 2.2, "truck": 2.5, "motorcycle": 0.3, "tricycle": 0.5}
    for data in vehicle_data.values():
        v_type = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        sumWait += waitTime * weights.get(v_type, 1.0)
    return sumWait

def _northIntersection_queue():
    queues = [_weighted_waits(f"e2_{i}") for i in range(8)]
    pedestrian = 0
    junction_data = traci.junction.getContextSubscriptionResults("4902876117")
    if junction_data:
        for data in junction_data.values():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return queues + [pedestrian]  # 9 values total

def _southIntersection_queue():
    queues = [_weighted_waits(f"e2_{i}") for i in range(8, 13)]
    pedestrian = 0
    junction_data = traci.junction.getContextSubscriptionResults("12188714")
    if junction_data:
        for data in junction_data.values():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return queues + [pedestrian]  # 6 values total

def calculate_reward(unnormalized_queue):
    if unnormalized_queue is None:
        return 0.0
    return -float(np.sum(unnormalized_queue))

def save_history(filename, headers, reward_hist, a_loss_hist, c_loss_hist, frequency):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        for i in range(len(reward_hist)):
            r = reward_hist[i] if i < len(reward_hist) else 0
            a = a_loss_hist[i] if i < len(a_loss_hist) else 0
            c = c_loss_hist[i] if i < len(c_loss_hist) else 0
            writer.writerow([i * frequency, r, a, c])

# ==========================================
# MAIN EXECUTION
# ==========================================
traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("4902876117")
_junctionSubscription("12188714")

print("=" * 60)
print(f"STARTING BALIBAGO BASELINE DDPG | {'TRAIN' if TRAIN_MODE == 1 else 'TEST'} Mode")
print("=" * 60)

try:
    while traci.simulation.getMinExpectedNumber() > 0:
        step_counter += 1

        northCurrentPhaseDuration -= stepLength
        southCurrentPhaseDuration -= stepLength

        # -------------------------------------------------
        # 1. OBSERVATION PHASE
        # -------------------------------------------------
        obs_north = None
        obs_south = None
        reward_north = 0
        reward_south = 0

        north_decision_needed = (northCurrentPhaseDuration <= 0) and (northCurrentPhase % 2 == 0)
        south_decision_needed = (southCurrentPhaseDuration <= 0) and (southCurrentPhase % 2 == 0)

        # --- North State ---
        if north_decision_needed:
            raw_q = _northIntersection_queue()          # 9 values (8 detectors + 1 ped)
            queue_arr = np.array(raw_q, dtype=np.float32)
            n_norm_queue = queue_arr / 100.0

            n_phase_oh = to_categorical(northCurrentPhase // 2, num_classes=4).flatten()
            obs_north = np.concatenate([n_norm_queue, n_phase_oh]).astype(np.float32)  # 13

            reward_north = calculate_reward(queue_arr) / reward_scale
            total_reward_N += reward_north

        # --- South State ---
        if south_decision_needed:
            raw_q = _southIntersection_queue()          # 6 values (5 detectors + 1 ped)
            queue_arr = np.array(raw_q, dtype=np.float32)
            s_norm_queue = queue_arr / 100.0

            s_phase_oh = to_categorical(southCurrentPhase // 2, num_classes=4).flatten()
            obs_south = np.concatenate([s_norm_queue, s_phase_oh]).astype(np.float32)  # 10

            reward_south = calculate_reward(queue_arr) / reward_scale
            total_reward_S += reward_south

        # -------------------------------------------------
        # 2. MEMORY & TRAINING PHASE
        # -------------------------------------------------
        if TRAIN_MODE == 1:
            if north_decision_needed and northPrevState is not None:
                NorthAgent.remember(northPrevState, northPrevAction, reward_north, obs_north, False)
                if len(NorthAgent.replay_buffer) >= BATCH_SIZE:
                    a_loss, c_loss = NorthAgent.train()
                    if a_loss is not None:
                        actor_loss_N.append(a_loss)
                        critic_loss_N.append(c_loss)
                        reward_history_N.append(total_reward_N)
                        total_reward_N = 0

            if south_decision_needed and southPrevState is not None:
                SouthAgent.remember(southPrevState, southPrevAction, reward_south, obs_south, False)
                if len(SouthAgent.replay_buffer) >= BATCH_SIZE:
                    a_loss, c_loss = SouthAgent.train()
                    if a_loss is not None:
                        actor_loss_S.append(a_loss)
                        critic_loss_S.append(c_loss)
                        reward_history_S.append(total_reward_S)
                        total_reward_S = 0

        # -------------------------------------------------
        # 3. ACTION SELECTION PHASE
        # -------------------------------------------------
        north_action_val = None
        south_action_val = None

        if north_decision_needed:
            if TRAIN_MODE == 1:
                NorthAgent.noise.std_dev = max(MIN_NOISE_STD, NorthAgent.noise.std_dev * NOISE_DECAY)
            north_action_val = NorthAgent.get_action(obs_north, add_noise=(TRAIN_MODE == 1))
            northPrevState = obs_north
            northPrevAction = north_action_val

            if TRAIN_MODE == 1:
                print(f"North | Step {step_counter} | R: {reward_north:.3f} | Act: {north_action_val[0]:.3f}")

        elif northCurrentPhaseDuration <= 0:
            north_action_val = northPrevAction if northPrevAction is not None else np.zeros(1)

        if south_decision_needed:
            if TRAIN_MODE == 1:
                SouthAgent.noise.std_dev = max(MIN_NOISE_STD, SouthAgent.noise.std_dev * NOISE_DECAY)
            south_action_val = SouthAgent.get_action(obs_south, add_noise=(TRAIN_MODE == 1))
            southPrevState = obs_south
            southPrevAction = south_action_val

            if TRAIN_MODE == 1:
                print(f"South | Step {step_counter} | R: {reward_south:.3f} | Act: {south_action_val[0]:.3f}")

        elif southCurrentPhaseDuration <= 0:
            south_action_val = southPrevAction if southPrevAction is not None else np.zeros(1)

        # -------------------------------------------------
        # 4. EXECUTION PHASE
        # -------------------------------------------------

        # Apply North
        if northCurrentPhaseDuration <= 0:
            northCurrentPhase = (northCurrentPhase + 1) % 8
            traci.trafficlight.setPhase("4902876117", northCurrentPhase)

            if northCurrentPhase % 2 == 1:
                # Yellow — fixed, no agent adjustment
                northCurrentPhaseDuration = YELLOW_DURATION
            else:
                base = NORTH_BASE_DURATIONS.get(northCurrentPhase, 30.0)
                adjustment = float(north_action_val[0]) * ACTION_SCALE
                northCurrentPhaseDuration = max(5.0, min(180.0, base + adjustment))

            traci.trafficlight.setPhaseDuration("4902876117", float(northCurrentPhaseDuration))

        # Apply South
        if southCurrentPhaseDuration <= 0:
            southCurrentPhase = (southCurrentPhase + 1) % 8
            traci.trafficlight.setPhase("12188714", southCurrentPhase)

            if southCurrentPhase % 2 == 1:
                # Yellow — fixed, no agent adjustment
                southCurrentPhaseDuration = YELLOW_DURATION
            else:
                base = SOUTH_BASE_DURATIONS.get(southCurrentPhase, 30.0)
                adjustment = float(south_action_val[0]) * ACTION_SCALE
                southCurrentPhaseDuration = max(5.0, min(180.0, base + adjustment))

            traci.trafficlight.setPhaseDuration("12188714", float(southCurrentPhaseDuration))

        # -------------------------------------------------
        # 5. METRICS COLLECTION
        # -------------------------------------------------
        TRACK_INTERVAL_STEPS = int(60 / stepLength)  # Every 60 seconds
        if step_counter % TRACK_INTERVAL_STEPS == 0:
            jam_length = 0
            throughput = 0
            metric_observation_count += 1

            for det_id in detector_ids:
                detector_stats = traci.lanearea.getSubscriptionResults(det_id)
                if not detector_stats:
                    continue
                jam_length += detector_stats.get(traci.constants.JAM_LENGTH_METERS, 0)
                throughput += detector_stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)

            jam_length /= detector_count
            jam_length_total += jam_length
            throughput_total += throughput

        traci.simulationStep()

except Exception as e:
    print(f"Simulation interrupted: {e}")
    import traceback
    traceback.print_exc()

finally:
    # ==========================================
    # CLEANUP & SAVE
    # ==========================================
    print("Cleaning up...")
    try:
        traci.close()
    except:
        pass

    if metric_observation_count > 0:
        print("\n --- Final Metrics ---")
        print(f" Avg Queue Length: {jam_length_total / metric_observation_count:.2f}")
        print(f" Avg Throughput:   {throughput_total / metric_observation_count:.2f}")
    else:
        print("No metrics collected.")

    if TRAIN_MODE == 1:
        print("\nSaving trained models...")
        NorthAgent.save()
        SouthAgent.save()
        NorthAgent.save_replay_buffer()
        SouthAgent.save_replay_buffer()

        print("Saving training history...")
        headers = ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss']

        save_history(
            './Balibago_traci/output_DDPG/baseline_North_history.csv',
            headers, reward_history_N, actor_loss_N, critic_loss_N, TRAIN_FREQUENCY
        )
        save_history(
            './Balibago_traci/output_DDPG/baseline_South_history.csv',
            headers, reward_history_S, actor_loss_S, critic_loss_S, TRAIN_FREQUENCY
        )
        print("All histories saved successfully!")

    print("Simulation Finished.")