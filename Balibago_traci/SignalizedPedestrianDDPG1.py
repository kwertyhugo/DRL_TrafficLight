import os
import sys
import traci
import numpy as np
import csv
from keras.utils import to_categorical

# ==========================================
# PATH & MODEL SETUP
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
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
NOISE_STD_INIT = 0.2
NOISE_DECAY = 0.9999

# Action bounds: 4 outputs, each -1 to 1, scaled to ±25s
ACTION_SIZE = 4
action_low  = np.full(ACTION_SIZE, -1.0, dtype=np.float32)
action_high = np.full(ACTION_SIZE,  1.0, dtype=np.float32)
ACTION_SCALE = 25.0

# Green phase index → action vector index
# North green phases: 0, 2, 4, 6  →  action indices 0, 1, 2, 3
NORTH_GREEN_PHASES = {0: 0, 2: 1, 4: 2, 6: 3}
# South green phases: 0, 2, 4, 6  →  action indices 0, 1, 2, 3
SOUTH_GREEN_PHASES = {0: 0, 2: 1, 4: 2, 6: 3}

# Base green durations (seconds)
NORTH_BASE = {0: 45, 2: 130, 4: 30, 6: 90}
SOUTH_BASE  = {0: 25, 2:  30, 4: 40, 6: 45}

# ==========================================
# AGENT INITIALIZATION
# ==========================================
# North: 8 detectors + 4 phase one-hot = 12 state size
NorthAgent = ddpg(
    state_size=12,
    action_size=ACTION_SIZE,
    action_low=action_low,
    action_high=action_high,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    name='North1_DDPGAgent'
)

# South: 5 detectors + 4 phase one-hot = 9 state size
SouthAgent = ddpg(
    state_size=9,
    action_size=ACTION_SIZE,
    action_low=action_low,
    action_high=action_high,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    name='South1_DDPGAgent'
)

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
    'sumo',
    '-c', 'Balibago_traci/signalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Balibago_traci/output_DDPG/test_jam_stats.xml',
    '--tripinfo-output', r'Balibago_traci/output_DDPG/test_jam_trips.xml'
]

# ==========================================
# SIMULATION VARIABLES
# ==========================================
trainMode = 0
stepLength = 0.1

northCurrentPhase = 0
northCurrentPhaseDuration = 30
southCurrentPhase = 0
southCurrentPhaseDuration = 30

detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4", "e2_5", "e2_6", "e2_7",
    "e2_8", "e2_9", "e2_10", "e2_11", "e2_12"
]
detector_count = len(detector_ids)

metric_observation_count = 0
throughput_total = 0
jam_length_total = 0

northPrevState  = None
northPrevAction = None
southPrevState  = None
southPrevAction = None

step_counter = 0
MAX_STEPS = 150000

# History Buffers
reward_history_N, actor_loss_N, critic_loss_N = [], [], []
total_reward_N = 0

reward_history_S, actor_loss_S, critic_loss_S = [], [], []
total_reward_S = 0

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
    weights = {"car": 1.0, "jeep": 1.5, "bus": 2.2, "truck": 2.5,
               "motorcycle": 0.3, "tricycle": 0.5}
    for data in vehicle_data.values():
        v_type  = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        sumWait += waitTime * weights.get(v_type, 1.0)
    return sumWait

def _northIntersection_queue():
    return [_weighted_waits(f"e2_{i}") for i in range(8)]

def _southIntersection_queue():
    return [_weighted_waits(f"e2_{i}") for i in range(8, 13)]

def calculate_reward(current_state):
    if current_state is None:
        return 0
    return -np.sum(current_state)

def save_history(filename, headers, reward_hist, a_loss_hist, c_loss_hist):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        length = len(reward_hist)
        for i in range(length):
            r = reward_hist[i] if i < len(reward_hist) else 0
            a = a_loss_hist[i] if i < len(a_loss_hist) else 0
            c = c_loss_hist[i] if i < len(c_loss_hist) else 0
            writer.writerow([i, r, a, c])

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

    if trainMode == 1:
        try:
            NorthAgent.load_replay_buffer()
            SouthAgent.load_replay_buffer()
            print(" - Replay buffers loaded.")
        except Exception as e:
            print(f" - Warning: Could not load replay buffers. Error: {e}")
    print("=" * 60)
else:
    print("No saved history found. Starting fresh.")

# ==========================================
# MAIN EXECUTION
# ==========================================
traci.start(Sumo_config, port=8814)
_subscribe_all_detectors()
_junctionSubscription("4902876117")
_junctionSubscription("12188714")

print(f"STARTING DDPG | Balibago | action_size={ACTION_SIZE} per agent | trainMode={trainMode}")

try:
    while traci.simulation.getMinExpectedNumber() > 0:
        step_counter += 1

        northCurrentPhaseDuration -= stepLength
        southCurrentPhaseDuration -= stepLength

        # -------------------------------------------------
        # 1. OBSERVATION PHASE
        # -------------------------------------------------
        obs_north  = None
        obs_south  = None
        reward_north = 0
        reward_south = 0

        north_decision_needed = (northCurrentPhaseDuration <= 0) and (northCurrentPhase % 2 == 0)
        south_decision_needed = (southCurrentPhaseDuration <= 0) and (southCurrentPhase % 2 == 0)

        # --- North State ---
        if north_decision_needed:
            queue = np.array(_northIntersection_queue())
            n_norm_queue = queue / 100.0
            n_phase_oh = to_categorical(northCurrentPhase // 2, num_classes=4).flatten()
            obs_north = np.concatenate([n_norm_queue, n_phase_oh]).astype(np.float32)

            if trainMode == 1:
                reward_north = calculate_reward(n_norm_queue * 10)
                total_reward_N += reward_north

        # --- South State ---
        if south_decision_needed:
            queue = np.array(_southIntersection_queue())
            s_norm_queue = queue / 100.0
            s_phase_oh = to_categorical(southCurrentPhase // 2, num_classes=4).flatten()
            obs_south = np.concatenate([s_norm_queue, s_phase_oh]).astype(np.float32)

            if trainMode == 1:
                reward_south = calculate_reward(s_norm_queue * 10)
                total_reward_S += reward_south

        # -------------------------------------------------
        # 2. MEMORY & TRAINING PHASE
        # -------------------------------------------------
        if trainMode == 1:
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
        north_action_vec = None
        south_action_vec = None

        # North: get full 4-action vector, use only the relevant index for current phase
        if north_decision_needed:
            north_action_vec = NorthAgent.get_action(obs_north, add_noise=(trainMode == 1))
            northPrevState  = obs_north
            northPrevAction = north_action_vec
            if trainMode == 1:
                act_idx = NORTH_GREEN_PHASES.get(northCurrentPhase, 0)
                print(f"North | Phase: {northCurrentPhase} | Q: {np.sum(obs_north[:8]):.2f} | "
                      f"R: {reward_north:.2f} | Act[{act_idx}]: {north_action_vec[act_idx]:.3f}")

        elif northCurrentPhaseDuration <= 0:
            north_action_vec = northPrevAction if northPrevAction is not None else np.zeros(ACTION_SIZE)

        # South
        if south_decision_needed:
            south_action_vec = SouthAgent.get_action(obs_south, add_noise=(trainMode == 1))
            southPrevState  = obs_south
            southPrevAction = south_action_vec
            if trainMode == 1:
                act_idx = SOUTH_GREEN_PHASES.get(southCurrentPhase, 0)
                print(f"South | Phase: {southCurrentPhase} | Q: {np.sum(obs_south[:5]):.2f} | "
                      f"R: {reward_south:.2f} | Act[{act_idx}]: {south_action_vec[act_idx]:.3f}")

        elif southCurrentPhaseDuration <= 0:
            south_action_vec = southPrevAction if southPrevAction is not None else np.zeros(ACTION_SIZE)

        # -------------------------------------------------
        # 4. EXECUTION PHASE
        # -------------------------------------------------

        # Apply North
        if northCurrentPhaseDuration <= 0:
            northCurrentPhase = (northCurrentPhase + 1) % 8
            traci.trafficlight.setPhase("4902876117", northCurrentPhase)

            if northCurrentPhase % 2 == 1:
                northCurrentPhaseDuration = 5  # Yellow
            else:
                act_idx = NORTH_GREEN_PHASES.get(northCurrentPhase, 0)
                base = NORTH_BASE.get(northCurrentPhase, 30)
                adjustment = float(north_action_vec[act_idx]) * ACTION_SCALE
                northCurrentPhaseDuration = max(5.0, min(180.0, base + adjustment))

            traci.trafficlight.setPhaseDuration("4902876117", float(northCurrentPhaseDuration))

        # Apply South
        if southCurrentPhaseDuration <= 0:
            southCurrentPhase = (southCurrentPhase + 1) % 8
            traci.trafficlight.setPhase("12188714", southCurrentPhase)

            if southCurrentPhase % 2 == 1:
                southCurrentPhaseDuration = 5  # Yellow
            else:
                act_idx = SOUTH_GREEN_PHASES.get(southCurrentPhase, 0)
                base = SOUTH_BASE.get(southCurrentPhase, 30)
                adjustment = float(south_action_vec[act_idx]) * ACTION_SCALE
                southCurrentPhaseDuration = max(5.0, min(180.0, base + adjustment))

            traci.trafficlight.setPhaseDuration("12188714", float(southCurrentPhaseDuration))

        # -------------------------------------------------
        # 5. METRICS (evaluation mode only)
        # -------------------------------------------------
        if trainMode == 0 and step_counter % 600 == 0:
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
    print("Cleaning up...")
    try:
        traci.close()
    except:
        pass

    if metric_observation_count > 0:
        print("\n Queue Length Avg:", jam_length_total / metric_observation_count)
        print("\n Throughput Avg:", throughput_total / metric_observation_count)

    if trainMode == 1:
        print("Saving trained models...")
        NorthAgent.save()
        SouthAgent.save()

        print("Saving training history...")
        headers = ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss']
        save_history('./Balibago_traci/output_DDPG/North_history1.csv', headers,
                     reward_history_N, actor_loss_N, critic_loss_N)
        save_history('./Balibago_traci/output_DDPG/South_history1.csv', headers,
                     reward_history_S, actor_loss_S, critic_loss_S)
        print("All histories saved successfully!")

    print("Simulation Finished.")