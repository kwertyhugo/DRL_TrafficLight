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

os.makedirs('./Banlic-Mamatid_traci/output_DDPG', exist_ok=True)

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

# Action bounds: 5 outputs, each -1 to 1, scaled to ±25s
ACTION_SIZE = 5
action_low  = np.full(ACTION_SIZE, -1.0, dtype=np.float32)
action_high = np.full(ACTION_SIZE,  1.0, dtype=np.float32)
ACTION_SCALE = 25.0

# Green phase index → action vector index
# Banlic-Mamatid green phases: 0, 2, 4, 6, 8  →  action indices 0, 1, 2, 3, 4
GREEN_PHASES = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4}

# Base green durations (seconds)
BASE_DURATIONS = {0: 30, 2: 30, 4: 45, 6: 60, 8: 25}

# ==========================================
# AGENT INITIALIZATION
# ==========================================
# State: 6 detectors + 5 phase one-hot = 11
MainAgent = ddpg(
    state_size=11,
    action_size=ACTION_SIZE,
    action_low=action_low,
    action_high=action_high,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    name='Main_DDPGAgent'
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
    '-c', 'Banlic-Mamatid_traci/signalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Banlic-Mamatid_traci/output_DDPG/training.xml',
    '--tripinfo-output', r'Banlic-Mamatid_traci/output_DDPG/training.xml'
]

# ==========================================
# SIMULATION VARIABLES
# ==========================================
trainMode = 1
stepLength = 0.1

currentPhase = 0
currentPhaseDuration = 30

detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4", "e2_5"
]
detector_count = len(detector_ids)

metric_observation_count = 0
throughput_total = 0
jam_length_total = 0

prevState  = None
prevAction = None

step_counter = 0
MAX_STEPS = 150000

# History Buffers
reward_history   = []
actor_loss_history  = []
critic_loss_history = []
total_reward = 0

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def _subscribe_all_detectors():
    vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_context_vars)
        traci.lanearea.subscribe(det_id, vehicle_vars)

def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id,
        traci.constants.CMD_GET_PERSON_VARIABLE,
        20.0,
        [traci.constants.VAR_WAITING_TIME]
    )

def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
    if not vehicle_data:
        return 0
    weights = {"car": 1.0, "jeep": 1.5, "bus": 2.2, "truck": 2.5,
               "motorcycle": 0.3, "tricycle": 0.5}
    for data in vehicle_data.values():
        v_type   = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        sumWait += waitTime * weights.get(v_type, 1.0)
    return sumWait

def _intersection_queue():
    return [_weighted_waits(det_id) for det_id in detector_ids]

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
model_path = './Banlic-Mamatid_traci/output_DDPG/'

if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
    print("\n" + "=" * 60)
    print("Found saved history. Attempting to load...")
    try:
        MainAgent.load()
        print(" - Model weights loaded successfully.")
    except Exception as e:
        print(f" - Warning: Could not load model weights. Starting fresh. Error: {e}")

    if trainMode == 1:
        try:
            MainAgent.load_replay_buffer()
            print(" - Replay buffer loaded.")
        except Exception as e:
            print(f" - Warning: Could not load replay buffer. Error: {e}")
    print("=" * 60)
else:
    print("No saved history found. Starting fresh.")

# ==========================================
# MAIN EXECUTION
# ==========================================
traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("253768576")

print(f"STARTING DDPG | Banlic-Mamatid | action_size={ACTION_SIZE} | trainMode={trainMode}")

try:
    while traci.simulation.getMinExpectedNumber() > 0 and step_counter < MAX_STEPS:
        step_counter += 1
        currentPhaseDuration -= stepLength

        # -------------------------------------------------
        # 1. OBSERVATION PHASE
        # -------------------------------------------------
        obs    = None
        reward = 0

        decision_needed = (currentPhaseDuration <= 0) and (currentPhase % 2 == 0)

        if decision_needed:
            queue = np.array(_intersection_queue())
            norm_queue = queue / 1000.0
            phase_oh = to_categorical(currentPhase // 2, num_classes=5).flatten()
            obs = np.concatenate([norm_queue, phase_oh]).astype(np.float32)

            if trainMode == 1:
                reward = calculate_reward(norm_queue * 10)
                total_reward += reward

        # -------------------------------------------------
        # 2. MEMORY & TRAINING PHASE
        # -------------------------------------------------
        if trainMode == 1:
            if decision_needed and prevState is not None:
                MainAgent.remember(prevState, prevAction, reward, obs, False)

                if len(MainAgent.replay_buffer) >= BATCH_SIZE:
                    a_loss, c_loss = MainAgent.train()
                    if a_loss is not None:
                        actor_loss_history.append(a_loss)
                        critic_loss_history.append(c_loss)
                        reward_history.append(total_reward)
                        total_reward = 0

        # -------------------------------------------------
        # 3. ACTION SELECTION PHASE
        # -------------------------------------------------
        action_vec = None

        if decision_needed:
            # Get full 5-action vector; only the relevant index is applied this step
            action_vec = MainAgent.get_action(obs, add_noise=(trainMode == 1))
            prevState  = obs
            prevAction = action_vec

            if trainMode == 1:
                act_idx = GREEN_PHASES.get(currentPhase, 0)
                print(f"Main | Phase: {currentPhase} | Q: {np.sum(norm_queue):.2f} | "
                      f"R: {reward:.2f} | Act[{act_idx}]: {action_vec[act_idx]:.3f}")

        elif currentPhaseDuration <= 0:
            # Yellow phase: carry over previous action vector
            action_vec = prevAction if prevAction is not None else np.zeros(ACTION_SIZE)

        # -------------------------------------------------
        # 4. EXECUTION PHASE
        # -------------------------------------------------
        if currentPhaseDuration <= 0:
            currentPhase = (currentPhase + 1) % 10
            traci.trafficlight.setPhase("253768576", currentPhase)
            traci.trafficlight.setPhase("253499548", currentPhase)

            if currentPhase % 2 == 1:
                currentPhaseDuration = 5  # Yellow
            else:
                # Use the action output that corresponds to this specific green phase
                act_idx = GREEN_PHASES.get(currentPhase, 0)
                base = BASE_DURATIONS.get(currentPhase, 30)
                adjustment = float(action_vec[act_idx]) * ACTION_SCALE
                currentPhaseDuration = max(5.0, min(180.0, base + adjustment))

            traci.trafficlight.setPhaseDuration("253768576", float(currentPhaseDuration))
            traci.trafficlight.setPhaseDuration("253499548", float(currentPhaseDuration))

        # -------------------------------------------------
        # 5. METRICS (evaluation mode only)
        # -------------------------------------------------
        TRACK_INTERVAL_STEPS = int(60 / stepLength)
        if trainMode == 0 and step_counter % TRACK_INTERVAL_STEPS == 0:
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
        print("Saving trained model...")
        MainAgent.save()

        print("Saving training history...")
        headers = ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss']
        save_history('./Banlic-Mamatid_traci/output_DDPG/Main_history.csv', headers,
                     reward_history, actor_loss_history, critic_loss_history)
        print("All histories saved successfully!")

    print("Simulation Finished.")