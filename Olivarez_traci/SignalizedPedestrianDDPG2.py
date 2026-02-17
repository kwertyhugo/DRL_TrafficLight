import os
import sys

# Path setup to find models folder (assumes models folder is in parent directory)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create output directories if they don't exist
os.makedirs('./Olivarez_traci/output_DDPG', exist_ok=True)

import traci
import numpy as np
import csv
from keras.utils import to_categorical
from models.DDPG import DDPGAgent as ddpg

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_MODE = 0          # 1 = Train, 0 = Test
reward_scale = 10.0     
noise_decay = 0.9995    
min_noise_std = 0.01    

# State & Action definitions
STATE_SIZE = 19
ACTION_SIZE = 3
action_low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

# Normalization tracking
state_mean = np.zeros(STATE_SIZE, dtype=np.float32)
state_std = np.ones(STATE_SIZE, dtype=np.float32)

# ==========================================
# AGENT INITIALIZATION
# ==========================================
trafficLightAgent = ddpg(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    action_low=action_low,
    action_high=action_high,
    actor_lr=0.0001,
    critic_lr=0.001,
    name='New_Global_Olivarez_DDPG'
)

# ==========================================
# LOAD EXISTING MODEL
# ==========================================
model_path = './Olivarez_traci/output_DDPG/'

if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
    print("\n" + "=" * 70)
    print("Found saved history. Attempting to load...")
    try:
        trafficLightAgent.load() 
        print("Model weights loaded successfully.")
        if TRAIN_MODE == 1:
            trafficLightAgent.load_replay_buffer()
            print("Replay buffer (memory) loaded.")
    except Exception as e:
        print(f"Warning: Could not load model/memory. Error: {e}")
        print("Starting with fresh agent.")
    print("=" * 70)
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
    'sumo',
    '-c', 'Olivarez_traci/signalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_DDPG\Global_SD_DDPG_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_DDPG\Global_SD_DDPG_trips.xml'
]

# Simulation Variables
currentPhase = 0
currentPhaseDuration = 30.0
stepLength = 0.1
step_counter = 0

# Metrics History
reward_history = []
actor_loss_history = []
critic_loss_history = []
cumulative_reward = 0
TRAIN_FREQUENCY = 100 
BATCH_SIZE = 64

# --- METRICS VARIABLES (MATCHING YOUR REQUEST) ---
throughput_total = 0
jam_length_total = 0
metric_observation_count = 0
# -------------------------------------------------

# IDs
detector_ids = [
    "e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10",
    "e2_0", "e2_1", "e2_2", "e2_3"
]
detector_count = len(detector_ids)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize_state(state, mean, std_var):
    return (state - mean) / (np.sqrt(std_var) + 1e-8)

def _weighted_waits(detector_id):
    sumWait = 0
    try:
        vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
        if not vehicle_data: return 0
        weights = {"car": 1.0, "jeep": 1.5, "bus": 2.2, "truck": 2.5, "motorcycle": 0.3, "tricycle": 0.5}
        for data in vehicle_data.values():
            vtype = data.get(traci.constants.VAR_TYPE, "car")
            waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
            sumWait += waitTime * weights.get(vtype, 1.0)
    except: return 0
    return sumWait

def get_unified_state(phase_idx):
    # 1. Vehicle Detectors
    queues = []
    # Crossing approaches
    for det in ["e2_0", "e2_1", "e2_2", "e2_3"]:
        queues.append(_weighted_waits(det))
    # Main intersection approaches
    for det in ["e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10"]:
        queues.append(_weighted_waits(det))

    # 2. Pedestrians
    pedM = 0
    subM = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
    if subM: 
        for pid, data in subM.items(): pedM += data.get(traci.constants.VAR_WAITING_TIME, 0)

    pedW = 0
    subW = traci.junction.getContextSubscriptionResults("6401523012")
    if subW: 
        for pid, data in subW.items(): pedW += data.get(traci.constants.VAR_WAITING_TIME, 0)

    pedE = 0
    subE = traci.junction.getContextSubscriptionResults("3285696417")
    if subE: 
        for pid, data in subE.items(): pedE += data.get(traci.constants.VAR_WAITING_TIME, 0)
    
    queue_state = queues + [pedM*1.25, pedW*1.25, pedE*1.25]
    phase_encoding = to_categorical(phase_idx // 2, num_classes=5).flatten()
    full_state = np.concatenate([queue_state, phase_encoding])
    return np.array(full_state, dtype=np.float32), np.array(queue_state)

def calculate_reward(unnormalized_queues):
    if unnormalized_queues is None: return 0.0
    total_wait = float(np.sum(unnormalized_queues))
    return -total_wait

def _apply_unified_phase(action_vals):
    global currentPhase, currentPhaseDuration
    currentPhase = (currentPhase + 1) % 10
    
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", currentPhase)
    traci.trafficlight.setPhase("6401523012", currentPhase)
    traci.trafficlight.setPhase("3285696417", currentPhase)

    if currentPhase % 2 == 1:
        # Yellow
        durations = [5.0, 5.0, 5.0]
        currentPhaseDuration = 5.0
    else:
        # Green
        base_duration = 20.0 if currentPhase == 4 else 30.0
        durations = []
        for val in action_vals:
            adj = float(np.clip(val, -1.0, 1.0) * 25.0)
            dur = float(max(5.0, min(60.0, base_duration + adj)))
            durations.append(dur)
        currentPhaseDuration = durations[0]
    
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", durations[0])
    traci.trafficlight.setPhaseDuration("6401523012", durations[1])
    traci.trafficlight.setPhaseDuration("3285696417", durations[2])

def save_history(filename, headers, reward_hist, actor_loss_hist, critic_loss_hist, frequency):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        for i in range(len(reward_hist)):
            writer.writerow([i * frequency, reward_hist[i], actor_loss_hist[i], critic_loss_hist[i]])

# ==========================================
# MAIN EXECUTION
# ==========================================

traci.start(Sumo_config)

# Subscriptions
for junc in ["cluster_295373794_3477931123_7465167861", "6401523012", "3285696417"]:
    traci.junction.subscribeContext(junc, traci.constants.CMD_GET_PERSON_VARIABLE, 10.0, [traci.constants.VAR_WAITING_TIME])

vehicle_context = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
# Metrics subscriptions
vehicle_metric_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]

for det in detector_ids:
    traci.lanearea.subscribeContext(det, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_context)
    traci.lanearea.subscribe(det, vehicle_metric_vars) 

prev_state = None
prev_action = None

print("=" * 70)
print("Starting UNIFIED DDPG Training (3 Actions)")
print("=" * 70)

while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    currentPhaseDuration -= stepLength
    
    # DECISION POINT
    if currentPhaseDuration <= 0:
        raw_full_state, raw_queues = get_unified_state(currentPhase)
        
        state_mean = 0.99 * state_mean + 0.01 * raw_full_state
        state_std = 0.99 * state_std + 0.01 * np.square(raw_full_state - state_mean)
        current_state = normalize_state(raw_full_state, state_mean, state_std)

        reward = calculate_reward(raw_queues) / reward_scale 
        cumulative_reward += reward

        # Train & Remember
        if TRAIN_MODE == 1 and prev_state is not None:
            trafficLightAgent.remember(prev_state, prev_action, reward, current_state, False)
            
            if len(trafficLightAgent.replay_buffer) >= BATCH_SIZE:
                a_loss, c_loss = trafficLightAgent.train()
                if a_loss:
                    actor_loss_history.append(a_loss)
                    critic_loss_history.append(c_loss)
                    reward_history.append(cumulative_reward)
                    cumulative_reward = 0

        # Action
        if TRAIN_MODE == 1:
            trafficLightAgent.noise.std_dev = max(min_noise_std, trafficLightAgent.noise.std_dev * noise_decay)
        
        action = trafficLightAgent.get_action(current_state, add_noise=(TRAIN_MODE == 1))
        
        if currentPhase % 2 == 0:
            actual_action = action
        else:
            actual_action = prev_action if prev_action is not None else np.zeros(ACTION_SIZE)

        _apply_unified_phase(actual_action)

        prev_state = current_state
        prev_action = actual_action

        if step_counter % 100 == 0 and TRAIN_MODE == 1:
            print(f"Step {step_counter} | Reward: {reward:.2f}")

    # --- UPDATED METRICS COLLECTION (EXACT MATCH TO REQUEST) ---
    TRACK_INTERVAL_STEPS = int(60 / stepLength)
    if TRAIN_MODE == 0 and step_counter % TRACK_INTERVAL_STEPS == 0:
        jam_length = 0
        throughput = 0
        metric_observation_count += 1
        
        for det_id in detector_ids:
            detector_stats = traci.lanearea.getSubscriptionResults(det_id)
            if not detector_stats: continue
            
            jam_length += detector_stats.get(traci.constants.JAM_LENGTH_METERS, 0)
            throughput += detector_stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)
        
        jam_length /= detector_count
        jam_length_total += jam_length
        throughput_total += throughput

    traci.simulationStep()

# ==========================================
# CLEANUP & SAVE
# ==========================================
traci.close()

# --- FINAL METRICS OUTPUT (EXACT MATCH) ---
if metric_observation_count > 0:
    print("\n Queue Length Avg:", jam_length_total / metric_observation_count)
    print("\n Throughput Avg:", throughput_total / metric_observation_count)

if TRAIN_MODE == 1:
    print("\nSaving trained model...")
    trafficLightAgent.save()
    trafficLightAgent.save_replay_buffer()
    
    save_history('./Olivarez_traci/output_DDPG/global_agent_history.csv', 
                 ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss'],
                 reward_history, actor_loss_history, critic_loss_history, TRAIN_FREQUENCY)
    print("History saved successfully!")

print("Simulation Done!")