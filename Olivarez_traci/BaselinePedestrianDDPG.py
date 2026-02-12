import os
import sys
import traci
import numpy as np
import csv
from keras.utils import to_categorical
from models.DDPG import DDPGAgent as ddpg

# ------- Configuration -------
TRAIN_MODE = 1
reward_scale = 10.0  
noise_decay = 0.9995 
min_noise_std = 0.01

# State: 7 detectors + 1 ped + 5 phase bits = 13
STATE_SIZE = 13
ACTION_SIZE = 1

# Action bounds (DDPG outputs -1 to 1)
action_low = np.array([-1.0], dtype=np.float32)
action_high = np.array([1.0], dtype=np.float32)

detector_ids = ["e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10"]
detector_count = len(detector_ids)

# Initialize Agent
mainAgent = ddpg(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    action_low=action_low,
    action_high=action_high,
    actor_lr=0.0001,
    critic_lr=0.001,
    name='Baseline_Main_Intersection_DDPG'
)

# ------- Load History / Weights -------
model_path = './Olivarez_traci/output_DDPG/'

if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
    print("Found saved history. Attempting to load...")
    try:
        mainAgent.load() 
        print(" - Model weights loaded successfully.")
    except Exception as e:
        print(f" - Warning: Could not load model weights. Starting random. Error: {e}")

    try:
        # Load memory only if training to resume learning
        if TRAIN_MODE == 1:
            mainAgent.load_replay_buffer()
            print(" - Replay buffer loaded.")
    except Exception as e:
        print(f" - Warning: Could not load replay buffer. Error: {e}")
else:
    print("No history found. Starting fresh.")

# ------- Simulation Variables -------
mainCurrentPhase = 0
mainCurrentPhaseDuration = 30.0
stepLength = 0.05
step_counter = 0

# Metrics
reward_history = []
actor_loss_history = []
critic_loss_history = []
cumulative_reward = 0
TRAIN_FREQUENCY = 100 
BATCH_SIZE = 64

# Statistical Metrics (Same as DQN)
throughput_average = 0
throughput_total = 0
jam_length_average = 0
jam_length_total = 0
metric_observation_count = 0

# Normalization tracking
state_mean = np.zeros(STATE_SIZE, dtype=np.float32)
state_std = np.ones(STATE_SIZE, dtype=np.float32)

def normalize_state(state, mean, std_var):
    return (state - mean) / (np.sqrt(std_var) + 1e-8)

def calculate_reward(unnormalized_queue_state):
    # Only calculate reward based on the queue/wait times (first 8 elements)
    if unnormalized_queue_state is None: return 0.0
    total_wait = float(np.sum(unnormalized_queue_state))
    return -total_wait

def save_history(filename, headers, reward_hist, actor_loss_hist, critic_loss_hist, train_frequency):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    existing_rows = 0
    if file_exists:
        with open(filename, 'r') as f:
            existing_rows = sum(1 for _ in f) - 1
    
    if existing_rows < 0: existing_rows = 0

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists or existing_rows == 0:
            writer.writerow(headers)

        start_index = existing_rows
        for i in range(start_index, len(reward_hist)):
            writer.writerow([i * train_frequency, reward_hist[i], actor_loss_hist[i], critic_loss_hist[i]])

# ------- Helper Functions -------

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

def get_intersection_state(phase_idx):
    # 1. Get Queues/Waits
    q_data = []
    # Detectors e2_4 to e2_10
    for det in detector_ids:
        q_data.append(_weighted_waits(det))
    
    # Pedestrian
    ped_wait = 0
    sub_m = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
    if sub_m: 
        for pid, data in sub_m.items(): ped_wait += data.get(traci.constants.VAR_WAITING_TIME, 0)
    q_data.append(ped_wait)

    # 2. Get Phase One-Hot (Phase 0-9 -> encoded as 5 groups of 2)
    # Assuming phase structure matches DQN: 0,2,4,6,8 are Green phases
    phase_encoding = to_categorical(phase_idx // 2, num_classes=5).flatten()
    
    queue_array = np.array(q_data, dtype=np.float32)
    
    # Combine Queues + Phase
    full_state = np.concatenate([queue_array, phase_encoding])
    
    return full_state, queue_array

def _apply_phase(action_val):
    global mainCurrentPhase, mainCurrentPhaseDuration
    
    mainCurrentPhase = (mainCurrentPhase + 1) % 10
    
    # Map [-1, 1] to [-20s, +20s] roughly matching DQN range
    duration_adjustment = float(np.clip(action_val, -1.0, 1.0) * 20.0)
    
    if mainCurrentPhase in [2, 4]:
        base_duration = 15.0
    elif mainCurrentPhase % 2 == 0:
        base_duration = 30.0
    else:
        # Yellow phases (odd numbers) usually fixed
        base_duration = 3.0
        duration_adjustment = 0 # Do not adjust yellow time

    mainCurrentPhaseDuration = float(max(5.0, min(60.0, base_duration + duration_adjustment)))
    
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", mainCurrentPhase)
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", mainCurrentPhaseDuration)

# ------- Main Execution -------

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path: sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci 

Sumo_config = [
    'sumo',
    '-c', r'Olivarez_traci\baselinePed.sumocfg',
    '--step-length', '0.05',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_DDPG\baseline_SD_DDPG_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_DDPG\baseline_SD_DDPG_trips.xml'
]

traci.start(Sumo_config)

# Subscriptions
traci.junction.subscribeContext("cluster_295373794_3477931123_7465167861", 
                                traci.constants.CMD_GET_PERSON_VARIABLE, 10.0, [traci.constants.VAR_WAITING_TIME])

vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]

for det in detector_ids:
    traci.lanearea.subscribeContext(det, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_context_vars)
    traci.lanearea.subscribe(det, vehicle_vars) # For throughput/jam metrics

prev_state = None
prev_action = None

while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    mainCurrentPhaseDuration -= stepLength

    # Decision Point
    if mainCurrentPhaseDuration <= 0:
        raw_full_state, raw_queues = get_intersection_state(mainCurrentPhase)
        
        # Normalize (Dynamic running mean/std)
        state_mean = 0.99 * state_mean + 0.01 * raw_full_state
        state_std = 0.99 * state_std + 0.01 * np.square(raw_full_state - state_mean)
        current_state = normalize_state(raw_full_state, state_mean, state_std)

        # Reward
        reward = calculate_reward(raw_queues) / reward_scale
        cumulative_reward += reward

        # Train / Remember
        if TRAIN_MODE == 1 and prev_state is not None:
            mainAgent.remember(prev_state, prev_action, reward, current_state, False)
            
            if len(mainAgent.replay_buffer) >= BATCH_SIZE:
                a_loss, c_loss = mainAgent.train()
                if a_loss:
                    actor_loss_history.append(a_loss)
                    critic_loss_history.append(c_loss)
                    reward_history.append(cumulative_reward)
                    cumulative_reward = 0

        # Action (Noise only if training)
        if TRAIN_MODE == 1:
            mainAgent.noise.std_dev = max(min_noise_std, mainAgent.noise.std_dev * noise_decay)
        
        action = mainAgent.get_action(current_state, add_noise=(TRAIN_MODE == 1))
        
        # Apply Logic
        if mainCurrentPhase % 2 == 0:
            # Green Phase: Agent decides duration
            actual_action = action # Keep as array
        else:
            # Yellow Phase: Dummy action, fixed time
            actual_action = np.zeros(ACTION_SIZE) 

        _apply_phase(actual_action[0])
        
        prev_state = current_state
        prev_action = actual_action
        
        if step_counter % 100 == 0:
             print(f"Step {step_counter} | Reward: {reward:.2f} | Action: {actual_action[0]:.2f}")

    # Metrics Collection (Same as DQN)
    TRACK_INTERVAL_STEPS = int(6 / stepLength)
    if TRAIN_MODE == 0 and step_counter % TRACK_INTERVAL_STEPS == 0:
        jam_length = 0
        throughput = 0
        metric_observation_count += 1
        
        for det_id in detector_ids:
            stats = traci.lanearea.getSubscriptionResults(det_id)
            if stats:
                jam_length += stats.get(traci.constants.JAM_LENGTH_METERS, 0)
                throughput += stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)
        
        jam_length /= detector_count
        jam_length_total += jam_length
        throughput_total += throughput

    traci.simulationStep()

# ------- Wrap Up -------
traci.close()

if metric_observation_count > 0:
    print("\n --- Final Metrics ---")
    print(f" Avg Queue Length: {jam_length_total / metric_observation_count:.2f}")
    print(f" Avg Throughput: {throughput_total / metric_observation_count:.2f}")

if TRAIN_MODE == 1:
    mainAgent.save()
    mainAgent.save_replay_buffer()
    save_history('./Olivarez_traci/output_DDPG/baseline_main_agent_history.csv', 
                 ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss'],
                 reward_history, actor_loss_history, critic_loss_history, TRAIN_FREQUENCY)

print("Simulation Done!")