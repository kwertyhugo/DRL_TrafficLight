import os
import sys 
import traci
import numpy as np
import csv
from keras.utils import to_categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.DQN import DQNAgent as dqn

# --- CONFIGURATION ---
BATCH_SIZE = 32
TRAIN_FREQUENCY = 20  
MEMORY_SIZE = 2000
GAMMA = 0.95
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.00005

# --- AGENT INITIALIZATION ---
NorthAgent = dqn(state_size=12, action_size=11, memory_size=MEMORY_SIZE, gamma=GAMMA, 
                 epsilon=1.0, epsilon_decay_rate=EPSILON_DECAY, epsilon_min=0.01, 
                 learning_rate=LEARNING_RATE, target_update_freq=500, 
                 name='North_DQNAgent', area='Balibago')

SouthAgent = dqn(state_size=9, action_size=11, memory_size=MEMORY_SIZE, gamma=GAMMA, 
                 epsilon=1.0, epsilon_decay_rate=EPSILON_DECAY, epsilon_min=0.01, 
                 learning_rate=LEARNING_RATE, target_update_freq=500, 
                 name='South_DQNAgent', area='Balibago')

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo', 
    '-c', 'Balibago_traci/baselinePed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Balibago_traci/output_DQN/BP_DQN_stats_TRAINING.xml',
    '--tripinfo-output', r'Balibago_traci/output_DQN/BP_DQN_trips_TRAINING.xml'
]

# --- SIMULATION VARIABLES ---
trainMode = 1
stepLength = 0.1

northCurrentPhase = 0
northCurrentPhaseDuration = 30
southCurrentPhase = 0
southCurrentPhaseDuration = 30

actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)

detector_ids = [
        "e2_0", "e2_1", "e2_2", "e2_3", "e2_4", "e2_5", "e2_6", "e2_7", 
        "e2_8", "e2_9", "e2_10", "e2_11", "e2_12"
    ]
detector_count = len(detector_ids)

metric_observation_count = 0
throughput_total = 0
jam_length_total = 0

northPrevState = None
northPrevAction = None
southPrevState = None
southPrevAction = None

step_counter = 0

MAX_STEPS = 567000

# --- DATA LOGGING ---
reward_history_N = []
step_history_N = []
loss_history_N = []
epsilon_history_N = []
training_steps_N = []

reward_history_S = []
step_history_S = []
loss_history_S = []
epsilon_history_S = []
training_steps_S = []

# --- HELPER FUNCTIONS ---
def _subscribe_all_detectors():
    vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_context_vars)
        traci.lanearea.subscribe(det_id, vehicle_vars)

def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
    if not vehicle_data: return 0

    for data in vehicle_data.values():
        v_type = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        
        weights = {
            "car": 1.0, "jeep": 1.5, "bus": 2.2, "truck": 2.5,
            "motorcycle": 0.3, "tricycle": 0.5
        }
        sumWait += waitTime * weights.get(v_type, 1.0)
    return sumWait

def _northIntersection_queue():
    queues = [_weighted_waits(f"e2_{i}") for i in range(8)]
    return queues

def _southIntersection_queue():
    queues = [_weighted_waits(f"e2_{i}") for i in range(8, 13)]
    return queues

def calculate_reward(current_state):
    if current_state is None: return 0
    return -sum(current_state)

def get_next_phase_duration(current_phase, action_index):
    # Determine next duration without setting it yet
    if current_phase % 2 == 1: 
        return 5 # Yellow/Transition
    else: 
        duration_adjustment = actionSpace[action_index]
        base_durations = {0: 45, 2: 130, 4: 30, 6: 90} if current_phase in [0,2,4,6] else {0: 30, 2: 30, 4: 45}
        # Note: Logic slightly differs per intersection in original code, handled in APPLY block below
        return 0 # Placeholder, logic moved to apply function for clarity

def save_history(filename, headers, step_hist, reward_hist, training_steps, loss_hist, epsilon_hist):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        
        for i, reward in enumerate(reward_hist):
            reward_step = step_hist[i]
            
            # Find the most recent training step at or before this reward step
            loss_val = None
            epsilon_val = None
            for j in range(len(training_steps) - 1, -1, -1):
                if training_steps[j] <= reward_step:
                    loss_val = loss_hist[j]
                    epsilon_val = epsilon_hist[j]
                    break
            
            writer.writerow([reward_step, reward, loss_val, epsilon_val])

# --- MAIN EXECUTION ---
traci.start(Sumo_config, port=8813)
_subscribe_all_detectors()

while traci.simulation.getMinExpectedNumber() > 0 and step_counter < MAX_STEPS:
    step_counter += 1
    northCurrentPhaseDuration -= stepLength
    southCurrentPhaseDuration -= stepLength

    # 1. OBSERVATION PHASE (IMMUTABLE)
    # --------------------------------
    obs_north = None
    obs_south = None
    reward_north = 0
    reward_south = 0
    
    # Flags to determine if we are in a decision state (Timer expired & Green Phase)
    north_decision_needed = (northCurrentPhaseDuration <= 0) and (northCurrentPhase % 2 == 0)
    south_decision_needed = (southCurrentPhaseDuration <= 0) and (southCurrentPhase % 2 == 0)
    
    # Capture North State if needed
    if north_decision_needed:
        queue = np.array(_northIntersection_queue())
        n_norm_queue = queue/1000
        n_phase_oh = to_categorical(northCurrentPhase//2, num_classes=4).flatten()
        obs_north = np.concatenate([n_norm_queue, n_phase_oh]).astype(np.float32)
        
        if trainMode == 1:
            reward_north = calculate_reward(n_norm_queue*10)
            reward_history_N.append(reward_north)
            step_history_N.append(step_counter)

    # Capture South State if needed
    if south_decision_needed:
        queue = np.array(_southIntersection_queue())
        s_norm_queue = queue/1000
        s_phase_oh = to_categorical(southCurrentPhase//2, num_classes=4).flatten()
        obs_south = np.concatenate([s_norm_queue, s_phase_oh]).astype(np.float32)

        if trainMode == 1:
            reward_south = calculate_reward(s_norm_queue*10)
            reward_history_S.append(reward_south)
            step_history_S.append(step_counter)

    # 2. MEMORY PHASE
    # ---------------
    if trainMode == 1:
        if north_decision_needed and northPrevState is not None:
            NorthAgent.remember(northPrevState, northPrevAction, reward_north, obs_north, False)
        
        if south_decision_needed and southPrevState is not None:
            SouthAgent.remember(southPrevState, southPrevAction, reward_south, obs_south, False)

    # 3. ACTION SELECTION PHASE
    # -------------------------
    next_action_N_idx = None
    next_action_S_idx = None

    if north_decision_needed:
        next_action_N_idx = NorthAgent.act(obs_north)
        # Store for next cycle
        northPrevState = obs_north
        northPrevAction = next_action_N_idx
        if trainMode == 1:
            print(f"North | Q: {np.sum(obs_north[:9]):.2f} | R: {reward_north:.2f} | Act: {actionSpace[next_action_N_idx]}")
    elif northCurrentPhaseDuration <= 0:
        # Yellow phase logic: simply carry over previous action index (not used for calculation but for consistency)
        next_action_N_idx = northPrevAction if northPrevAction is not None else 0

    if south_decision_needed:
        next_action_S_idx = SouthAgent.act(obs_south)
        # Store for next cycle
        southPrevState = obs_south
        southPrevAction = next_action_S_idx
        if trainMode == 1:
            print(f"South | Q: {np.sum(obs_south[:6]):.2f} | R: {reward_south:.2f} | Act: {actionSpace[next_action_S_idx]}")
    elif southCurrentPhaseDuration <= 0:
        next_action_S_idx = southPrevAction if southPrevAction is not None else 0

    # 4. EXECUTION PHASE (APPLY TO SIMULATION)
    # ----------------------------------------
    
    # Apply North
    if northCurrentPhaseDuration <= 0:
        northCurrentPhase = (northCurrentPhase + 1) % 6
        traci.trafficlight.setPhase("4902876117", northCurrentPhase)

        if northCurrentPhase % 2 == 1:
            northCurrentPhaseDuration = 5
        else:
            # Determine Duration
            idx = next_action_N_idx
            duration_adj = actionSpace[idx]
            base = {0: 45, 2: 130, 4: 30, 6: 90}.get(northCurrentPhase, 30)
            northCurrentPhaseDuration = max(5, min(180, base + duration_adj))
        
        traci.trafficlight.setPhaseDuration("4902876117", northCurrentPhaseDuration)

    # Apply South
    if southCurrentPhaseDuration <= 0:
        southCurrentPhase = (southCurrentPhase + 1) % 6
        traci.trafficlight.setPhase("12188714", southCurrentPhase)

        if southCurrentPhase % 2 == 1:
            southCurrentPhaseDuration = 5
        else:
            idx = next_action_S_idx
            duration_adj = actionSpace[idx]
            base = {0: 30, 2: 30, 4: 45}.get(southCurrentPhase, 30)
            southCurrentPhaseDuration = max(5, min(180, base + duration_adj))
        
        traci.trafficlight.setPhaseDuration("12188714", southCurrentPhaseDuration)


    # --- TRAINING REPLAY ---
    if trainMode == 1 and step_counter % TRAIN_FREQUENCY == 0 and step_counter > 1200/stepLength:
        if len(NorthAgent.memory) >= BATCH_SIZE:
            loss_n = NorthAgent.replay(BATCH_SIZE)
            loss_history_N.append(loss_n)
            epsilon_history_N.append(NorthAgent.epsilon)
            training_steps_N.append(step_counter)
            NorthAgent.epsilon = max(NorthAgent.epsilon_min, NorthAgent.epsilon * NorthAgent.epsilon_decay_rate)
        
        if len(SouthAgent.memory) >= BATCH_SIZE:
            loss_s = SouthAgent.replay(BATCH_SIZE)
            loss_history_S.append(loss_s)
            epsilon_history_S.append(SouthAgent.epsilon)
            training_steps_S.append(step_counter)
            SouthAgent.epsilon = max(SouthAgent.epsilon_min, SouthAgent.epsilon * SouthAgent.epsilon_decay_rate)
    
    # --- METRICS ---
    TRACK_INTERVAL_STEPS = int(60 / stepLength)
    if trainMode == 0 and step_counter % TRACK_INTERVAL_STEPS == 0 :
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

# --- END OF SIMULATION ---
if metric_observation_count > 0:
    print("\n Queue Length Avg:", jam_length_total / metric_observation_count)
    print("\n Throughput Avg:", throughput_total / metric_observation_count)

if trainMode == 1:
    print("Saving trained models...")
    NorthAgent.save()
    SouthAgent.save() 

    print("Saving training history...")
    headers = ['Step', 'Reward', 'Loss', 'Epsilon']
    
    save_history('./Balibago_traci/output_DQN/BP_North_history.csv', headers, 
                step_history_N, reward_history_N, training_steps_N, loss_history_N, epsilon_history_N)
    
    save_history('./Balibago_traci/output_DQN/BP_South_history.csv', headers, 
                step_history_S, reward_history_S, training_steps_S, loss_history_S, epsilon_history_S)
    
    print("All histories saved successfully!")

traci.close()