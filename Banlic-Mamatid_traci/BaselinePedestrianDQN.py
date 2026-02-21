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

MainAgent = dqn(state_size=10, action_size=11, memory_size=MEMORY_SIZE, gamma=GAMMA, 
                 epsilon=1.0, epsilon_decay_rate=EPSILON_DECAY, epsilon_min=0.01, 
                 learning_rate=LEARNING_RATE, target_update_freq=500, 
                 name='Main_DQNAgent', area='Banlic-Mamatid')

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo', 
    '-c', 'Banlic-Mamatid_traci/baselinePed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Banlic-Mamatid_traci/output_DQN/BP_DQN_stats_trafficjam.xml',
    '--tripinfo-output', r'Banlic-Mamatid_traci/output_DQN/BP_DQN_trips_trafficjam.xml'
]

# --- SIMULATION VARIABLES ---
trainMode = 1
stepLength = 0.1

currentPhase = 0
currentPhaseDuration = 30

actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)

detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4", "e2_5"
]
detector_count = len(detector_ids)  

metric_observation_count = 0
throughput_total = 0
jam_length_total = 0

prevState = None
prevAction = None

step_counter = 0

# --- DATA LOGGING ---
reward_history = []
loss_history = []
epsilon_history = []
total_reward = 0

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
    if not vehicle_data:
        return 0

    for data in vehicle_data.values():
        v_type = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        
        weights = {
            "car": 1.0, "jeep": 1.5, "bus": 2.2, "truck": 2.5,
            "motorcycle": 0.3, "tricycle": 0.5
        }
        sumWait += waitTime * weights.get(v_type, 1.0)
    return sumWait

def _intersection_queue():
    return [_weighted_waits(det_id) for det_id in detector_ids]

def calculate_reward(current_state):
    if current_state is None:
        return 0
    return -sum(current_state)

def save_history(filename, headers, reward_hist, loss_hist, epsilon_hist, train_frequency):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        for i in range(len(reward_hist)):
            writer.writerow([i * train_frequency, reward_hist[i], loss_hist[i], epsilon_hist[i]])

# --- MAIN EXECUTION ---
traci.start(Sumo_config)
_subscribe_all_detectors()

# Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    currentPhaseDuration -= stepLength

    # 1. OBSERVATION PHASE
    obs = None
    reward = 0
    
    # Decision state: timer expired AND we are on a green (even) phase
    decision_needed = (currentPhaseDuration <= 0) and (currentPhase % 2 == 0)

    if decision_needed:
        queue = np.array(_intersection_queue())
        norm_queue = queue / 1000
        phase_oh = to_categorical(currentPhase // 2, num_classes=4).flatten()
        obs = np.concatenate([norm_queue, phase_oh]).astype(np.float32)

        if trainMode == 1:
            reward = calculate_reward(norm_queue * 10)
            total_reward += reward

    # 2. MEMORY PHASE
    if trainMode == 1:
        if decision_needed and prevState is not None:
            MainAgent.remember(prevState, prevAction, reward, obs, False)

    # 3. ACTION SELECTION PHASE
    next_action_idx = None

    if decision_needed:
        next_action_idx = MainAgent.act(obs)
        prevState = obs
        prevAction = next_action_idx
        if trainMode == 1:
            print(f"Main | Phase: {currentPhase} | Q: {np.sum(norm_queue):.2f} | R: {reward:.2f} | Act: {actionSpace[next_action_idx]}")
    elif currentPhaseDuration <= 0:
        # Yellow phase: carry over previous action index for phase transition
        next_action_idx = prevAction if prevAction is not None else 0

    # 4. EXECUTION PHASE (APPLY TO SIMULATION)
    if currentPhaseDuration <= 0:
        currentPhase = (currentPhase + 1) % 8
        # NOTE: Replace with your actual traffic light junction ID from .sumocfg
        traci.trafficlight.setPhase("253768576", currentPhase)
        traci.trafficlight.setPhase("253499548", currentPhase)

        if currentPhase % 2 == 1:
            currentPhaseDuration = 5
        else:
            # Green phase — adjust duration based on agent action
            duration_adj = actionSpace[next_action_idx]
            base = {0: 45, 2: 30, 4: 45, 6: 30}.get(currentPhase, 30)
            currentPhaseDuration = max(5, min(180, base + duration_adj))

        traci.trafficlight.setPhaseDuration("253768576", currentPhaseDuration)
        traci.trafficlight.setPhaseDuration("253499548", currentPhaseDuration)

    # 5. TRAINING REPLAY
    if trainMode == 1 and step_counter % TRAIN_FREQUENCY == 0 and step_counter > 1200 / stepLength:
        if len(MainAgent.memory) >= BATCH_SIZE:
            loss = MainAgent.replay(BATCH_SIZE)
            loss_history.append(loss)
            reward_history.append(total_reward)
            epsilon_history.append(MainAgent.epsilon)
            total_reward = 0
            MainAgent.epsilon = max(MainAgent.epsilon_min, MainAgent.epsilon * MainAgent.epsilon_decay_rate)

    # 6. METRICS (evaluation mode only)
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

# --- END OF SIMULATION ---
if metric_observation_count > 0:
    print("\n Queue Length Avg:", jam_length_total / metric_observation_count)
    print("\n Throughput Avg:", throughput_total / metric_observation_count)

if trainMode == 1:
    print("Saving trained model...")
    MainAgent.save()

    print("Saving training history...")
    headers = ['Step', 'Reward', 'Loss', 'Epsilon']
    save_history('./Banlic-Mamatid_traci/output_DQN/Main_history.csv', headers,
                 reward_history, loss_history, epsilon_history, TRAIN_FREQUENCY)
    print("All histories saved successfully!")

traci.close()