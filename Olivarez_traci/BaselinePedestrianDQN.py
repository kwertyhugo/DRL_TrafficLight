import os
import sys 
import traci
import numpy as np
import csv
from keras.utils import to_categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.DQN import DQNAgent as dqn

mainIntersectionAgent = dqn(state_size=13, action_size=11, memory_size=2000, gamma=0.95, epsilon=1.0, epsilon_decay_rate=0.995, epsilon_min=0, learning_rate=0.00005, target_update_freq=100, name='ReLU_DQNAgent_Baseline')

mainIntersectionAgent.load()

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo',
    '-c', r'Olivarez_traci\baselinePed.sumocfg',
    '--step-length', '0.05',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_DQN\BP_DQN_stats_trafficjamNEW.xml',
    '--tripinfo-output', r'Olivarez_traci\output_DQN\BP_DQN_trips_trafficjamNEW.xml'
]

# Simulation Variables
trainMode = 0
stepLength = 0.05
mainCurrentPhase = 0
mainCurrentPhaseDuration = 30
actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)
detector_count = 7

# Store previous states and actions for learning
mainPrevState = None
mainPrevAction = None

# Batch training parameters
BATCH_SIZE = 32
TRAIN_FREQUENCY = 100  # Train every 100 steps / 5 seconds
step_counter = 0

#Metrics
throughput_average = 0
throughput_total = 0
jam_length_average = 0
jam_length_total = 0
metric_observation_count = 0

# -- Data storage for plotting --
main_reward_history = []
main_loss_history = []
main_epsilon_history = []
total_main_reward = 0

#Object Context Subscription in SUMO
detector_ids = ["e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10"]

def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id,
        traci.constants.CMD_GET_PERSON_VARIABLE,
        10.0,
        [traci.constants.VAR_WAITING_TIME]
    )
    
def _subscribe_all_detectors():
    global detector_ids 
    
    vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(
            det_id,
            traci.constants.CMD_GET_VEHICLE_VARIABLE,
            3,
            vehicle_context_vars
        )
    
    for det_id in detector_ids:
        traci.lanearea.subscribe(
            det_id,
            vehicle_vars
        )

# Inputs to the Model
def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)

    if not vehicle_data:
        return 0

    for data in vehicle_data.values():
        type = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        
        if type == "car":
            sumWait += waitTime
        elif type == "jeep":
            sumWait += waitTime * 1.5
        elif type == "bus":
            sumWait += waitTime * 2.2
        elif type == "truck":
            sumWait += waitTime * 2.5
        elif type == "motorcycle":
            sumWait += waitTime * 0.3
        elif type == "tricycle":
            sumWait += waitTime * 0.5
    return sumWait

def _mainIntersection_queue():
    e2_4 = _weighted_waits("e2_4") 
    e2_5 = _weighted_waits("e2_5")
    e2_6 = _weighted_waits("e2_6")
    e2_7 = _weighted_waits("e2_7")
    e2_8 = _weighted_waits("e2_8")
    e2_9 = _weighted_waits("e2_9")
    e2_10 = _weighted_waits("e2_10")
    
    pedestrian = 0
    junction_subscription = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
    
    if junction_subscription:
        for pid, data in junction_subscription.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
            
    return [e2_4, e2_5, e2_6, e2_7, e2_8, e2_9, e2_10, pedestrian]

def calculate_reward(current_state):
    if current_state is None:
        print("ERROR: STATE UNDETECETED")
        return 0
    
    current_total = sum(current_state)
    return -current_total

def _mainIntersection_phase(action_index):
    global mainCurrentPhase, mainCurrentPhaseDuration
    
    mainCurrentPhase += 1
    mainCurrentPhase = mainCurrentPhase % 10
    
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", mainCurrentPhase)
    
    if mainCurrentPhase % 2 == 1:
        phase_duration = 5
    else:
        duration_adjustment = actionSpace[action_index]
        if mainCurrentPhase == 4:
            base_duration = 20
        else:
            base_duration = 30
        
        phase_duration = max(5, min(60, base_duration + duration_adjustment))
    
    mainCurrentPhaseDuration = phase_duration
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", mainCurrentPhaseDuration)
    
def save_history(filename, headers, reward_hist, loss_hist, epsilon_hist, train_frequency):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header only if new file
            if not file_exists:
                writer.writerow(headers)
            for i in range(len(reward_hist)):
            # Save the simulation step number (i * frequency)
                writer.writerow([i * train_frequency, reward_hist[i], loss_hist[i], epsilon_hist[i]])









traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("cluster_295373794_3477931123_7465167861")

main_phase = to_categorical(mainCurrentPhase//2, num_classes=5).flatten()

# Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    mainCurrentPhaseDuration -= stepLength
    
    #Observation and Reward
    if mainCurrentPhaseDuration <= 0:
        if mainCurrentPhase % 2 == 0:
            main_queue = np.array(_mainIntersection_queue())
            normalized_main_queue = main_queue/1000
            main_phase = to_categorical(mainCurrentPhase//2, num_classes=5).flatten()
            mainCurrentState = np.concatenate([normalized_main_queue, main_phase]).astype(np.float32)
            
            if trainMode == 1:
                mainReward = calculate_reward(normalized_main_queue*10)
                total_main_reward += mainReward
                
                if mainPrevState is not None and mainPrevAction is not None:
                    done = False
                    mainIntersectionAgent.remember(mainPrevState, mainPrevAction, mainReward, mainCurrentState, done)
        
    #Action
    if mainCurrentPhaseDuration <= 0:
        if mainCurrentPhase % 2 == 0:
            mainActionIndex = mainIntersectionAgent.act(mainCurrentState)
            mainPrevState = mainCurrentState
            mainPrevAction = mainActionIndex
            
            if trainMode == 1:
                print(f"\nMain Intersection - Queue: {sum(normalized_main_queue)}, Reward: {mainReward}, Action: {actionSpace[mainActionIndex]}")
        else:
            mainActionIndex = mainPrevAction

        _mainIntersection_phase(mainActionIndex)
        
    # Periodic training (replay)
    if trainMode == 1 and step_counter % TRAIN_FREQUENCY == 0:
        if len(mainIntersectionAgent.memory) >= BATCH_SIZE:
            loss = mainIntersectionAgent.replay(BATCH_SIZE)
            main_loss_history.append(loss)
            main_reward_history.append(total_main_reward)
            main_epsilon_history.append(mainIntersectionAgent.epsilon)
            total_main_reward = 0
            mainIntersectionAgent.epsilon = max(mainIntersectionAgent.epsilon_min, 
                                               mainIntersectionAgent.epsilon * mainIntersectionAgent.epsilon_decay_rate)
        
    # Periodic tracking (throughput and queue_length)
    TRACK_INTERVAL_STEPS = int(6 / stepLength)
    if trainMode == 0 and step_counter % TRACK_INTERVAL_STEPS == 0 :
        jam_length = 0
        throughput = 0
        metric_observation_count += 1
        
        for det_id in detector_ids:
            detector_stats = traci.lanearea.getSubscriptionResults(det_id)

            if not detector_stats:
                print("Lane Data Error: Undetected")
                break
            
            jam_length += detector_stats.get(traci.constants.JAM_LENGTH_METERS, 0)
            throughput += detector_stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)
                
        jam_length /= detector_count
        jam_length_total += jam_length
        throughput_total += throughput
        
    traci.simulationStep()

if metric_observation_count > 0:
    jam_length_average = jam_length_total / metric_observation_count
    throughput_average = throughput_total / metric_observation_count
else:
    jam_length_average = 0
    throughput_average = 0

print("\n Queue Length:", jam_length_average)
print("\n Throughput:", throughput_average)

if trainMode == 1:
    # Save trained models
    print("Saving trained models...")
    mainIntersectionAgent.save()
    print("Models saved successfully!")

    print("Saving training history...")
    save_history('./Olivarez_traci/output_DQN_Baseline/baseline_agent_history.csv', ['Step', 'Reward', 'Loss', 'Epsilon'], 
                main_reward_history, main_loss_history, main_epsilon_history, TRAIN_FREQUENCY)
    print("History saved successfully!")

traci.close()