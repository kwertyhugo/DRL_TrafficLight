import os
import sys 
import traci
import numpy as np
import csv
from keras.utils import to_categorical

from models.DQN import DQNAgent as dqn

mainIntersectionAgent = dqn(state_size=13, action_size=11, memory_size=2000, gamma=0.95, epsilon=0, epsilon_decay_rate=0.995, epsilon_min=0.001, learning_rate=0.00005, target_update_freq=500, name='ReLU_DQNAgent_Baseline')

# mainIntersectionAgent.load()

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
    '--statistic-output', r'Olivarez_traci\output_DQN\SD_DQN_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_DQN\SD_DQN_trips.xml'
]

# Simulation Variables
trainMode = 1
stepLength = 0.05
mainCurrentPhase = 0
mainCurrentPhaseDuration = 30
actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)

# Store previous states and actions for learning
mainPrevState = None
mainPrevAction = None

# Batch training parameters
BATCH_SIZE = 32
TRAIN_FREQUENCY = 100  # Train every 100 steps / 5 seconds
step_counter = 0

# -- Data storage for plotting --
main_reward_history = []
main_loss_history = []
main_epsilon_history = []
total_main_reward = 0

#Object Context Subscription in SUMO
def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id,
        traci.constants.CMD_GET_PERSON_VARIABLE,
        10.0,
        [traci.constants.VAR_WAITING_TIME]
    )
    
def _subscribe_all_detectors():
    detector_ids = [
        "e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10"
    ]
    
    vehicle_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(
            det_id,
            traci.constants.CMD_GET_VEHICLE_VARIABLE,
            3,
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

def _swPedXing_queue():
    e2_0 = _weighted_waits("e2_0")
    e2_1 = _weighted_waits("e2_1")
    e2_4 = _weighted_waits("e2_4")
    e2_5 = _weighted_waits("e2_5")
    pedestrian = 0
    junction_subscription = traci.junction.getContextSubscriptionResults("6401523012")
    
    if junction_subscription:
        for pid, data in junction_subscription.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
            
    return [e2_0, e2_1, e2_4, e2_5, pedestrian]

def _sePedXing_queue():
    e2_2 = _weighted_waits("e2_2")
    e2_3 = _weighted_waits("e2_3")
    e2_6 = _weighted_waits("e2_6")
    e2_7 = _weighted_waits("e2_7")
    pedestrian = 0
    junction_subscription = traci.junction.getContextSubscriptionResults("3285696417")
    
    if junction_subscription:
        for pid, data in junction_subscription.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)

    return [e2_2, e2_3, e2_6, e2_7, pedestrian]

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
        if mainCurrentPhase == 2 or mainCurrentPhase == 4:
            base_duration = 15
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

detector_ids = traci.lanearea.getIDList()
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
        
    traci.simulationStep()





if trainMode == 1:
    # Save trained models
    print("Saving trained models...")
    mainIntersectionAgent.save()
    print("Models saved successfully!")

    print("Saving training history...")
    save_history('./Olivarez_traci/output_DQN/baseline_agent_history.csv', ['Step', 'Reward', 'Loss', 'Epsilon'], 
                main_reward_history, main_loss_history, main_epsilon_history, TRAIN_FREQUENCY)
    print("History saved successfully!")

traci.close()