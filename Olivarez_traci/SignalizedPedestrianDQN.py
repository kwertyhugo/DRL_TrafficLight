import os
import sys 
import traci
import numpy as np
import csv
from keras.utils import to_categorical

from models.DQN import DQNAgent as dqn

### --- CHANGED --- ###
# State sizes are recalculated based on exclusive states.
# Main Agent: 12 queues (11 car + 1 ped) + 9 phases (5+2+2) = 21
# Ped Agents: 1 queue (1 ped) + 9 phases (5+2+2) = 10
mainIntersectionAgent = dqn(state_size=21, action_size=11, memory_size=200, gamma=0.95, epsilon=1, epsilon_decay_rate=0.995, epsilon_min=0.01, learning_rate=0.00005, target_update_freq=500, name='ReLU_DQNAgent')
swPedXingAgent = dqn(state_size=10, action_size=11, memory_size=200, gamma=0.95, epsilon=1, epsilon_decay_rate=0.995, epsilon_min=0.01, learning_rate=0.00005, target_update_freq=500, name='SW_PedXing_Agent')
sePedXingAgent = dqn(state_size=10, action_size=11, memory_size=200, gamma=0.95, epsilon=1, epsilon_decay_rate=0.995, epsilon_min=0.01, learning_rate=0.00005, target_update_freq=500, name='SE_PedXing_Agent')

# Start fresh training
# mainIntersectionAgent.load()
# swPedXingAgent.load()
# sePedXingAgent.load()

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo',
    '-c', 'Olivarez_traci\map.sumocfg',
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
swCurrentPhase = 0
swCurrentPhaseDuration = 30
seCurrentPhase = 0
seCurrentPhaseDuration = 30
actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)

# Store previous states and actions for learning
mainPrevState = None
mainPrevAction = None
swPrevState = None
swPrevAction = None
sePrevState = None
sePrevAction = None

# Batch training parameters
BATCH_SIZE = 32
TRAIN_FREQUENCY = 100  # Train every 100 steps / 5 seconds
step_counter = 0

# -- Data storage for plotting --
main_reward_history = []
main_loss_history = []
main_epsilon_history = []

sw_reward_history = []
sw_loss_history = []
sw_epsilon_history = []

se_reward_history = []
se_loss_history = []
se_epsilon_history = []

total_main_reward = 0
total_sw_reward = 0
total_se_reward = 0

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
        "e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10",
        "e2_0", "e2_1", "e2_2", "e2_3"
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

### --- CHANGED --- ###
# Main agent now "owns" ALL vehicle detectors
def _mainIntersection_queue():
    # All 11 car detectors
    e2_0 = _weighted_waits("e2_0")
    e2_1 = _weighted_waits("e2_1")
    e2_2 = _weighted_waits("e2_2")
    e2_3 = _weighted_waits("e2_3")
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
            
    # Return all 12 queue values
    return [e2_0, e2_1, e2_2, e2_3, e2_4, e2_5, e2_6, e2_7, e2_8, e2_9, e2_10, pedestrian]

### --- CHANGED --- ###
# SW Ped agent now "owns" ONLY its pedestrian detector
def _swPedXing_queue():
    # e2_0 = _weighted_waits("e2_0") # <-- REMOVED
    # e2_1 = _weighted_waits("e2_1") # <-- REMOVED
    # e2_4 = _weighted_waits("e2_4") # <-- REMOVED
    # e2_5 = _weighted_waits("e2_5") # <-- REMOVED
    pedestrian = 0
    junction_subscription = traci.junction.getContextSubscriptionResults("6401523012")
    
    if junction_subscription:
        for pid, data in junction_subscription.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
            
    return [pedestrian] # State is ONLY pedestrian queue

### --- CHANGED --- ###
# SE Ped agent now "owns" ONLY its pedestrian detector
def _sePedXing_queue():
    # e2_2 = _weighted_waits("e2_2") # <-- REMOVED
    # e2_3 = _weighted_waits("e2_3") # <-- REMOVED
    # e2_6 = _weighted_waits("e2_6") # <-- REMOVED
    # e2_7 = _weighted_waits("e2_7") # <-- REMOVED
    pedestrian = 0
    junction_subscription = traci.junction.getContextSubscriptionResults("3285696417")
    
    if junction_subscription:
        for pid, data in junction_subscription.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)

    return [pedestrian] # State is ONLY pedestrian queue

# This reward function is perfect and works for all agents
def calculate_reward(current_state_queues):
    if current_state_queues is None:
        print("ERROR: STATE UNDETECETED")
        return 0
    
    # current_state_queues will be [pedestrian] for ped agents
    # and [e2_0, ..., pedestrian] for main agent. sum() works perfectly.
    current_total = sum(current_state_queues)
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
    
def _swPedXing_phase(action_index):
    global swCurrentPhase, swCurrentPhaseDuration
    
    swCurrentPhase += 1
    swCurrentPhase = swCurrentPhase % 4
    
    traci.trafficlight.setPhase("6401523012", swCurrentPhase)

    if swCurrentPhase % 2 == 1:
        phase_duration = 5
    else:
        duration_adjustment = actionSpace[action_index]
        base_duration = 30
        phase_duration = max(5, min(60, base_duration + duration_adjustment))

    swCurrentPhaseDuration = phase_duration
    traci.trafficlight.setPhaseDuration("6401523012", swCurrentPhaseDuration)
    
def _sePedXing_phase(action_index):
    global seCurrentPhase, seCurrentPhaseDuration
    
    seCurrentPhase += 1
    seCurrentPhase = seCurrentPhase % 4
    
    traci.trafficlight.setPhase("3285696417", seCurrentPhase)
    
    
    if seCurrentPhase % 2 == 1:
        phase_duration = 5
    else:
        duration_adjustment = actionSpace[action_index]
        base_duration = 30
        phase_duration = max(5, min(60, base_duration + duration_adjustment))
    
    seCurrentPhaseDuration = phase_duration
    traci.trafficlight.setPhaseDuration("3285696417", seCurrentPhaseDuration)

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
_junctionSubscription("6401523012")
_junctionSubscription("3285696417")

detector_ids = traci.lanearea.getIDList()
main_phase = to_categorical(mainCurrentPhase//2, num_classes=5).flatten()
swPed_phase = to_categorical(swCurrentPhase//2, num_classes=2).flatten()
sePed_phase = to_categorical(seCurrentPhase//2, num_classes=2).flatten()


# Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    mainCurrentPhaseDuration -= stepLength
    swCurrentPhaseDuration -= stepLength
    seCurrentPhaseDuration -= stepLength
    
    #Observation and Reward
    if mainCurrentPhaseDuration <= 0:
        if mainCurrentPhase % 2 == 0:
            ### --- CHANGED --- ###
            # main_queue now has 12 elements
            main_queue = np.array(_mainIntersection_queue())
            normalized_main_queue = main_queue/1000
            main_phase = to_categorical(mainCurrentPhase//2, num_classes=5).flatten()
            
            # This is the state - it has 12 + 5 + 2 + 2 = 21 elements
            mainCurrentState = np.concatenate([normalized_main_queue, main_phase, swPed_phase, sePed_phase]).astype(np.float32)
            
            if trainMode == 1:
                # Reward is based on all 12 queues
                mainReward = calculate_reward(main_queue) # Pass raw queue for reward
                total_main_reward += mainReward
                
                if mainPrevState is not None and mainPrevAction is not None:
                    done = False
                    mainIntersectionAgent.remember(mainPrevState, mainPrevAction, mainReward, mainCurrentState, done)
    
    if swCurrentPhaseDuration <= 0:
        if swCurrentPhase % 2 == 0:
            ### --- CHANGED --- ###
            # swPed_queue now has 1 element
            swPed_queue = np.array(_swPedXing_queue())
            normalized_swPed_queue = swPed_queue/1000
            swPed_phase = to_categorical(swCurrentPhase//2, num_classes=2).flatten()
            
            # This is the state - it has 1 + 5 + 2 + 2 = 10 elements
            swCurrentState = np.concatenate([normalized_swPed_queue, main_phase, swPed_phase, sePed_phase]).astype(np.float32)
            
            if trainMode == 1:
                # Reward is based only on the 1 pedestrian queue
                swReward = calculate_reward(swPed_queue) # Pass raw queue for reward
                total_sw_reward += swReward
                
                if swPrevState is not None and swPrevAction is not None:
                    done = False
                    swPedXingAgent.remember(swPrevState, swPrevAction, swReward, swCurrentState, done)
    
    if seCurrentPhaseDuration <= 0:
        if seCurrentPhase % 2 == 0:
            ### --- CHANGED --- ###
            # sePed_queue now has 1 element
            sePed_queue = np.array(_sePedXing_queue())
            normalized_sePed_queue = sePed_queue/1000
            sePed_phase = to_categorical(seCurrentPhase//2, num_classes=2).flatten()
            
            # This is the state - it has 1 + 5 + 2 + 2 = 10 elements
            seCurrentState = np.concatenate([normalized_sePed_queue, main_phase, swPed_phase, sePed_phase]).astype(np.float32)
            
            if trainMode == 1:
                # Reward is based only on the 1 pedestrian queue
                seReward = calculate_reward(sePed_queue) # Pass raw queue for reward
                total_se_reward += seReward
                
                if sePrevState is not None and sePrevAction is not None:
                    done = False
                    sePedXingAgent.remember(sePrevState, sePrevAction, seReward, seCurrentState, done)
    
    #Action
    if mainCurrentPhaseDuration <= 0:
        if mainCurrentPhase % 2 == 0:
            mainActionIndex = mainIntersectionAgent.act(mainCurrentState)
            mainPrevState = mainCurrentState
            mainPrevAction = mainActionIndex
            
            if trainMode == 1:
                # Pass normalized queue for printing
                print(f"Main Intersection - Queue: {sum(normalized_main_queue):.2f}, Reward: {mainReward:.2f}, Action: {actionSpace[mainActionIndex]}")
        else:
            mainActionIndex = mainPrevAction

        _mainIntersection_phase(mainActionIndex)
        
    if swCurrentPhaseDuration <= 0:
        if swCurrentPhase % 2 == 0:
            swActionIndex = swPedXingAgent.act(swCurrentState)
            swPrevState = swCurrentState
            swPrevAction = swActionIndex
            
            if trainMode == 1:
                # Pass normalized queue for printing
                print(f"SW Ped Crossing - Queue: {sum(normalized_swPed_queue):.2f}, Reward: {swReward:.2f}, Action: {actionSpace[swActionIndex]}")
        else:
            swActionIndex = swPrevAction
        
        _swPedXing_phase(swActionIndex)

    if seCurrentPhaseDuration <= 0:
        if seCurrentPhase % 2 == 0:
            seActionIndex = sePedXingAgent.act(seCurrentState)
            sePrevState = seCurrentState
            sePrevAction = seActionIndex
            
            if trainMode == 1:
                # Pass normalized queue for printing
                print(f"SE Ped Crossing - Queue: {sum(normalized_sePed_queue):.2f}, Reward: {seReward:.2f}, Action: {actionSpace[seActionIndex]}")
        else:
            seActionIndex = sePrevAction
        
        _sePedXing_phase(seActionIndex)

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
        
        if len(swPedXingAgent.memory) >= BATCH_SIZE:
            loss = swPedXingAgent.replay(BATCH_SIZE)
            sw_loss_history.append(loss)
            sw_reward_history.append(total_sw_reward)
            sw_epsilon_history.append(swPedXingAgent.epsilon)
            total_sw_reward = 0
            swPedXingAgent.epsilon = max(swPedXingAgent.epsilon_min,
                                        swPedXingAgent.epsilon * swPedXingAgent.epsilon_decay_rate)
        
        if len(sePedXingAgent.memory) >= BATCH_SIZE:
            loss = sePedXingAgent.replay(BATCH_SIZE)
            se_loss_history.append(loss)
            se_reward_history.append(total_se_reward)
            se_epsilon_history.append(sePedXingAgent.epsilon)
            total_se_reward = 0
            sePedXingAgent.epsilon = max(sePedXingAgent.epsilon_min,
                                        sePedXingAgent.epsilon * sePedXingAgent.epsilon_decay_rate)
    
    traci.simulationStep()

if trainMode == 1:
    # Save trained models
    print("Saving trained models...")
    mainIntersectionAgent.save()
    swPedXingAgent.save()
    sePedXingAgent.save()
    print("Models saved successfully!")

    print("Saving training history...")
    save_history('./Olivarez_traci/output_DQN/main_agent_history.csv', ['Step', 'Reward', 'Loss', 'Epsilon'], 
                 main_reward_history, main_loss_history, main_epsilon_history, TRAIN_FREQUENCY)
                 
    save_history('./Olivarez_traci/output_DQN/sw_agent_history.csv', ['Step', 'Reward', 'Loss', 'Epsilon'], 
                 sw_reward_history, sw_loss_history, sw_epsilon_history, TRAIN_FREQUENCY)
                 
    save_history('./Olivarez_traci/output_DQN/se_agent_history.csv', ['Step', 'Reward', 'Loss', 'Epsilon'], 
                 se_reward_history, se_loss_history, se_epsilon_history, TRAIN_FREQUENCY)

    print("History saved successfully!")

traci.close()