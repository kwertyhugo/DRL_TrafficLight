import os
import sys 
import traci
import numpy as np
import csv
from keras.utils import to_categorical

from models.DQN import DQNAgent as dqn

trafficLightAgent = dqn(state_size=19, action_size=11, memory_size=2000, gamma=0.95, epsilon=0, epsilon_decay_rate=0.995, epsilon_min=0, learning_rate=0.00005, target_update_freq=500, name='ReLU_DQNAgent1')

# trafficLightAgent.load()

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo-gui',
    '-c', 'Olivarez_traci\signalizedPed.sumocfg',
    '--step-length', '0.05',
    '--delay', '100',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_DQN\SD_DQN_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_DQN\SD_DQN_trips.xml'
]

# Simulation Variables
trainMode = 0
stepLength = 0.05
currentPhase = 0
currentPhaseDuration = 30
actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)

# Store previous states and actions for learning
prevState = None
prevAction = None

# Batch training parameters
BATCH_SIZE = 32
TRAIN_FREQUENCY = 20  # Train every 20 steps / 1 seconds
step_counter = 0

# -- Data storage for plotting --
reward_history = []
loss_history = []
epsilon_history = []
total_reward = 0

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

def _mainIntersection_queue():
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
    
    pedestrianM = 0
    junction_subscriptionM = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
    pedestrianW = 0
    junction_subscriptionW = traci.junction.getContextSubscriptionResults("6401523012")
    pedestrianE = 0
    junction_subscriptionE = traci.junction.getContextSubscriptionResults("3285696417")

    if junction_subscriptionM:
        for pid, data in junction_subscriptionM.items():
            pedestrianM += data.get(traci.constants.VAR_WAITING_TIME, 0)

    if junction_subscriptionW:
        for pid, data in junction_subscriptionW.items():
            pedestrianW += data.get(traci.constants.VAR_WAITING_TIME, 0)

    if junction_subscriptionE:
        for pid, data in junction_subscriptionE.items():
            pedestrianE += data.get(traci.constants.VAR_WAITING_TIME, 0)
            
    return [e2_0, e2_1, e2_2, e2_3, e2_4, e2_5, e2_6, e2_7, e2_8, e2_9, e2_10, pedestrianM*1.5, pedestrianW*1.5, pedestrianE*1.5]


def calculate_reward(current_state):
    if current_state is None:
        print("ERROR: STATE UNDETECETED")
        return 0
    
    current_total = sum(current_state)
    return -current_total

def _trafficLight_phase(action_index):
    global currentPhase, currentPhaseDuration
    
    currentPhase += 1
    currentPhase = currentPhase % 10
    
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", currentPhase)
    traci.trafficlight.setPhase("6401523012", currentPhase)
    traci.trafficlight.setPhase("3285696417", currentPhase)
    
    if currentPhase % 2 == 1:
        phase_duration = 5
    else:
        duration_adjustment = actionSpace[action_index]
        if currentPhase != 2:
            base_duration = 30
        else:
            base_duration = 20
        
        phase_duration = max(5, min(60, base_duration + duration_adjustment))
    
    currentPhaseDuration = phase_duration
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", currentPhaseDuration)
    traci.trafficlight.setPhaseDuration("6401523012", currentPhaseDuration)
    traci.trafficlight.setPhaseDuration("3285696417", currentPhaseDuration)

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
currentPhase_onehot = to_categorical(currentPhase//2, num_classes=5).flatten()


# Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    currentPhaseDuration -= stepLength

    #Observation and Reward
    if currentPhaseDuration <= 0:
        if currentPhase % 2 == 0:
            queue = np.array(_mainIntersection_queue())
            normalized_queue = queue/1000
            currentPhase_onehot = to_categorical(currentPhase//2, num_classes=5).flatten()
            currentState = np.concatenate([normalized_queue, currentPhase_onehot]).astype(np.float32)
            
            if trainMode == 1:
                reward = calculate_reward(normalized_queue*10)
                total_reward += reward
                
                if prevState is not None and prevAction is not None:
                    done = False
                    trafficLightAgent.remember(prevState, prevAction, reward, currentState, done)

    #Action
    if currentPhaseDuration <= 0:
        if currentPhase % 2 == 0:
            actionIndex = trafficLightAgent.act(currentState)
            prevState = currentState
            prevAction = actionIndex
            
            if trainMode == 1:
                print(f"Intersection - Queue: {np.sum(normalized_queue)}, Reward: {reward}, Action: {actionSpace[actionIndex]}")
        else:
            actionIndex = prevAction

        _trafficLight_phase(actionIndex)
        

    # Periodic training (replay)
    if trainMode == 1 and step_counter % TRAIN_FREQUENCY == 0 and step_counter > 600/stepLength:
        if len(trafficLightAgent.memory) >= BATCH_SIZE:
            loss = trafficLightAgent.replay(BATCH_SIZE)
            loss_history.append(loss)
            reward_history.append(total_reward)
            epsilon_history.append(trafficLightAgent.epsilon)
            total_reward = 0
            trafficLightAgent.epsilon = max(trafficLightAgent.epsilon_min, 
                                               trafficLightAgent.epsilon * trafficLightAgent.epsilon_decay_rate)
        
    traci.simulationStep()





if trainMode == 1:
    # Save trained models
    print("Saving trained models...")
    trafficLightAgent.save()
    print("Models saved successfully!")

    print("Saving training history...")
    save_history('./Olivarez_traci/output_DQN/main_agent_history.csv', ['Step', 'Reward', 'Loss', 'Epsilon'], 
                reward_history, loss_history, epsilon_history, TRAIN_FREQUENCY)
    print("History saved successfully!")

traci.close()