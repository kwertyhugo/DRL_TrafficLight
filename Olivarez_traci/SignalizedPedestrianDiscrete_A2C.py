import os
import sys

# Add parent directory to path so we can import from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traci
import numpy as np
import csv
from keras.utils import to_categorical

from models.A2C import A2CAgent as a2c

# Select A2C Agents
mainIntersectionAgent = a2c(state_size=15, action_size=7, gamma=0.95, learning_rate=0.0001, 
                        entropy_coef=0.1, value_coef=0.5, name='A2C_Main_Agent')
swPedXingAgent = a2c(state_size=7, action_size=7, gamma=0.95, learning_rate=0.0001,
                    entropy_coef=0.1, value_coef=0.5, name='A2C_SW_PedXing_Agent')
sePedXingAgent = a2c(state_size=7, action_size=7, gamma=0.95, learning_rate=0.0001,
                    entropy_coef=0.1, value_coef=0.5, name='A2C_SE_PedXing_Agent')

# Uncomment to load pre-trained models
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
    '--statistic-output', r'Olivarez_traci\output_A2C\SD_DQN_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_A2C\SD_DQN_trips.xml'
]

# Simulation Variables
stepLength = 0.05
mainCurrentPhase = 0
mainCurrentPhaseDuration = 30
swCurrentPhase = 0
swCurrentPhaseDuration = 30
seCurrentPhase = 0
seCurrentPhaseDuration = 30
actionSpace = (-15, -10, -5, 0, 5, 10, 15)

# Store previous states and actions for learning
mainPrevState = None
mainPrevAction = None
swPrevState = None
swPrevAction = None
sePrevState = None
sePrevAction = None

# Training parameters
TRAIN_FREQUENCY = 100  # Train every 100 steps / 5 seconds
step_counter = 0

# -- Data storage for plotting --
main_actor_loss_history = []
main_critic_loss_history = []
main_entropy_history = []
main_reward_history = []

sw_actor_loss_history = []
sw_critic_loss_history = []
sw_entropy_history = []
sw_reward_history = []

se_actor_loss_history = []
se_critic_loss_history = []
se_entropy_history = []
se_reward_history = []

# -- Variables to accumulate reward between training steps --
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
    # List of all your vehicle detectors
    detector_ids = [
        "e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10",
        "e2_0", "e2_1", "e2_2", "e2_3"
    ]
    
    # The variables we want from each vehicle inside the detector
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
    southwest = _weighted_waits("e2_4") + _weighted_waits("e2_5")
    southeast = _weighted_waits("e2_6") + _weighted_waits("e2_7")
    northeast = _weighted_waits("e2_8")
    northwest = _weighted_waits("e2_9") + _weighted_waits("e2_10")
    
    pedestrian = 0
    junction_subscription = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
    
    if junction_subscription:
        for pid, data in junction_subscription.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
            
    return [southwest, southeast, northeast, northwest, pedestrian]

def _swPedXing_queue():
    north = _weighted_waits("e2_0") + _weighted_waits("e2_1")
    south = _weighted_waits("e2_4") + _weighted_waits("e2_5")
    pedestrian = 0
    junction_subscription = traci.junction.getContextSubscriptionResults("6401523012")
    
    if junction_subscription:
        for pid, data in junction_subscription.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
            
    return [south, north, pedestrian]

def _sePedXing_queue():
    west = _weighted_waits("e2_2") + _weighted_waits("e2_3")
    east = _weighted_waits("e2_6") + _weighted_waits("e2_7")
    pedestrian = 0
    junction_subscription = traci.junction.getContextSubscriptionResults("3285696417")
    
    if junction_subscription:
        for pid, data in junction_subscription.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)

    return [west, east, pedestrian]

# Calculate reward based on queue reduction
def calculate_reward(current_state, prev_state):
    if prev_state is None:
        return 0
    
    current_total = sum(current_state)
    prev_total = sum(prev_state)
    
    # Reward for reducing queue, penalty for increasing
    queue_diff = prev_total - current_total
    
    # Scale down and clip reward for stability
    reward = np.clip(queue_diff * 0.1, -5, 5)
    
    return reward

# Output of the model
def _mainIntersection_phase(action_index):
    global mainCurrentPhase, mainCurrentPhaseDuration
    
    mainCurrentPhase += 1
    mainCurrentPhase = mainCurrentPhase % 10
    
    duration_adjustment = actionSpace[action_index]
    
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", mainCurrentPhase)
    
    if mainCurrentPhase == 2 or mainCurrentPhase == 4:
        base_duration = 15
    elif mainCurrentPhase % 2 == 0:
        base_duration = 30
    else:
        base_duration = 3
    
    mainCurrentPhaseDuration = max(5, min(60, base_duration + duration_adjustment))
    
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", mainCurrentPhaseDuration)
    
def _swPedXing_phase(action_index):
    global swCurrentPhase, swCurrentPhaseDuration
    
    swCurrentPhase += 1
    swCurrentPhase = swCurrentPhase % 4
    
    duration_adjustment = actionSpace[action_index]
    
    traci.trafficlight.setPhase("6401523012", swCurrentPhase)
    
    if swCurrentPhase % 2 == 1:
        base_duration = 5
    elif swCurrentPhase % 2 == 0:
        base_duration = 30
    
    swCurrentPhaseDuration = max(5, min(60, base_duration + duration_adjustment))
    
    traci.trafficlight.setPhaseDuration("6401523012", swCurrentPhaseDuration)
    
def _sePedXing_phase(action_index):
    global seCurrentPhase, seCurrentPhaseDuration
    
    seCurrentPhase += 1
    seCurrentPhase = seCurrentPhase % 4
    
    duration_adjustment = actionSpace[action_index]
    
    traci.trafficlight.setPhase("3285696417", seCurrentPhase)
    
    if seCurrentPhase % 2 == 1:
        base_duration = 5
    elif seCurrentPhase % 2 == 0:
        base_duration = 30
    
    seCurrentPhaseDuration = max(5, min(60, base_duration + duration_adjustment))
    
    traci.trafficlight.setPhaseDuration("3285696417", seCurrentPhaseDuration)

def save_history(filename, headers, reward_hist, actor_loss_hist, critic_loss_hist, entropy_hist, train_frequency):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header only if new file
            if not file_exists:
                writer.writerow(headers)
            for i in range(len(reward_hist)):
                writer.writerow([i * train_frequency, reward_hist[i], actor_loss_hist[i], 
                        critic_loss_hist[i], entropy_hist[i]])

traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("cluster_295373794_3477931123_7465167861")
_junctionSubscription("6401523012")
_junctionSubscription("3285696417")

detector_ids = traci.lanearea.getIDList()

# Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    
    # Main Intersection Agent Logic
    mainCurrentPhaseDuration -= stepLength
    if mainCurrentPhaseDuration <= 0:
        # Get current state
        state_list = _mainIntersection_queue()
        state_vector = to_categorical(mainCurrentPhase, num_classes=10).flatten()
        mainCurrentState = np.concatenate([state_list, state_vector]).astype(np.float32)
        
        # Calculate reward from previous action
        mainReward = calculate_reward(mainCurrentState, mainPrevState)
        total_main_reward += mainReward
        
        # Store experience (A2C stores in trajectory, not replay buffer)
        if mainPrevState is not None and mainPrevAction is not None:
            done = False
            mainIntersectionAgent.remember(mainPrevState, mainPrevAction, mainReward, mainCurrentState, done)
        
        # Choose new action (sampled from policy)
        mainActionIndex = mainIntersectionAgent.act(mainCurrentState)
        
        # Apply phase change with action
        _mainIntersection_phase(mainActionIndex)
        
        # Store current state and action for next iteration
        mainPrevState = mainCurrentState
        mainPrevAction = mainActionIndex
        
        print(f"Main Intersection - Queue: {sum(mainCurrentState[:5]):.2f}, Reward: {mainReward:.2f}, Action: {actionSpace[mainActionIndex]}")
    
    # SW Pedestrian Crossing Agent Logic
    swCurrentPhaseDuration -= stepLength
    if swCurrentPhaseDuration <= 0:
        state_list = _swPedXing_queue()
        state_vector = to_categorical(swCurrentPhase, num_classes=4).flatten()
        swCurrentState = np.concatenate([state_list, state_vector]).astype(np.float32)
        
        swReward = calculate_reward(swCurrentState, swPrevState)
        total_sw_reward += swReward
        
        if swPrevState is not None and swPrevAction is not None:
            done = False
            swPedXingAgent.remember(swPrevState, swPrevAction, swReward, swCurrentState, done)
        
        swActionIndex = swPedXingAgent.act(swCurrentState)
        _swPedXing_phase(swActionIndex)
        
        swPrevState = swCurrentState
        swPrevAction = swActionIndex
        
        print(f"SW Ped Crossing - Queue: {sum(swCurrentState[:3]):.2f}, Reward: {swReward:.2f}, Action: {actionSpace[swActionIndex]}")
    
    # SE Pedestrian Crossing Agent Logic
    seCurrentPhaseDuration -= stepLength
    if seCurrentPhaseDuration <= 0:
        state_list = _sePedXing_queue()
        state_vector = to_categorical(seCurrentPhase, num_classes=4).flatten()
        seCurrentState = np.concatenate([state_list, state_vector]).astype(np.float32)
        
        seReward = calculate_reward(seCurrentState, sePrevState)
        total_se_reward += seReward
        
        if sePrevState is not None and sePrevAction is not None:
            done = False
            sePedXingAgent.remember(sePrevState, sePrevAction, seReward, seCurrentState, done)
        
        seActionIndex = sePedXingAgent.act(seCurrentState)
        _sePedXing_phase(seActionIndex)
        
        sePrevState = seCurrentState
        sePrevAction = seActionIndex
        
        print(f"SE Ped Crossing - Queue: {sum(seCurrentState[:3]):.2f}, Reward: {seReward:.2f}, Action: {actionSpace[seActionIndex]}")
    
    # Periodic training (A2C trains on trajectory)
    if step_counter % TRAIN_FREQUENCY == 0:
        # Train main intersection agent
        actor_loss, critic_loss, entropy = mainIntersectionAgent.train()
        if actor_loss > 0:  # Only log if training occurred
            main_actor_loss_history.append(actor_loss)
            main_critic_loss_history.append(critic_loss)
            main_entropy_history.append(entropy)
            main_reward_history.append(total_main_reward)
            total_main_reward = 0
            print(f"[Main] Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")
        
        # Train SW pedestrian crossing agent
        actor_loss, critic_loss, entropy = swPedXingAgent.train()
        if actor_loss > 0:
            sw_actor_loss_history.append(actor_loss)
            sw_critic_loss_history.append(critic_loss)
            sw_entropy_history.append(entropy)
            sw_reward_history.append(total_sw_reward)
            total_sw_reward = 0
            print(f"[SW] Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")
        
        # Train SE pedestrian crossing agent
        actor_loss, critic_loss, entropy = sePedXingAgent.train()
        if actor_loss > 0:
            se_actor_loss_history.append(actor_loss)
            se_critic_loss_history.append(critic_loss)
            se_entropy_history.append(entropy)
            se_reward_history.append(total_se_reward)
            total_se_reward = 0
            print(f"[SE] Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")
    
    traci.simulationStep()

# Save trained models
print("Saving trained models...")
mainIntersectionAgent.save()
swPedXingAgent.save()
sePedXingAgent.save()
print("Models saved successfully!")

print("Saving training history...")
save_history('a2c_main_agent_history.csv', ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'], 
            main_reward_history, main_actor_loss_history, main_critic_loss_history, main_entropy_history, TRAIN_FREQUENCY)
            
save_history('a2c_sw_agent_history.csv', ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'], 
            sw_reward_history, sw_actor_loss_history, sw_critic_loss_history, sw_entropy_history, TRAIN_FREQUENCY)
            
save_history('a2c_se_agent_history.csv', ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'], 
            se_reward_history, se_actor_loss_history, se_critic_loss_history, se_entropy_history, TRAIN_FREQUENCY)

print("History saved successfully!")

traci.close()