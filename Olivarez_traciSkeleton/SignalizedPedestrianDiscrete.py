import os
import sys 
import traci
import numpy as np

from models.DQN import DQNAgent as dqn

# Select DRL Agent
mainIntersectionAgent = dqn(state_size=5, action_size=7, memory_size=200, gamma=0.95, epsilon=1.0, epsilon_decay_rate=0.995, epsilon_min=0.01, learning_rate=0.0002, name='ReLU_DQNAgent')
swPedXingAgent = dqn(state_size=3, action_size=7, memory_size=200, gamma=0.95, epsilon=1.0, epsilon_decay_rate=0.995, epsilon_min=0.01, learning_rate=0.0002, name='SW_PedXing_Agent')
sePedXingAgent = dqn(state_size=3, action_size=7, memory_size=200, gamma=0.95, epsilon=1.0, epsilon_decay_rate=0.995, epsilon_min=0.01, learning_rate=0.0002, name='SE_PedXing_Agent')

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo-gui',
    '-c', 'Olivarez_traciSkeleton\map.sumocfg',
    '--step-length', '0.05',
    '--delay', '0',
    '--lateral-resolution', '0.1'
]

def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id,
        traci.constants.CMD_GET_PERSON_VARIABLE,
        10.0,
        [traci.constants.VAR_SPEED]
    )

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

# Batch training parameters
BATCH_SIZE = 32
TRAIN_FREQUENCY = 1000  # Train every 1000 steps / 50 seconds
step_counter = 0

# Inputs to the Model
def _mainIntersection_queue():
    # vehicle detectors
    southwest = traci.lanearea.getLastStepVehicleNumber("e2_4") + traci.lanearea.getLastStepVehicleNumber("e2_5")
    southeast = traci.lanearea.getLastStepVehicleNumber("e2_6") + traci.lanearea.getLastStepVehicleNumber("e2_7")
    northeast = traci.lanearea.getLastStepVehicleNumber("e2_8")
    northwest = traci.lanearea.getLastStepVehicleNumber("e2_9") + traci.lanearea.getLastStepVehicleNumber("e2_10")
    
    # pedestrian detectors
    pedestrian = 0
    
    return (southwest, southeast, northeast, northwest, pedestrian)

def _swPedXing_queue():
    north = traci.lanearea.getLastStepVehicleNumber("e2_0") + traci.lanearea.getLastStepVehicleNumber("e2_1")
    south = traci.lanearea.getLastStepVehicleNumber("e2_4") + traci.lanearea.getLastStepVehicleNumber("e2_5")
    pedestrian = 0
    junction_id = "6401523012"
    
    _junctionSubscription(junction_id)
    junction_subscription = traci.junction.getContextSubscriptionResults(junction_id)
    
    if junction_subscription:
        for pid, data in junction_subscription.items():
            speed = data.get(traci.constants.VAR_SPEED, 0)
            if speed <= 0.5:
                pedestrian += 1 
            
    return (south, north, pedestrian)

def _sePedXing_queue():
    west = traci.lanearea.getLastStepVehicleNumber("e2_2") + traci.lanearea.getLastStepVehicleNumber("e2_3")
    east = traci.lanearea.getLastStepVehicleNumber("e2_6") + traci.lanearea.getLastStepVehicleNumber("e2_7")
    pedestrian = 0
    junction_id = "3285696417"
    
    _junctionSubscription(junction_id)
    junction_subscription = traci.junction.getContextSubscriptionResults(junction_id)
    
    if junction_subscription:
        for pid, data in junction_subscription.items():
            speed = data.get(traci.constants.VAR_SPEED, 0)
            if speed == 0:
                pedestrian += 1 
                   
    return (west, east, pedestrian)

# Calculate reward based on queue reduction
def calculate_reward(current_state, prev_state):
    if prev_state is None:
        return 0
    
    current_total = sum(current_state)
    prev_total = sum(prev_state)
    
    # Reward for reducing queue, penalty for increasing
    queue_diff = prev_total - current_total
    
    # Normalize reward
    if queue_diff > 0:
        reward = queue_diff * 2  # Bonus for reducing queue
    else:
        reward = queue_diff  # Penalty for increasing queue
    
    return reward

# Output of the model
def _mainIntersection_phase(action_index):
    global mainCurrentPhase, mainCurrentPhaseDuration
    
    mainCurrentPhase += 1
    mainCurrentPhase = mainCurrentPhase % 10
    
    # Apply action to adjust duration
    duration_adjustment = actionSpace[action_index]
    
    # Set phase
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", mainCurrentPhase)
    
    # Base duration
    if mainCurrentPhase == 2 or mainCurrentPhase == 4:
        base_duration = 15
    elif mainCurrentPhase % 2 == 0:
        base_duration = 30
    else:
        base_duration = 3
    
    # Apply adjustment and clamp to reasonable bounds
    mainCurrentPhaseDuration = max(5, min(60, base_duration + duration_adjustment))
    
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", mainCurrentPhaseDuration)
    
def _swPedXing_phase(action_index):
    global swCurrentPhase, swCurrentPhaseDuration
    
    swCurrentPhase += 1
    swCurrentPhase = swCurrentPhase % 4
    
    # Apply action to adjust duration
    duration_adjustment = actionSpace[action_index]
    
    # Set phase
    traci.trafficlight.setPhase("6401523012", swCurrentPhase)
    
    # Base duration
    if swCurrentPhase % 2 == 1:
        base_duration = 5
    elif swCurrentPhase % 2 == 0:
        base_duration = 30
    
    # Apply adjustment and clamp to reasonable bounds
    swCurrentPhaseDuration = max(5, min(60, base_duration + duration_adjustment))
    
    traci.trafficlight.setPhaseDuration("6401523012", swCurrentPhaseDuration)
    
def _sePedXing_phase(action_index):
    global seCurrentPhase, seCurrentPhaseDuration
    
    seCurrentPhase += 1
    seCurrentPhase = seCurrentPhase % 4
    
    # Apply action to adjust duration
    duration_adjustment = actionSpace[action_index]
    
    # Set phase
    traci.trafficlight.setPhase("3285696417", seCurrentPhase)
    
    # Base duration
    if seCurrentPhase % 2 == 1:
        base_duration = 5
    elif seCurrentPhase % 2 == 0:
        base_duration = 30
    
    # Apply adjustment and clamp to reasonable bounds
    seCurrentPhaseDuration = max(5, min(60, base_duration + duration_adjustment))
    
    traci.trafficlight.setPhaseDuration("3285696417", seCurrentPhaseDuration)


traci.start(Sumo_config)
detector_ids = traci.lanearea.getIDList()

# Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    
    # Main Intersection Agent Logic
    mainCurrentPhaseDuration -= stepLength
    if mainCurrentPhaseDuration <= 0:
        # Get current state
        mainCurrentState = np.array(_mainIntersection_queue(), dtype=np.float32)
        
        # Calculate reward from previous action
        mainReward = calculate_reward(mainCurrentState, mainPrevState)
        
        # Store experience if we have previous state/action
        if mainPrevState is not None and mainPrevAction is not None:
            # Check if episode is done (could add custom logic here)
            done = False
            mainIntersectionAgent.remember(mainPrevState, mainPrevAction, mainReward, mainCurrentState, done)
        
        # Choose new action
        mainActionIndex = mainIntersectionAgent.act(mainCurrentState)
        
        # Apply phase change with action
        _mainIntersection_phase(mainActionIndex)
        
        # Store current state and action for next iteration
        mainPrevState = mainCurrentState
        mainPrevAction = mainActionIndex
        
        print(f"Main Intersection - Queue: {sum(mainCurrentState)}, Reward: {mainReward}, Action: {actionSpace[mainActionIndex]}")
    
    # SW Pedestrian Crossing Agent Logic
    swCurrentPhaseDuration -= stepLength
    if swCurrentPhaseDuration <= 0:
        # Get current state
        swCurrentState = np.array(_swPedXing_queue(), dtype=np.float32)
        
        # Calculate reward
        swReward = calculate_reward(swCurrentState, swPrevState)
        
        # Store experience
        if swPrevState is not None and swPrevAction is not None:
            done = False
            swPedXingAgent.remember(swPrevState, swPrevAction, swReward, swCurrentState, done)
        
        # Choose new action
        swActionIndex = swPedXingAgent.act(swCurrentState)
        
        # Apply phase change
        _swPedXing_phase(swActionIndex)
        
        # Store state and action
        swPrevState = swCurrentState
        swPrevAction = swActionIndex
        
        print(f"SW Ped Crossing - Queue: {sum(swCurrentState)}, Reward: {swReward}, Action: {actionSpace[swActionIndex]}")
    
    # SE Pedestrian Crossing Agent Logic
    seCurrentPhaseDuration -= stepLength
    if seCurrentPhaseDuration <= 0:
        # Get current state
        seCurrentState = np.array(_sePedXing_queue(), dtype=np.float32)
        
        # Calculate reward
        seReward = calculate_reward(seCurrentState, sePrevState)
        
        # Store experience
        if sePrevState is not None and sePrevAction is not None:
            done = False
            sePedXingAgent.remember(sePrevState, sePrevAction, seReward, seCurrentState, done)
        
        # Choose new action
        seActionIndex = sePedXingAgent.act(seCurrentState)
        
        # Apply phase change
        _sePedXing_phase(seActionIndex)
        
        # Store state and action
        sePrevState = seCurrentState
        sePrevAction = seActionIndex
        
        print(f"SE Ped Crossing - Queue: {sum(seCurrentState)}, Reward: {seReward}, Action: {actionSpace[seActionIndex]}")
    
    # Periodic training (replay)
    if step_counter % TRAIN_FREQUENCY == 0:
        # Train main intersection agent
        if len(mainIntersectionAgent.memory) >= BATCH_SIZE:
            mainIntersectionAgent.replay(BATCH_SIZE)
            mainIntersectionAgent.epsilon = max(mainIntersectionAgent.epsilon_min, 
                                               mainIntersectionAgent.epsilon * mainIntersectionAgent.epsilon_decay_rate)
            print(f"Main Agent trained - Epsilon: {mainIntersectionAgent.epsilon:.4f}")
        
        # Train SW pedestrian crossing agent
        if len(swPedXingAgent.memory) >= BATCH_SIZE:
            swPedXingAgent.replay(BATCH_SIZE)
            swPedXingAgent.epsilon = max(swPedXingAgent.epsilon_min,
                                        swPedXingAgent.epsilon * swPedXingAgent.epsilon_decay_rate)
        
        # Train SE pedestrian crossing agent
        if len(sePedXingAgent.memory) >= BATCH_SIZE:
            sePedXingAgent.replay(BATCH_SIZE)
            sePedXingAgent.epsilon = max(sePedXingAgent.epsilon_min,
                                        sePedXingAgent.epsilon * sePedXingAgent.epsilon_decay_rate)
    
    traci.simulationStep()

# Save trained models
print("Saving trained models...")
mainIntersectionAgent.save()
swPedXingAgent.save()
sePedXingAgent.save()
print("Models saved successfully!")

traci.close()