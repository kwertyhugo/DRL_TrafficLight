import os
import sys 
import traci
import numpy as np
import csv
from keras.utils import to_categorical

from models.DQN import DQNAgent as dqn

# Select DRL Agent
mainIntersectionAgent = dqn(state_size=15, action_size=7, memory_size=200, gamma=0.95, epsilon=0, epsilon_min=0, name='ReLU_DQNAgent')
swPedXingAgent = dqn(state_size=7, action_size=7, memory_size=200, gamma=0.95, epsilon=0, epsilon_min=0, name='SW_PedXing_Agent')
sePedXingAgent = dqn(state_size=7, action_size=7, memory_size=200, gamma=0.95, epsilon=0, epsilon_min=0, name='SE_PedXing_Agent')

mainIntersectionAgent.load()
swPedXingAgent.load()
sePedXingAgent.load()

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Sumo_config = [
#     'sumo-gui',
#     '-c', r'Olivarez_traci\map.sumocfg',
#     '--step-length', '0.05',
#     '--delay', '100',
#     '--lateral-resolution', '0.1',
#     '--end', '360',
#     '--statistic-output', r'Olivarez_traci\SD_DQN_stats.xml',
#     '--tripinfo-output', r'Olivarez_traci\SD_DQN_stats.xml'
# ]

Sumo_config = [
    'sumo-gui',
    '-c', r'Olivarez_traci\map.sumocfg',
    '--step-length', '0.05',
    '--delay', '100',
    '--lateral-resolution', '0.1',
    '--end', '360',
    '--statistic-output', r'Olivarez_traci\SD_none_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\SD_none_stats.xml'
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
step_counter = 0

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
            traci.constants.CMD_GET_VEHICLE_VARIABLE,  # We want vehicle data
            3,  # Radius (0 means only vehicles *inside* the detector)
            vehicle_vars
        )

# Inputs to the Model
def _weighted_waits(detector_id):
    sumWait = 0
    # This ONE call gets all vehicle data (Type and Wait Time)
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)

    if not vehicle_data:
        return 0

    # Loop over the results dictionary (fast, all in Python)
    for data in vehicle_data.values():
        type = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        
        # Your custom weighting logic
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
    # vehicle detectors
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
def calculate_reward(current_state):
    if current_state is None:
        return 0
    
    current_total = sum(current_state)
    return -current_total

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
_subscribe_all_detectors()
_junctionSubscription("cluster_295373794_3477931123_7465167861")
_junctionSubscription("6401523012")
_junctionSubscription("3285696417")

detector_ids = traci.lanearea.getIDList()

# Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    
    mainCurrentPhaseDuration -= stepLength
    if mainCurrentPhaseDuration <= 0:
        state_list = np.array(_mainIntersection_queue())
        normalized_state_list = state_list/1000
        state_vector = to_categorical(mainCurrentPhase, num_classes=10).flatten()
        mainCurrentState = np.concatenate([normalized_state_list, state_vector]).astype(np.float32)
        mainReward = calculate_reward(normalized_state_list)
        #mainActionIndex = mainIntersectionAgent.act(mainCurrentState) #For with DQN agent
        mainActionIndex = 4 #For with no agent
        _mainIntersection_phase(mainActionIndex)
    
    swCurrentPhaseDuration -= stepLength
    if swCurrentPhaseDuration <= 0:
        state_list = np.array(_swPedXing_queue())
        normalized_state_list = state_list/1000
        state_vector = to_categorical(swCurrentPhase, num_classes=4).flatten()
        swCurrentState = np.concatenate([normalized_state_list, state_vector]).astype(np.float32)
        swReward = calculate_reward(normalized_state_list)
        #swActionIndex = swPedXingAgent.act(swCurrentState) #For with DQN agent
        swActionIndex = 4 #For with no agent
        _swPedXing_phase(swActionIndex)
    
    seCurrentPhaseDuration -= stepLength
    if seCurrentPhaseDuration <= 0:
        state_list = np.array(_sePedXing_queue())
        normalized_state_list = state_list/1000
        state_vector = to_categorical(seCurrentPhase, num_classes=4).flatten()
        seCurrentState = np.concatenate([normalized_state_list, state_vector]).astype(np.float32)
        seReward = calculate_reward(normalized_state_list)
        #seActionIndex = sePedXingAgent.act(seCurrentState) #For with DQN agent
        seActionIndex = 4 #For with no agent
        _sePedXing_phase(seActionIndex)
        
    traci.simulationStep()

traci.close()