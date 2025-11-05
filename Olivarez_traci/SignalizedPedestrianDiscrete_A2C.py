import os
import sys

# Add parent directory to path so we can import from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traci
import numpy as np
import csv
from keras.utils import to_categorical

from models.A2C import A2CAgent as a2c

# Select A2C Agents with conservative hyperparameters for stability
mainIntersectionAgent = a2c(state_size=15, action_size=7, gamma=0.95, learning_rate=0.0003, 
                        entropy_coef=0.2, value_coef=0.5, max_grad_norm=0.5, name='A2C_Main_Agent')
swPedXingAgent = a2c(state_size=7, action_size=7, gamma=0.95, learning_rate=0.0003,
                    entropy_coef=0.2, value_coef=0.5, max_grad_norm=0.5, name='A2C_SW_PedXing_Agent')
sePedXingAgent = a2c(state_size=7, action_size=7, gamma=0.95, learning_rate=0.0003,
                    entropy_coef=0.2, value_coef=0.5, max_grad_norm=0.5, name='A2C_SE_PedXing_Agent')

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
    '-c', 'Olivarez_traci/map.sumocfg',
    '--step-length', '0.05',
    '--delay', '0',
    '--lateral-resolution', '0.1'
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

# Episode tracking
mainEpisodeNumber = 0
swEpisodeNumber = 0
seEpisodeNumber = 0

# Store state and action from previous step for each agent
mainPrevState = None
mainPrevAction = None
mainPrevQueue = None

swPrevState = None
swPrevAction = None
swPrevQueue = None

sePrevState = None
sePrevAction = None
sePrevQueue = None

# Data storage for plotting
main_actor_loss_history = []
main_critic_loss_history = []
main_entropy_history = []
main_reward_history = []
main_episode_steps = []

sw_actor_loss_history = []
sw_critic_loss_history = []
sw_entropy_history = []
sw_reward_history = []
sw_episode_steps = []

se_actor_loss_history = []
se_critic_loss_history = []
se_entropy_history = []
se_reward_history = []
se_episode_steps = []

# Object Context Subscription in SUMO
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
        
        weight_map = {
            "car": 1.0,
            "jeep": 1.5,
            "bus": 2.2,
            "truck": 2.5,
            "motorcycle": 0.3,
            "tricycle": 0.5
        }
        
        weight = weight_map.get(type, 1.0)
        sumWait += waitTime * weight
        
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

# Reward function - simpler and more stable
def calculate_reward(current_queue_total, prev_queue_total):
    """
    Simple reward based on queue reduction.
    """
    if prev_queue_total is None:
        return 0.0
    
    # Positive reward for reducing queue
    reduction = (prev_queue_total - current_queue_total)
    
    # Scale down for stability
    reward = reduction * 0.01
    
    # Clip
    return np.clip(reward, -1.0, 1.0)

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

def save_history(filename, headers, episodes, rewards, actor_losses, critic_losses, entropies):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(len(episodes)):
            writer.writerow([episodes[i], rewards[i], actor_losses[i], 
                           critic_losses[i], entropies[i]])

# Start SUMO
traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("cluster_295373794_3477931123_7465167861")
_junctionSubscription("6401523012")
_junctionSubscription("3285696417")

print("Starting A2C training for traffic light control...")
print("Training occurs at the end of each complete phase cycle (episode)")
print("Reward: queue reduction (simpler is better)")
print("=" * 70)

# Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    
    # ===== MAIN INTERSECTION AGENT =====
    mainCurrentPhaseDuration -= stepLength
    
    if mainCurrentPhaseDuration <= 0:
        # Get current queue and state
        queue = _mainIntersection_queue()
        queue_total = sum(queue)
        phase_vector = to_categorical(mainCurrentPhase, num_classes=10).flatten()
        current_state = np.concatenate([queue, phase_vector]).astype(np.float32)
        
        # If we have a previous state, calculate reward and store transition
        if mainPrevState is not None and mainPrevAction is not None:
            reward = calculate_reward(queue_total, mainPrevQueue)
            mainIntersectionAgent.store_transition(mainPrevState, mainPrevAction, reward)
        
        # Choose next action
        action = mainIntersectionAgent.act(current_state, training=True)
        
        # Apply the action
        _mainIntersection_phase(action)
        
        # Check if episode is complete (cycled back to phase 0)
        if mainCurrentPhase == 0 and mainPrevState is not None:
            mainEpisodeNumber += 1
            
            # Train on complete episode
            actor_loss, critic_loss, entropy, total_reward = mainIntersectionAgent.train_on_episode()
            
            # Log results
            main_actor_loss_history.append(actor_loss)
            main_critic_loss_history.append(critic_loss)
            main_entropy_history.append(entropy)
            main_reward_history.append(total_reward)
            main_episode_steps.append(mainEpisodeNumber)
            
            print(f"[MAIN Ep {mainEpisodeNumber:3d}] Reward: {total_reward:7.2f} | "
                  f"Loss: {actor_loss:6.4f}/{critic_loss:7.2f} | "
                  f"Ent: {entropy:5.3f} | Queue: {queue_total:7.2f}")
        
        # Store current state and action for next iteration
        mainPrevState = current_state
        mainPrevAction = action
        mainPrevQueue = queue_total
    
    # ===== SW PEDESTRIAN CROSSING AGENT =====
    swCurrentPhaseDuration -= stepLength
    
    if swCurrentPhaseDuration <= 0:
        queue = _swPedXing_queue()
        queue_total = sum(queue)
        phase_vector = to_categorical(swCurrentPhase, num_classes=4).flatten()
        current_state = np.concatenate([queue, phase_vector]).astype(np.float32)
        
        if swPrevState is not None and swPrevAction is not None:
            reward = calculate_reward(queue_total, swPrevQueue)
            swPedXingAgent.store_transition(swPrevState, swPrevAction, reward)
        
        action = swPedXingAgent.act(current_state, training=True)
        _swPedXing_phase(action)
        
        if swCurrentPhase == 0 and swPrevState is not None:
            swEpisodeNumber += 1
            
            actor_loss, critic_loss, entropy, total_reward = swPedXingAgent.train_on_episode()
            
            sw_actor_loss_history.append(actor_loss)
            sw_critic_loss_history.append(critic_loss)
            sw_entropy_history.append(entropy)
            sw_reward_history.append(total_reward)
            sw_episode_steps.append(swEpisodeNumber)
            
            print(f"[SW   Ep {swEpisodeNumber:3d}] Reward: {total_reward:7.2f} | "
                  f"Loss: {actor_loss:6.4f}/{critic_loss:7.2f} | "
                  f"Ent: {entropy:5.3f} | Queue: {queue_total:7.2f}")
        
        swPrevState = current_state
        swPrevAction = action
        swPrevQueue = queue_total
    
    # ===== SE PEDESTRIAN CROSSING AGENT =====
    seCurrentPhaseDuration -= stepLength
    
    if seCurrentPhaseDuration <= 0:
        queue = _sePedXing_queue()
        queue_total = sum(queue)
        phase_vector = to_categorical(seCurrentPhase, num_classes=4).flatten()
        current_state = np.concatenate([queue, phase_vector]).astype(np.float32)
        
        if sePrevState is not None and sePrevAction is not None:
            reward = calculate_reward(queue_total, sePrevQueue)
            sePedXingAgent.store_transition(sePrevState, sePrevAction, reward)
        
        action = sePedXingAgent.act(current_state, training=True)
        _sePedXing_phase(action)
        
        if seCurrentPhase == 0 and sePrevState is not None:
            seEpisodeNumber += 1
            
            actor_loss, critic_loss, entropy, total_reward = sePedXingAgent.train_on_episode()
            
            se_actor_loss_history.append(actor_loss)
            se_critic_loss_history.append(critic_loss)
            se_entropy_history.append(entropy)
            se_reward_history.append(total_reward)
            se_episode_steps.append(seEpisodeNumber)
            
            print(f"[SE   Ep {seEpisodeNumber:3d}] Reward: {total_reward:7.2f} | "
                  f"Loss: {actor_loss:6.4f}/{critic_loss:7.2f} | "
                  f"Ent: {entropy:5.3f} | Queue: {queue_total:7.2f}")
        
        sePrevState = current_state
        sePrevAction = action
        sePrevQueue = queue_total
    
    traci.simulationStep()

# Save trained models
print("\n" + "=" * 70)
print("Simulation complete! Saving trained models...")
mainIntersectionAgent.save()
swPedXingAgent.save()
sePedXingAgent.save()

# Save training history
print("Saving training history...")
save_history('a2c_main_agent_history.csv', 
            ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'], 
            main_episode_steps, main_reward_history, main_actor_loss_history, 
            main_critic_loss_history, main_entropy_history)
            
save_history('a2c_sw_agent_history.csv', 
            ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'], 
            sw_episode_steps, sw_reward_history, sw_actor_loss_history, 
            sw_critic_loss_history, sw_entropy_history)
            
save_history('a2c_se_agent_history.csv', 
            ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'], 
            se_episode_steps, se_reward_history, se_actor_loss_history, 
            se_critic_loss_history, se_entropy_history)

print(f"\nTraining Summary:")
print(f"  Main Intersection: {mainEpisodeNumber} episodes")
print(f"  SW Ped Crossing: {swEpisodeNumber} episodes")
print(f"  SE Ped Crossing: {seEpisodeNumber} episodes")

traci.close()