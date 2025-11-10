import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create output directories if they don't exist

os.makedirs('./Olivarez_traci/output_A2C', exist_ok=True)
os.makedirs('./Olivarez_traci/models_A2C', exist_ok=True)
import traci
import numpy as np
import csv
from keras.utils import to_categorical
from models.A2C import A2CAgent as a2c

# === AGENT INITIALIZATION - Focus on stable value function ===
mainIntersectionAgent = a2c(
    state_size=26, action_size=7, 
    gamma=0.99,
    learning_rate=0.0005,    # ✅ Balanced (not 0.007!)
    entropy_coef=0.08,       # ✅ Keep exploration high
    value_coef=0.5,
    max_grad_norm=1.0,       # ✅ Slightly higher for stability
    name='A2C_Main_Agent'
)

swPedXingAgent = a2c(
    state_size=23, action_size=7, 
    gamma=0.99,
    learning_rate=0.0005,    # ✅ Match main agent
    entropy_coef=0.08,       # ✅ Match main agent
    value_coef=0.5,
    max_grad_norm=1.0,       # ✅ Match main agent
    name='A2C_SW_PedXing_Agent'
)

sePedXingAgent = a2c(
    state_size=23, action_size=7, 
    gamma=0.99,
    learning_rate=0.0005,    # ✅ Match main agent
    entropy_coef=0.08,       # ✅ Match main agent
    value_coef=0.5,
    max_grad_norm=1.0,       # ✅ Match main agent
    name='A2C_SE_PedXing_Agent'
)

# === LOAD EXISTING MODELS IF THEY EXIST ===
CONTINUE_TRAINING = True  # Set to False to start fresh

if CONTINUE_TRAINING:
    try:
        print("\n" + "=" * 70)
        print("Attempting to load existing models for continued training...")
        print("=" * 70)
        
        main_model_path = './Olivarez_traci/models_A2C/A2C_Main_Agent.keras'
        sw_model_path = './Olivarez_traci/models_A2C/A2C_SW_PedXing_Agent.keras'
        se_model_path = './Olivarez_traci/models_A2C/A2C_SE_PedXing_Agent.keras'
        
        if os.path.exists(main_model_path):
            # Your A2CAgent.load() doesn't take filepath, so load directly
            mainIntersectionAgent.model = load_model(main_model_path)
            print(f"✓ Loaded Main Agent from {main_model_path}")
        else:
            print(f"⚠ Main Agent model not found - starting fresh")
            
        if os.path.exists(sw_model_path):
            swPedXingAgent.model = load_model(sw_model_path)
            print(f"✓ Loaded SW Agent from {sw_model_path}")
        else:
            print(f"⚠ SW Agent model not found - starting fresh")
            
        if os.path.exists(se_model_path):
            sePedXingAgent.model = load_model(se_model_path)
            print(f"✓ Loaded SE Agent from {se_model_path}")
        else:
            print(f"⚠ SE Agent model not found - starting fresh")
            
        print("=" * 70)
        
    except Exception as e:
        print(f"⚠ Error loading models: {e}")
        print("Starting with fresh models instead.")
else:
    print("\n" + "=" * 70)
    print("CONTINUE_TRAINING = False - Starting with fresh random weights")
    print("=" * 70)

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

# Episode counters - will be loaded from history if continuing
mainEpisodeNumber = 0
swEpisodeNumber = 0
seEpisodeNumber = 0

# Load last episode numbers if continuing training
if CONTINUE_TRAINING:
    def get_last_episode(filename):
        """Get the last episode number from CSV file"""
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Has data beyond header
                        last_line = lines[-1].strip()
                        if last_line:
                            return int(last_line.split(',')[0])
            except:
                pass
        return 0
    
    mainEpisodeNumber = get_last_episode('./Olivarez_traci/output_A2C/a2c_main_agent_history.csv')
    swEpisodeNumber = get_last_episode('./Olivarez_traci/output_A2C/a2c_sw_agent_history.csv')
    seEpisodeNumber = get_last_episode('./Olivarez_traci/output_A2C/a2c_se_agent_history.csv')
    
    print(f"Continuing from:")
    print(f"  Main Agent: Episode {mainEpisodeNumber}")
    print(f"  SW Agent: Episode {swEpisodeNumber}")
    print(f"  SE Agent: Episode {seEpisodeNumber}")
    print("=" * 70)

mainPrevState = None
mainPrevAction = None
swPrevState = None
swPrevAction = None
sePrevState = None
sePrevAction = None

# Data storage
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

def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id, traci.constants.CMD_GET_PERSON_VARIABLE,
        10.0, [traci.constants.VAR_WAITING_TIME]
    )
    
def _subscribe_all_detectors():
    detector_ids = [
        "e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10",
        "e2_0", "e2_1", "e2_2", "e2_3"
    ]
    vehicle_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(
            det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_vars
        )

def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
    if not vehicle_data:
        return 0
    
    weight_map = {"car": 1.0, "jeep": 1.5, "bus": 2.2, 
                  "truck": 2.5, "motorcycle": 0.3, "tricycle": 0.5}
    
    for data in vehicle_data.values():
        vtype = data.get(traci.constants.VAR_TYPE, "car")
        wait = data.get(traci.constants.VAR_WAITING_TIME, 0)
        sumWait += wait * weight_map.get(vtype, 1.0)
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
    junction_sub = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
    if junction_sub:
        for pid, data in junction_sub.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return [e2_4, e2_5, e2_6, e2_7, e2_8, e2_9, e2_10, pedestrian]

def _swPedXing_queue():
    e2_0 = _weighted_waits("e2_0")
    e2_1 = _weighted_waits("e2_1")
    e2_4 = _weighted_waits("e2_4")
    e2_5 = _weighted_waits("e2_5")
    
    pedestrian = 0
    junction_sub = traci.junction.getContextSubscriptionResults("6401523012")
    if junction_sub:
        for pid, data in junction_sub.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return [e2_0, e2_1, e2_4, e2_5, pedestrian]

def _sePedXing_queue():
    e2_2 = _weighted_waits("e2_2")
    e2_3 = _weighted_waits("e2_3")
    e2_6 = _weighted_waits("e2_6")
    e2_7 = _weighted_waits("e2_7")
    
    pedestrian = 0
    junction_sub = traci.junction.getContextSubscriptionResults("3285696417")
    if junction_sub:
        for pid, data in junction_sub.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return [e2_2, e2_3, e2_6, e2_7, pedestrian]

def calculate_reward(current_state):
    """
    KEY INSIGHT: The reward SHOULD vary naturally with traffic!
    The issue isn't the reward magnitude, it's that we need CONSISTENT scaling.
    
    Solution: Use the SAME normalization for rewards as we do for states.
    This keeps everything in the same scale the network was designed for.
    """
    if current_state is None:
        print("ERROR: STATE UNDETECTED")
        return 0.0
    
    current_total = sum(current_state)
    
    # Use SAME normalization as state: divide by 1000
    # This keeps reward in same scale as what the network "sees"
    normalized_queue = current_total / 1000.0
    
    # Simple negative reward (agent minimizes waiting time)
    reward = -normalized_queue
    
    # Clip to reasonable range to prevent extreme outliers
    # Allow range of [-10, 0] which handles queues up to 10,000
    reward = np.clip(reward, -10.0, 0.0)
    
    return reward

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
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        for i in range(len(episodes)):
            writer.writerow([episodes[i], rewards[i], actor_losses[i], critic_losses[i], entropies[i]])

# Start SUMO
traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("cluster_295373794_3477931123_7465167861")
_junctionSubscription("6401523012")
_junctionSubscription("3285696417")

print("=" * 70)
print("Starting A2C training with CONSISTENT REWARD SCALING")
print("=" * 70)
print("Key Fix: Rewards use SAME normalization as states (÷1000)")
print("This keeps the critic's value predictions in a learnable range")
print("  - State normalization: queue ÷ 1000")
print("  - Reward calculation: -queue ÷ 1000, clipped to [-10, 0]")
print("  - Learning rate: 0.0001 (balanced)")
print("  - Gradient clipping: 0.5 (prevents explosions)")
print("=" * 70)

# Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    
    # === MAIN INTERSECTION ===
    mainCurrentPhaseDuration -= stepLength
    if mainCurrentPhaseDuration <= 0:
        main_queue = np.array(_mainIntersection_queue())
        normalized_main_queue = main_queue / 1000.0  # For state
        
        main_phase = to_categorical(mainCurrentPhase, num_classes=10).flatten()
        swPed_phase = to_categorical(swCurrentPhase, num_classes=4).flatten()
        sePed_phase = to_categorical(seCurrentPhase, num_classes=4).flatten()
        
        mainCurrentState = np.concatenate([
            normalized_main_queue, main_phase, swPed_phase, sePed_phase
        ]).astype(np.float32)
        
        # Use consistent normalization for reward
        mainReward = calculate_reward(main_queue)
        
        if mainPrevState is not None and mainPrevAction is not None:
            mainIntersectionAgent.store_transition(mainPrevState, mainPrevAction, mainReward)
        
        mainActionIndex = mainIntersectionAgent.act(mainCurrentState, training=True)
        _mainIntersection_phase(mainActionIndex)
        
        if mainCurrentPhase == 0 and mainPrevState is not None:
            mainEpisodeNumber += 1
            actor_loss, critic_loss, entropy, total_reward = mainIntersectionAgent.train_on_episode()
            
            main_actor_loss_history.append(actor_loss)
            main_critic_loss_history.append(critic_loss)
            main_entropy_history.append(entropy)
            main_reward_history.append(total_reward)
            main_episode_steps.append(mainEpisodeNumber)
            
            raw_queue_total = sum(main_queue)
            print(f"[MAIN Ep {mainEpisodeNumber:3d}] EpisodeRwd: {total_reward:7.3f} | "
                  f"ALoss: {actor_loss:7.4f} CLoss: {critic_loss:9.2f} | "
                  f"Ent: {entropy:5.3f} | AvgQueue: {raw_queue_total/8:7.1f}")
        
        mainPrevState = mainCurrentState
        mainPrevAction = mainActionIndex
    
    # === SW PEDESTRIAN CROSSING ===
    swCurrentPhaseDuration -= stepLength
    if swCurrentPhaseDuration <= 0:
        swPed_queue = np.array(_swPedXing_queue())
        normalized_swPed_queue = swPed_queue / 1000.0
        
        main_phase = to_categorical(mainCurrentPhase, num_classes=10).flatten()
        swPed_phase = to_categorical(swCurrentPhase, num_classes=4).flatten()
        sePed_phase = to_categorical(seCurrentPhase, num_classes=4).flatten()
        
        swCurrentState = np.concatenate([
            normalized_swPed_queue, main_phase, swPed_phase, sePed_phase
        ]).astype(np.float32)
        
        swReward = calculate_reward(swPed_queue)
        
        if swPrevState is not None and swPrevAction is not None:
            swPedXingAgent.store_transition(swPrevState, swPrevAction, swReward)
        
        swActionIndex = swPedXingAgent.act(swCurrentState, training=True)
        _swPedXing_phase(swActionIndex)
        
        if swCurrentPhase == 0 and swPrevState is not None:
            swEpisodeNumber += 1
            actor_loss, critic_loss, entropy, total_reward = swPedXingAgent.train_on_episode()
            
            sw_actor_loss_history.append(actor_loss)
            sw_critic_loss_history.append(critic_loss)
            sw_entropy_history.append(entropy)
            sw_reward_history.append(total_reward)
            sw_episode_steps.append(swEpisodeNumber)
            
            raw_queue_total = sum(swPed_queue)
            print(f"[SW   Ep {swEpisodeNumber:3d}] EpisodeRwd: {total_reward:7.3f} | "
                  f"ALoss: {actor_loss:7.4f} CLoss: {critic_loss:9.2f} | "
                  f"Ent: {entropy:5.3f} | AvgQueue: {raw_queue_total/5:7.1f}")
        
        swPrevState = swCurrentState
        swPrevAction = swActionIndex
    
    # === SE PEDESTRIAN CROSSING ===
    seCurrentPhaseDuration -= stepLength
    if seCurrentPhaseDuration <= 0:
        sePed_queue = np.array(_sePedXing_queue())
        normalized_sePed_queue = sePed_queue / 1000.0
        
        main_phase = to_categorical(mainCurrentPhase, num_classes=10).flatten()
        swPed_phase = to_categorical(swCurrentPhase, num_classes=4).flatten()
        sePed_phase = to_categorical(seCurrentPhase, num_classes=4).flatten()
        
        seCurrentState = np.concatenate([
            normalized_sePed_queue, main_phase, swPed_phase, sePed_phase
        ]).astype(np.float32)
        
        seReward = calculate_reward(sePed_queue)
        
        if sePrevState is not None and sePrevAction is not None:
            sePedXingAgent.store_transition(sePrevState, sePrevAction, seReward)
        
        seActionIndex = sePedXingAgent.act(seCurrentState, training=True)
        _sePedXing_phase(seActionIndex)
        
        if seCurrentPhase == 0 and sePrevState is not None:
            seEpisodeNumber += 1
            actor_loss, critic_loss, entropy, total_reward = sePedXingAgent.train_on_episode()
            
            se_actor_loss_history.append(actor_loss)
            se_critic_loss_history.append(critic_loss)
            se_entropy_history.append(entropy)
            se_reward_history.append(total_reward)
            se_episode_steps.append(seEpisodeNumber)
            
            raw_queue_total = sum(sePed_queue)
            print(f"[SE   Ep {seEpisodeNumber:3d}] EpisodeRwd: {total_reward:7.3f} | "
                f"ALoss: {actor_loss:7.4f} CLoss: {critic_loss:9.2f} | "
                f"Ent: {entropy:5.3f} | AvgQueue: {raw_queue_total/5:7.1f}")
        
        sePrevState = seCurrentState
        sePrevAction = seActionIndex
    
    traci.simulationStep()

print("\n" + "=" * 70)
print("Simulation complete! Saving trained models...")
mainIntersectionAgent.save()
swPedXingAgent.save()
sePedXingAgent.save()

print("Saving training history...")
save_history('./Olivarez_traci/output_A2C/a2c_main_agent_history.csv', 
            ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'], 
            main_episode_steps, main_reward_history, main_actor_loss_history, 
            main_critic_loss_history, main_entropy_history)
            
save_history('./Olivarez_traci/output_A2C/a2c_sw_agent_history.csv', 
            ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'], 
            sw_episode_steps, sw_reward_history, sw_actor_loss_history, 
            sw_critic_loss_history, sw_entropy_history)
            
save_history('./Olivarez_traci/output_A2C/a2c_se_agent_history.csv', 
            ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'], 
            se_episode_steps, se_reward_history, se_actor_loss_history, 
            se_critic_loss_history, se_entropy_history)

print(f"\nTraining Summary:")
print(f"   Main Intersection: {mainEpisodeNumber} episodes")
print(f"   SW Ped Crossing: {swEpisodeNumber} episodes")
print(f"   SE Ped Crossing: {seEpisodeNumber} episodes")

traci.close()