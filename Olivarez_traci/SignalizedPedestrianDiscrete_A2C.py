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
from keras.models import load_model # Import load_model

# === AGENT INITIALIZATION - STATE SIZES ARE CORRECT FOR THIS COUPLED MODEL ===
mainIntersectionAgent = a2c(
    state_size=26, action_size=7,  # 8 queues + 10 + 4 + 4 phases = 26
    gamma=0.99,
    learning_rate=0.0005,
    entropy_coef=0.08,
    value_coef=0.5,
    max_grad_norm=1.0,
    name='A2C_Main_Agent'
)

swPedXingAgent = a2c(
    state_size=23, action_size=7,  # 5 queues + 10 + 4 + 4 phases = 23
    gamma=0.99,
    learning_rate=0.0005,
    entropy_coef=0.08,
    value_coef=0.5,
    max_grad_norm=1.0,
    name='A2C_SW_PedXing_Agent'
)

sePedXingAgent = a2c(
    state_size=23, action_size=7,  # 5 queues + 10 + 4 + 4 phases = 23
    gamma=0.99,
    learning_rate=0.0005,
    entropy_coef=0.08,
    value_coef=0.5,
    max_grad_norm=1.0,
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
    '-c', 'Olivarez_traci\signalizedPed.sumocfg',
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

# Episode counters
mainEpisodeNumber = 0
swEpisodeNumber = 0
seEpisodeNumber = 0

if CONTINUE_TRAINING:
    def get_last_episode(filename):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
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

# This function is correct.
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

# This function is correct.
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

# This function is correct.
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

# This reward function is perfect.
def calculate_reward(current_state_queues):
    if current_state_queues is None:
        return 0.0
    
    current_total = sum(current_state_queues)
    normalized_queue = current_total / 1000.0
    reward = -normalized_queue
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
print("Starting SYNCHRONOUS A2C Training")
print("=" * 70)
print("Key Fix: All agents act ONLY when the MAIN agent acts.")
print("This creates a stable, synchronous environment.")
print("  - Main Agent is the 'master' clock.")
print("  - SW and SE Agents are 'slaves' (act at the same time).")
print("  - All agents train at the same time (when Main ep ends).")
print("=" * 70)

# Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    
    ### --- CHANGED --- ###
    # The timers for SW and SE are now ONLY ticked.
    # They no longer trigger any actions.
    mainCurrentPhaseDuration -= stepLength
    swCurrentPhaseDuration -= stepLength
    seCurrentPhaseDuration -= stepLength
    
    # === SYNCHRONOUS DECISION BLOCK (Master Clock) ===
    # All logic is now inside the MAIN agent's timer.
    # This is the ONLY place agents observe, act, and store.
    if mainCurrentPhaseDuration <= 0:
        
        # --- 1. GET ALL STATES (Observe) ---
        main_phase = to_categorical(mainCurrentPhase, num_classes=10).flatten()
        swPed_phase = to_categorical(swCurrentPhase, num_classes=4).flatten()
        sePed_phase = to_categorical(seCurrentPhase, num_classes=4).flatten()
        
        # MAIN AGENT
        main_queue = np.array(_mainIntersection_queue())
        normalized_main_queue = main_queue / 1000.0
        mainCurrentState = np.concatenate([
            normalized_main_queue, main_phase, swPed_phase, sePed_phase
        ]).astype(np.float32)

        # SW AGENT
        swPed_queue = np.array(_swPedXing_queue())
        normalized_swPed_queue = swPed_queue / 1000.0
        swCurrentState = np.concatenate([
            normalized_swPed_queue, main_phase, swPed_phase, sePed_phase
        ]).astype(np.float32)
        
        # SE AGENT
        sePed_queue = np.array(_sePedXing_queue())
        normalized_sePed_queue = sePed_queue / 1000.0
        seCurrentState = np.concatenate([
            normalized_sePed_queue, main_phase, swPed_phase, sePed_phase
        ]).astype(np.float32)
        
        # --- 2. CALCULATE REWARDS (based on current state) ---
        mainReward = calculate_reward(main_queue)
        swReward = calculate_reward(swPed_queue)
        seReward = calculate_reward(sePed_queue)
        
        # --- 3. STORE TRANSITIONS (for previous step) ---
        if mainPrevState is not None:
            mainIntersectionAgent.store_transition(mainPrevState, mainPrevAction, mainReward)
            swPedXingAgent.store_transition(swPrevState, swPrevAction, swReward)
            sePedXingAgent.store_transition(sePrevState, sePrevAction, seReward)
        
        # --- 4. CHOOSE NEW ACTIONS (Act) ---
        mainActionIndex = mainIntersectionAgent.act(mainCurrentState, training=True)
        swActionIndex = swPedXingAgent.act(swCurrentState, training=True)
        seActionIndex = sePedXingAgent.act(seCurrentState, training=True)

        # --- 5. APPLY ACTIONS & UPDATE PHASES (Synchronized) ---
        _mainIntersection_phase(mainActionIndex)
        _swPedXing_phase(swActionIndex)
        _sePedXing_phase(seActionIndex)
        
        # --- 6. TRAIN (if episode ends for master) ---
        if mainCurrentPhase == 0 and mainPrevState is not None:
            mainEpisodeNumber += 1
            swEpisodeNumber += 1
            seEpisodeNumber += 1
            
            # Train MAIN
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
            
            # Train SW
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
            
            # Train SE
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
        
        # --- 7. SAVE CURRENT STATE/ACTIONS for next loop ---
        mainPrevState = mainCurrentState
        mainPrevAction = mainActionIndex
        swPrevState = swCurrentState
        swPrevAction = swActionIndex
        sePrevState = seCurrentState
        sePrevAction = seActionIndex
    
    
    ### --- REMOVED --- ###
    # The SW agent's decision block is GONE.
    # if swCurrentPhaseDuration <= 0:
    #    ... (All this logic was moved into the main block) ...
    
    ### --- REMOVED --- ###
    # The SE agent's decision block is GONE.
    # if seCurrentPhaseDuration <= 0:
    #    ... (All this logic was moved into the main block) ...
    
    
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