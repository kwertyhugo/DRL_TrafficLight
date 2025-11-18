import os
import sys
import csv
import numpy as np
import traci

# === PATH SETUP ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.makedirs('./Olivarez_traci/output_A2C_Baseline', exist_ok=True)
os.makedirs('./Olivarez_traci/models_A2C_Baseline', exist_ok=True)

from keras.utils import to_categorical
from keras.models import load_model
from models.A2C import A2CAgent as a2c

# === AGENT INITIALIZATION ===
mainIntersectionAgent = a2c(
    state_size=13,           # 8 queues + 5 phases
    action_size=7,           # 7 possible adjustments (matching signalized version)
    gamma=0.99,
    learning_rate=0.0005,    # matching signalized version
    entropy_coef=0.08,       # matching signalized version
    value_coef=0.5,
    max_grad_norm=1.0,
    name='A2C_Baseline_Agent'
)

# === CONTINUE TRAINING FLAG ===
CONTINUE_TRAINING = True  # set to True to load previous model

if CONTINUE_TRAINING:
    try:
        print("\n" + "=" * 70)
        print("Attempting to load existing model for continued training...")
        print("=" * 70)
        
        model_path = './Olivarez_traci/models_A2C_Baseline/A2C_Baseline_Agent.keras'
        if os.path.exists(model_path):
            mainIntersectionAgent.model = load_model(model_path)
            print(f"✓ Loaded model from {model_path}")
        else:
            print("⚠ Model not found. Starting fresh.")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        print("Starting with fresh weights.")
else:
    print("\n" + "=" * 70)
    print("Starting new training session (fresh weights)")
    print("=" * 70)

# === SUMO SETUP ===
if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

Sumo_config = [
    'sumo-gui',
    '-c', r'Olivarez_traci\baselinePed.sumocfg',
    '--step-length', '0.05',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_A2C_Baseline\A2C_Baseline_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_A2C_Baseline\A2C_Baseline_trips.xml'
]

# === SIMULATION VARIABLES ===
stepLength = 0.05
mainCurrentPhase = 0
mainCurrentPhaseDuration = 30
actionSpace = (-15, -10, -5, 0, 5, 10, 15)  # matching signalized version

# Episode counter
mainEpisodeNumber = 0

# Load episode number if continuing training
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
    
    mainEpisodeNumber = get_last_episode('./Olivarez_traci/output_A2C_Baseline/a2c_baseline_agent_history.csv')
    print(f"Continuing from Episode {mainEpisodeNumber}")
    print("=" * 70)

mainPrevState = None
mainPrevAction = None

# Data storage
main_actor_loss_history = []
main_critic_loss_history = []
main_entropy_history = []
main_reward_history = []
main_episode_steps = []

# === SUBSCRIPTIONS ===
def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id, 
        traci.constants.CMD_GET_PERSON_VARIABLE,
        10.0, 
        [traci.constants.VAR_WAITING_TIME]
    )

def _subscribe_all_detectors():
    detector_ids = ["e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10"]
    vehicle_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(
            det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_vars
        )

# === STATE & REWARD ===
def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
    if not vehicle_data:
        return 0
    
    weight_map = {
        "car": 1.0,
        "jeep": 1.5,
        "bus": 2.2,
        "truck": 2.5,
        "motorcycle": 0.3,
        "tricycle": 0.5
    }
    
    for data in vehicle_data.values():
        vtype = data.get(traci.constants.VAR_TYPE, "car")
        wait = data.get(traci.constants.VAR_WAITING_TIME, 0)
        sumWait += wait * weight_map.get(vtype, 1.0)
    
    return sumWait

def _mainIntersection_queue():
    e2_values = [_weighted_waits(f"e2_{i}") for i in range(4, 11)]
    
    pedestrian = 0
    junction_sub = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
    if junction_sub:
        for pid, data in junction_sub.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    
    return e2_values + [pedestrian]

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
    mainCurrentPhase = (mainCurrentPhase + 1) % 10
    duration_adjustment = actionSpace[action_index]
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", mainCurrentPhase)
    
    # Set base duration based on phase type
    if mainCurrentPhase == 2 or mainCurrentPhase == 4:
        base_duration = 15
    elif mainCurrentPhase % 2 == 0:
        base_duration = 30
    else:
        base_duration = 3
    
    mainCurrentPhaseDuration = max(5, min(60, base_duration + duration_adjustment))
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", mainCurrentPhaseDuration)

def save_history(filename, headers, episodes, rewards, actor_losses, critic_losses, entropies):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        for i in range(len(episodes)):
            writer.writerow([episodes[i], rewards[i], actor_losses[i], critic_losses[i], entropies[i]])

# === START SIMULATION ===
traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("cluster_295373794_3477931123_7465167861")

print("=" * 70)
print("Starting BASELINE A2C Training (No Pedestrian Signals)")
print("=" * 70)
print("System: Single Agent - Main Intersection Only")
print("  - State: 8 queues + 5 phases = 13 dims")
print("  - Actions: 7 phase duration adjustments (-15..+15)")
print("  - Reward: -sum(queue)/1000 (stable scaling)")
print("  - NO pedestrian crossing signals (baseline comparison)")
print("=" * 70)

# === MAIN TRAINING LOOP ===
while traci.simulation.getMinExpectedNumber() > 0:
    current_time = traci.simulation.getTime()
    
    # Stop at target time OR when vehicles are depleted
    if current_time >= 21500:
        print("Target time reached (21500). Saving model...")
        break
    
    # Safety check: stop if no vehicles remain for extended period
    if traci.simulation.getMinExpectedNumber() == 0:
        print(f"No more vehicles expected at time {current_time:.2f}. Stopping simulation...")
        break

    mainCurrentPhaseDuration -= stepLength

    if mainCurrentPhaseDuration <= 0:
        
        # --- 1. GET STATE (Observe) ---
        main_queue = np.array(_mainIntersection_queue())
        normalized_main_queue = main_queue / 1000.0
        main_phase = to_categorical(mainCurrentPhase // 2, num_classes=5).flatten()
        mainCurrentState = np.concatenate([
            normalized_main_queue, main_phase
        ]).astype(np.float32)
        
        # --- 2. CALCULATE REWARD ---
        mainReward = calculate_reward(main_queue)
        
        # --- 3. STORE TRANSITION (for previous step) ---
        if mainPrevState is not None:
            mainIntersectionAgent.store_transition(mainPrevState, mainPrevAction, mainReward)
        
        # --- 4. CHOOSE NEW ACTION (Act) ---
        mainActionIndex = mainIntersectionAgent.act(mainCurrentState, training=True)
        
        # --- 5. APPLY ACTION & UPDATE PHASE ---
        _mainIntersection_phase(mainActionIndex)
        
        # --- 6. TRAIN (if episode ends at phase 0) ---
        if mainCurrentPhase == 0 and mainPrevState is not None:
            mainEpisodeNumber += 1
            
            actor_loss, critic_loss, entropy, total_reward = mainIntersectionAgent.train_on_episode()
            main_actor_loss_history.append(actor_loss)
            main_critic_loss_history.append(critic_loss)
            main_entropy_history.append(entropy)
            main_reward_history.append(total_reward)
            main_episode_steps.append(mainEpisodeNumber)
            
            raw_queue_total = sum(main_queue)
            print(f"[BASELINE Ep {mainEpisodeNumber:3d}] EpisodeRwd: {total_reward:7.3f} | "
                  f"ALoss: {actor_loss:7.4f} CLoss: {critic_loss:9.2f} | "
                  f"Ent: {entropy:5.3f} | AvgQueue: {raw_queue_total/8:7.1f}")
        
        # --- 7. SAVE CURRENT STATE/ACTION for next loop ---
        mainPrevState = mainCurrentState
        mainPrevAction = mainActionIndex

    traci.simulationStep()

# === SAVE MODEL & HISTORY ===
print("\n" + "=" * 70)
print("Simulation complete! Saving trained model and history...")
try:
    mainIntersectionAgent.model.save('./Olivarez_traci/models_A2C_Baseline/A2C_Baseline_Agent.keras')
    print("✓ Model saved successfully")
except Exception as e:
    print(f"⚠ Error saving model: {e}")

save_history(
    './Olivarez_traci/output_A2C_Baseline/a2c_baseline_agent_history.csv',
    ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'],
    main_episode_steps, main_reward_history,
    main_actor_loss_history, main_critic_loss_history, main_entropy_history
)
print("✓ Training history saved successfully")

print(f"\nTraining Summary:")
print(f"  Episodes trained: {mainEpisodeNumber}")
print("=" * 70)

traci.close()