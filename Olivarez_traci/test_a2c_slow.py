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
from keras.models import load_model

# === UNIFIED AGENT INITIALIZATION ===
# Single agent controls all three synchronized traffic lights
trafficLightAgent = a2c(
    state_size=19,  # 14 queues (11 vehicle detectors + 3 pedestrian) + 5 phase encoding
    action_size=11,  # Same as DQN: -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25
    gamma=0.99,
    learning_rate=0.0005,  # Same as original multi-agent A2C
    entropy_coef=0.08,     # Same as original main agent
    value_coef=0.5,
    max_grad_norm=1.0,     # Same as original
    name='A2C_Unified_Agent'
)

# === LOAD EXISTING MODEL IF IT EXISTS ===
CONTINUE_TRAINING = True  # Set to False to start fresh
trainMode = 0; 
if trainMode == 0:
    # TEST MODE: Always load the trained model
    print("\n" + "=" * 70)
    print("TEST/EVALUATION MODE - Loading trained model...")
    print("=" * 70)
    
    model_path = './Olivarez_traci/models_A2C/A2C_Unified_Agent.keras'
    if os.path.exists(model_path):
        trafficLightAgent.model = load_model(model_path)
        print(f"✓ Loaded model from {model_path}")
    else:
        sys.exit(f"ERROR: Model not found at {model_path}. Train the model first!")
    
    print("=" * 70 + "\n")
    
elif CONTINUE_TRAINING:
    # TRAINING MODE: Try to load existing model to continue training
    try:
        print("\n" + "=" * 70)
        print("Attempting to load existing model for continued training...")
        print("=" * 70)
        
        model_path = './Olivarez_traci/models_A2C/A2C_Unified_Agent.keras'
        
        if os.path.exists(model_path):
            trafficLightAgent.model = load_model(model_path)
            print(f"✓ Loaded Unified Agent from {model_path}")
        else:
            print(f"⚠ Model not found - starting fresh")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        print("Starting with fresh model instead.")
else:
    print("\n" + "=" * 70)
    print("CONTINUE_TRAINING = False - Starting with fresh random weights")
    print("=" * 70)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO Configuration - different outputs for train vs test
if trainMode == 1:
    # Training mode - use default training traffic
    Sumo_config = [
        'sumo',
        '-c', 'Olivarez_traci\signalizedPed.sumocfg',
        '--step-length', '0.1',
        '--delay', '0',
        '--lateral-resolution', '0.1',
        '--statistic-output', r'Olivarez_traci\output_A2C\SD_A2C_stats.xml',
        '--tripinfo-output', r'Olivarez_traci\output_A2C\SD_A2C_trips.xml'
    ]
else:
    # Test mode - use normal traffic scenario
    Sumo_config = [
        'sumo',
        '-c', r'Olivarez_traci\signalizedPed.sumocfg',
        '--route-files', r'Olivarez_traci\demand\flows_slow_traffic.rou.xml',
        '--step-length', '0.1',
        '--delay', '0',
        '--lateral-resolution', '0.1',
        '--statistic-output', r'Olivarez_traci\output_A2C\SP_A2C_Unified_Slow_stats.xml',
        '--tripinfo-output', r'Olivarez_traci\output_A2C\SP_A2C_Unified_Slow_trips.xml'
    ]

# Simulation Variables
stepLength = 0.1
step_counter = 0
currentPhase = 0
currentPhaseDuration = 30
actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)
detector_count = 11

# Metrics
throughput_average = 0
throughput_total = 0
jam_length_average = 0
jam_length_total = 0
metric_observation_count = 0

# Store previous states and actions for learning
prevState = None
prevAction = None

# Episode counter
episodeNumber = 0

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
    
    episodeNumber = get_last_episode('./Olivarez_traci/output_A2C/a2c_agent_history.csv')
    print(f"Continuing from Episode {episodeNumber}")
    print("=" * 70)

# Data storage for plotting
actor_loss_history = []
critic_loss_history = []
entropy_history = []
reward_history = []
episode_steps = []

# Object Context Subscription in SUMO
detector_ids = [
    "e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10",
    "e2_0", "e2_1", "e2_2", "e2_3"
]

def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id,
        traci.constants.CMD_GET_PERSON_VARIABLE,
        10.0,
        [traci.constants.VAR_WAITING_TIME]
    )
    
def _subscribe_all_detectors():
    global detector_ids 
    
    vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(
            det_id,
            traci.constants.CMD_GET_VEHICLE_VARIABLE,
            3,
            vehicle_context_vars
        )
    
    for det_id in detector_ids:
        traci.lanearea.subscribe(
            det_id,
            vehicle_vars
        )

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
            
    return [e2_0, e2_1, e2_2, e2_3, e2_4, e2_5, e2_6, e2_7, e2_8, e2_9, e2_10, 
            pedestrianM*1.25, pedestrianW*1.25, pedestrianE*1.25]

def calculate_reward(current_state_queues):
    if current_state_queues is None:
        return 0.0
    
    current_total = sum(current_state_queues)
    # Match original A2C reward scaling
    normalized_queue = current_total / 1000.0
    reward = -normalized_queue
    reward = np.clip(reward, -10.0, 0.0)
    return reward

def _trafficLight_phase(action_index):
    global currentPhase, currentPhaseDuration
    
    currentPhase += 1
    currentPhase = currentPhase % 10
    
    # Synchronized control of all three traffic lights
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

detector_ids = traci.lanearea.getIDList()
currentPhase_onehot = to_categorical(currentPhase//2, num_classes=5).flatten()

print("=" * 70)
if trainMode == 1:
    print("Starting UNIFIED A2C Training")
    print("=" * 70)
    print("System: Single Agent Synchronized Control")
    print("  - One agent controls all three traffic lights synchronously")
    print("  - Matches DQN architecture for fair comparison")
else:
    print("TESTING UNIFIED A2C AGENT ON NORMAL TRAFFIC SCENARIO")
    print("=" * 70)
    print("Configuration:")
    print("  - Traffic Flow: flows_slow_traffic.rou.xml")
    print("  - Step Length: 0.1s")
    print("  - Mode: INFERENCE (training=False)")
    print("  - Outputs: SP_A2C_Unified_Normal_stats.xml, SP_A2C_Unified_Normal_trips.xml")
print("=" * 70)

# Simulation Loop
decision_step_count = 0
max_sim_steps = 50000 if trainMode == 0 else float('inf')

while traci.simulation.getMinExpectedNumber() > 0 and step_counter < max_sim_steps:
    step_counter += 1
    currentPhaseDuration -= stepLength

    # Observation and Reward
    if currentPhaseDuration <= 0:
        if currentPhase % 2 == 0:
            queue = np.array(_mainIntersection_queue())
            normalized_queue = queue / 1000
            currentPhase_onehot = to_categorical(currentPhase//2, num_classes=5).flatten()
            currentState = np.concatenate([normalized_queue, currentPhase_onehot]).astype(np.float32)
            
            if trainMode == 1:
                # Match original A2C: calculate reward from raw queue
                reward = calculate_reward(queue)
                
                if prevState is not None and prevAction is not None:
                    # Store transition in A2C agent
                    trafficLightAgent.store_transition(prevState, prevAction, reward)

    # Action
    if currentPhaseDuration <= 0:
        if currentPhase % 2 == 0:
            actionIndex = trafficLightAgent.act(currentState, training=(trainMode == 1))
            prevState = currentState
            prevAction = actionIndex
            
            # Optional: Print progress in test mode
            if trainMode == 0 and decision_step_count % 100 == 0:
                sim_time = traci.simulation.getTime()
                print(f"[Step {decision_step_count:5d}] Time: {sim_time:7.1f}s | "
                      f"Queue: {sum(queue):7.1f} | Action: {actionSpace[actionIndex]:3d} | SimStep: {step_counter}")
            
            decision_step_count += 1
        else:
            actionIndex = prevAction

        _trafficLight_phase(actionIndex)
        
        # Train at end of episode (when phase cycles back to 0)
        if trainMode == 1 and currentPhase == 0 and prevState is not None:
            episodeNumber += 1
            
            # Train on episode
            actor_loss, critic_loss, entropy, total_reward = trafficLightAgent.train_on_episode()
            
            # ENTROPY MANAGEMENT: Enforce minimum entropy to prevent premature exploitation
            MIN_ENTROPY = 1.0  # Don't let entropy drop below this
            if entropy < MIN_ENTROPY:
                # Increase entropy coefficient temporarily to encourage exploration
                trafficLightAgent.entropy_coef = min(0.2, trafficLightAgent.entropy_coef * 1.1)
            else:
                # Gradually decay entropy coefficient for late-stage fine-tuning
                trafficLightAgent.entropy_coef = max(0.01, trafficLightAgent.entropy_coef * 0.995)
            
            # Store metrics
            actor_loss_history.append(actor_loss)
            critic_loss_history.append(critic_loss)
            entropy_history.append(entropy)
            reward_history.append(total_reward)
            episode_steps.append(episodeNumber)
            
            raw_queue_total = sum(queue)
            print(f"\n[Episode {episodeNumber:3d}] Total Reward: {total_reward:7.3f} | "
                  f"Actor Loss: {actor_loss:7.4f} | Critic Loss: {critic_loss:9.2f} | "
                  f"Entropy: {entropy:5.3f} (coef: {trafficLightAgent.entropy_coef:.4f}) | "
                  f"Avg Queue: {raw_queue_total/14:7.1f}\n")
    
    # Periodic tracking (throughput and queue_length)
    if trainMode == 0 and step_counter % int(60 / stepLength) == 0:
        jam_length = 0
        throughput = 0
        metric_observation_count += 1
        
        for det_id in detector_ids:
            detector_stats = traci.lanearea.getSubscriptionResults(det_id)

            if not detector_stats:
                print("Lane Data Error: Undetected")
                break
            
            jam_length += detector_stats.get(traci.constants.JAM_LENGTH_METERS, 0)
            throughput += detector_stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)
                
        jam_length /= detector_count
        jam_length_total += jam_length
        throughput_total += throughput
        
    traci.simulationStep()
    
    # Force stop at limit in test mode
    if trainMode == 0 and step_counter >= max_sim_steps:
        print(f"\nReached simulation step limit ({max_sim_steps}). Stopping.")
        break

if trainMode == 0:
    jam_length_average = jam_length_total / metric_observation_count if metric_observation_count > 0 else 0
    throughput_average = throughput_total / metric_observation_count if metric_observation_count > 0 else 0
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)
    print(f"Total decision steps executed: {decision_step_count}")
    print(f"Total simulation steps executed: {step_counter}")
    print(f"\nPerformance Metrics:")
    print(f"  Average Jam Length: {jam_length_average:.2f} meters")
    print(f"  Average Throughput: {throughput_average:.2f} vehicles/minute")
    print(f"  Total Observations: {metric_observation_count}")
    print(f"\nOutput files saved:")
    print(f"  - Statistics: Olivarez_traci/output_A2C/SP_A2C_Unified_Normal_stats.xml")
    print(f"  - Trip Info: Olivarez_traci/output_A2C/SP_A2C_Unified_Normal_trips.xml")
    print("\nCompare these with your baseline to evaluate performance!")
    print("=" * 70 + "\n")

if trainMode == 1:
    # Save trained model
    print("\nSaving trained model...")
    trafficLightAgent.save()
    print("Model saved successfully!")

    print("Saving training history...")
    save_history('./Olivarez_traci/output_A2C/a2c_agent_history.csv', 
                ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy'], 
                episode_steps, reward_history, actor_loss_history, 
                critic_loss_history, entropy_history)
    print("History saved successfully!")
    
    print(f"\nTraining Summary:")
    print(f"  Total Episodes: {episodeNumber}")
    print(f"  Final Avg Reward: {np.mean(reward_history[-10:]) if len(reward_history) >= 10 else 'N/A'}")

traci.close()