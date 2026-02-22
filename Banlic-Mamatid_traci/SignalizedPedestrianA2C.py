import os
import sys
import traci
import numpy as np
import csv
from keras.utils import to_categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.A2C import A2CAgent as a2c

# === OUTPUT DIRECTORIES ===
os.makedirs('./Banlic-Mamatid_traci/output_A2C', exist_ok=True)
os.makedirs('./Banlic-Mamatid_traci/models_A2C', exist_ok=True)

# --- CONFIGURATION ---
GAMMA         = 0.99
LEARNING_RATE = 0.0005
ENTROPY_COEF  = 0.08
VALUE_COEF    = 0.5
MAX_GRAD_NORM = 1.0

# === AGENT INITIALIZATION ===
# Main Agent: 6 detectors (e2_0-e2_5) + 5 phase one-hot = 11
MainAgent = a2c(
    state_size=11,
    action_size=11,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    entropy_coef=ENTROPY_COEF,
    value_coef=VALUE_COEF,
    max_grad_norm=MAX_GRAD_NORM,
    name='Main_A2CAgent_Signalized'
)
# Override the hardcoded model_dir in A2CAgent
MainAgent.model_dir = './Banlic-Mamatid_traci/models_A2C/'

# === OPTIONAL: CONTINUE TRAINING FROM SAVED MODELS ===
CONTINUE_TRAINING = True

if CONTINUE_TRAINING:
    try:
        print("\n" + "=" * 70)
        print("Attempting to load existing models for continued training...")
        print("=" * 70)
        from keras.models import load_model

        main_path = './Banlic-Mamatid_traci/models_A2C/Main_A2CAgent_Signalized.keras'

        if os.path.exists(main_path):
            MainAgent.model = load_model(main_path)
            print(f"  OK Loaded Main Agent from {main_path}")
        else:
            print(f"  WARN Main model not found - starting fresh")

        print("=" * 70)
    except Exception as e:
        print(f"  WARN Error loading models: {e}")
        print("  Starting with fresh models instead.")
else:
    print("\n" + "=" * 70)
    print("CONTINUE_TRAINING = False - Starting with fresh random weights")
    print("=" * 70)

# === SUMO ENVIRONMENT ===
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo',
    '-c', 'Banlic-Mamatid_traci/signalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Banlic-Mamatid_traci/output_A2C/SD_A2C_stats.xml',
    '--tripinfo-output', r'Banlic-Mamatid_traci/output_A2C/SD_A2C_trips.xml'
]

# --- SIMULATION VARIABLES ---
trainMode  = 1
stepLength = 0.1

currentPhase         = 0
currentPhaseDuration = 30

actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)

detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4", "e2_5"
]
detector_count = len(detector_ids)

step_counter  = 0
main_episode = 0

# Previous state/action (reward is assigned at the NEXT green phase)
prevState  = None
prevAction = None

# Track how many green-phase decisions are in the buffer right now.
# Banlic-Mamatid has 10 phases (0-9): 5 green phases (0, 2, 4, 6, 8) per cycle
# Train every 5 decisions = 1 full cycle
buffer_count = 0
TRAIN_EVERY = 5   # one full signal cycle = 5 green decisions

# Evaluation metrics (trainMode == 0)
metric_observation_count = 0
throughput_total = 0
jam_length_total = 0

# Training history
actor_loss_history  = []
critic_loss_history = []
entropy_history     = []
reward_history      = []
episode_steps       = []

# --- EPISODE NUMBER RECOVERY ---
def _get_last_episode(filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip()
                    if last_line:
                        return int(last_line.split(',')[0])
        except Exception:
            pass
    return 0

if CONTINUE_TRAINING:
    main_episode = _get_last_episode('./Banlic-Mamatid_traci/output_A2C/Main_A2C_history.csv')
    print(f"Resuming - Main Episode: {main_episode}")
    print("=" * 70)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id,
        traci.constants.CMD_GET_PERSON_VARIABLE,
        20.0,
        [traci.constants.VAR_WAITING_TIME]
    )

def _subscribe_all_detectors():
    vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_context_vars)
        traci.lanearea.subscribe(det_id, vehicle_vars)

def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
    if not vehicle_data:
        return 0

    for data in vehicle_data.values():
        v_type = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        
        weights = {
            "car": 1.0, "jeep": 1.5, "bus": 2.2, "truck": 2.5,
            "motorcycle": 0.3, "tricycle": 0.5
        }
        sumWait += waitTime * weights.get(v_type, 1.0)
    return sumWait

def _intersection_queue():
    return [_weighted_waits(det_id) for det_id in detector_ids]

def calculate_reward(normalized_queue):
    """
    Reward = negative sum of normalized queue weights, clipped to [-10, 0].
    Lower queues → higher (less negative) reward.
    """
    return float(np.clip(-np.sum(normalized_queue), -10.0, 0.0))

def _train_agent(agent, episode_num, agent_name):
    """
    Train the agent on buffered episode and return metrics.
    """
    actor_loss, critic_loss, entropy, total_reward = agent.train_on_episode()
    
    # Entropy management — same as Balibago A2C
    MIN_ENTROPY = 1.0
    if entropy < MIN_ENTROPY:
        agent.entropy_coef = min(0.2, agent.entropy_coef * 1.1)
    else:
        agent.entropy_coef = max(0.01, agent.entropy_coef * 0.995)
    
    if trainMode == 1:
        print(f"[{agent_name} | Ep {episode_num:4d}] "
              f"Reward: {total_reward:7.3f} | "
              f"Actor Loss: {actor_loss:7.4f} | "
              f"Critic Loss: {critic_loss:9.2f} | "
              f"Entropy: {entropy:5.3f} (coef: {agent.entropy_coef:.4f})")
    
    return {
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'entropy': entropy,
        'total_reward': total_reward
    }

def save_history(filename, headers, episode_steps, reward_hist, 
                 actor_loss_hist, critic_loss_hist, entropy_hist):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        for i in range(len(episode_steps)):
            writer.writerow([
                episode_steps[i], 
                reward_hist[i], 
                actor_loss_hist[i], 
                critic_loss_hist[i], 
                entropy_hist[i]
            ])

# ============================================================
# MAIN EXECUTION
# ============================================================

traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("253768576")

# ============================================================
# SIMULATION LOOP
# ============================================================

while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    currentPhaseDuration -= stepLength

    # Decision state: timer expired AND we are on a green (even) phase
    decision_needed = (currentPhaseDuration <= 0) and (currentPhase % 2 == 0)

    obs = None

    # ----------------------------------------------------------
    # 1. OBSERVE CURRENT STATE
    # ----------------------------------------------------------
    if decision_needed:
        queue = np.array(_intersection_queue())
        norm_queue = queue / 1000.0  # Normalize queue values
        phase_oh = to_categorical(currentPhase // 2, num_classes=5).flatten()
        obs = np.concatenate([norm_queue, phase_oh]).astype(np.float32)
        # shape: 6 queue + 5 phase-OH = 11 ✓

    # ----------------------------------------------------------
    # 2. STORE TRANSITION
    #    The reward for the PREVIOUS action is the queue we see NOW.
    #    store_transition() appends to agent.states/actions/rewards.
    # ----------------------------------------------------------
    if trainMode == 1:
        if decision_needed and prevState is not None:
            reward = calculate_reward(norm_queue)
            MainAgent.store_transition(prevState, prevAction, reward)
            buffer_count += 1

    # ----------------------------------------------------------
    # 3. TRAIN ONCE BUFFER HAS A FULL CYCLE OF TRANSITIONS
    #    train_on_episode() clears the buffer after each call,
    #    so we reset the counter too.
    #    Training happens BEFORE act() so the updated policy is
    #    used for the very next action.
    # ----------------------------------------------------------
    if trainMode == 1:
        if decision_needed and buffer_count >= TRAIN_EVERY:
            main_episode += 1
            metrics = _train_agent(MainAgent, main_episode, "Main")
            actor_loss_history.append(metrics['actor_loss'])
            critic_loss_history.append(metrics['critic_loss'])
            entropy_history.append(metrics['entropy'])
            reward_history.append(metrics['total_reward'])
            episode_steps.append(main_episode)
            buffer_count = 0  # buffer cleared by train_on_episode()

    # ----------------------------------------------------------
    # 4. SELECT NEXT ACTION
    # ----------------------------------------------------------
    next_action_idx = None

    if decision_needed:
        next_action_idx = MainAgent.act(obs, training=(trainMode == 1))
        prevState = obs
        prevAction = next_action_idx
    elif currentPhaseDuration <= 0:
        # Yellow phase — carry over last green action
        next_action_idx = prevAction if prevAction is not None else 5

    # ----------------------------------------------------------
    # 5. APPLY PHASE TRANSITIONS TO SIMULATION
    # ----------------------------------------------------------
    
    # Banlic-Mamatid has 10-phase cycle (0-9)
    if currentPhaseDuration <= 0:
        currentPhase = (currentPhase + 1) % 10
        # Set phase for both junctions
        traci.trafficlight.setPhase("253768576", currentPhase)
        traci.trafficlight.setPhase("253499548", currentPhase)

        if currentPhase % 2 == 1:  # yellow/transition
            currentPhaseDuration = 5
        else:  # green
            duration_adj = actionSpace[next_action_idx]
            # Base durations for each green phase (0, 2, 4, 6, 8)
            base = {0: 30, 2: 30, 4: 45, 6: 60, 8: 25}.get(currentPhase, 30)
            currentPhaseDuration = max(5, min(180, base + duration_adj))

        traci.trafficlight.setPhaseDuration("253768576", currentPhaseDuration)
        traci.trafficlight.setPhaseDuration("253499548", currentPhaseDuration)

    # ----------------------------------------------------------
    # 6. EVALUATION METRICS (trainMode == 0 only)
    # ----------------------------------------------------------
    TRACK_INTERVAL_STEPS = int(60 / stepLength)
    if trainMode == 0 and step_counter % TRACK_INTERVAL_STEPS == 0:
        jam_length = 0
        throughput = 0
        metric_observation_count += 1

        for det_id in detector_ids:
            detector_stats = traci.lanearea.getSubscriptionResults(det_id)
            if not detector_stats:
                continue
            jam_length += detector_stats.get(traci.constants.JAM_LENGTH_METERS, 0)
            throughput += detector_stats.get(traci.constants.VAR_INTERVAL_NUMBER, 0)

        jam_length /= detector_count
        jam_length_total += jam_length
        throughput_total += throughput

    traci.simulationStep()

# ============================================================
# END OF SIMULATION
# ============================================================

if trainMode == 0 and metric_observation_count > 0:
    print(f"\n{'='*70}")
    print("Performance Metrics")
    print(f"{'='*70}")
    print(f"  Average Jam Length : {jam_length_total / metric_observation_count:.2f} m")
    print(f"  Average Throughput : {throughput_total / metric_observation_count:.2f} vehicles/min")

if trainMode == 1:
    # Flush any leftover transitions that haven't reached TRAIN_EVERY yet
    if buffer_count > 0 and prevState is not None:
        main_episode += 1
        metrics = _train_agent(MainAgent, main_episode, "Main [end flush]")
        actor_loss_history.append(metrics['actor_loss'])
        critic_loss_history.append(metrics['critic_loss'])
        entropy_history.append(metrics['entropy'])
        reward_history.append(metrics['total_reward'])
        episode_steps.append(main_episode)

    print("\nSaving trained models...")
    MainAgent.save()
    print("  OK Model saved to ./Banlic-Mamatid_traci/models_A2C/")

    print("\nSaving training history...")
    headers = ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy']
    save_history('./Banlic-Mamatid_traci/output_A2C/Main_A2C_history.csv', headers,
                 episode_steps, reward_history,
                 actor_loss_history, critic_loss_history, entropy_history)
    print("  OK Main_A2C_history.csv saved.")

    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    print(f"  Main - Total Episodes : {main_episode}")
    recent = reward_history[-10:] if len(reward_history) >= 10 else reward_history
    print(f"  Main - Last 10 Avg Reward : {np.mean(recent):.3f}" if recent else "  Main - N/A")

traci.close()