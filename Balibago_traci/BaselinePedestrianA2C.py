import os
import sys 
import traci
import numpy as np
import csv
from keras.utils import to_categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.A2C import A2CAgent as a2c

# === OUTPUT DIRECTORIES ===
os.makedirs('./Balibago_traci/output_A2C', exist_ok=True)
os.makedirs('./Balibago_traci/models_A2C_baseline', exist_ok=True)

# --- CONFIGURATION ---
GAMMA         = 0.99
LEARNING_RATE = 0.0005
ENTROPY_COEF  = 0.08
VALUE_COEF    = 0.5
MAX_GRAD_NORM = 1.0

# === AGENT INITIALIZATION ===
# North: 8 detectors (e2_0-e2_7) + 4 phase one-hot = 12 (no pedestrian for baseline)
NorthAgent = a2c(
    state_size=12,
    action_size=11,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    entropy_coef=ENTROPY_COEF,
    value_coef=VALUE_COEF,
    max_grad_norm=MAX_GRAD_NORM,
    name='North_A2CAgent_Baseline'
)
# Override the hardcoded model directory
NorthAgent.model_dir = './Balibago_traci/models_A2C_baseline/'

# South: 5 detectors (e2_8-e2_12) + 3 phase one-hot = 8 (no pedestrian for baseline)
# South has 6 phases (0-5) = 3 green phases
SouthAgent = a2c(
    state_size=8,
    action_size=11,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    entropy_coef=ENTROPY_COEF,
    value_coef=VALUE_COEF,
    max_grad_norm=MAX_GRAD_NORM,
    name='South_A2CAgent_Baseline'
)
# Override the hardcoded model directory
SouthAgent.model_dir = './Balibago_traci/models_A2C_baseline/'

# === SUMO ENVIRONMENT ===
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo', 
    '-c', 'Balibago_traci/baselinePed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Balibago_traci/output_A2C/SD_A2C_stats_baseline.xml',
    '--tripinfo-output', r'Balibago_traci/output_A2C/SD_A2C_trips_baseline.xml'
]

# --- SIMULATION VARIABLES ---
trainMode = 1
stepLength = 0.1

northCurrentPhase = 0
northCurrentPhaseDuration = 30
southCurrentPhase = 0
southCurrentPhaseDuration = 30

actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)

detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4", "e2_5", "e2_6", "e2_7", 
    "e2_8", "e2_9", "e2_10", "e2_11", "e2_12"
]
detector_count = len(detector_ids)

metric_observation_count = 0
throughput_total = 0
jam_length_total = 0

northPrevState = None
northPrevAction = None
southPrevState = None
southPrevAction = None

step_counter = 0
north_episode = 0
south_episode = 0

# Track how many green-phase decisions are in the buffer
# North: Train every 4 decisions = 1 full signal cycle (4 green phases in 8-phase cycle)
# South: Train every 3 decisions = 1 full signal cycle (3 green phases in 6-phase cycle)
north_buffer_count = 0
south_buffer_count = 0
TRAIN_EVERY_NORTH = 4
TRAIN_EVERY_SOUTH = 3

# --- DATA LOGGING ---
north_actor_loss_history = []
north_critic_loss_history = []
north_entropy_history = []
north_reward_history = []
north_episode_steps = []

south_actor_loss_history = []
south_critic_loss_history = []
south_entropy_history = []
south_reward_history = []
south_episode_steps = []

# --- HELPER FUNCTIONS ---
def _subscribe_all_detectors():
    vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_context_vars)
        traci.lanearea.subscribe(det_id, vehicle_vars)

def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
    if not vehicle_data: return 0

    for data in vehicle_data.values():
        v_type = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        
        weights = {
            "car": 1.0, "jeep": 1.5, "bus": 2.2, "truck": 2.5,
            "motorcycle": 0.3, "tricycle": 0.5
        }
        sumWait += waitTime * weights.get(v_type, 1.0)
    return sumWait

def _northIntersection_queue():
    queues = [_weighted_waits(f"e2_{i}") for i in range(8)]
    return queues

def _southIntersection_queue():
    queues = [_weighted_waits(f"e2_{i}") for i in range(8, 13)]
    return queues

def calculate_reward(norm_queue):
    """Negative normalized queue sum, clipped to [-10, 0]."""
    return float(np.clip(-sum(norm_queue), -10.0, 0.0))

def _train_agent(agent, episode_num, agent_name):
    """
    Train the agent on the buffered episode and return metrics.
    train_on_episode() returns a tuple: (actor_loss, critic_loss, entropy, total_reward)
    """
    actor_loss, critic_loss, entropy, total_reward = agent.train_on_episode()
    
    # Entropy management — same as Olivarez/Signalized A2C
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
    
    return {'actor_loss': actor_loss, 'critic_loss': critic_loss,
            'entropy': entropy, 'total_reward': total_reward}

def save_history(filename, headers, episodes, rewards, actor_losses, critic_losses, entropies):
    """Save training history to CSV"""
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        for i in range(len(episodes)):
            writer.writerow([episodes[i], rewards[i], actor_losses[i], critic_losses[i], entropies[i]])

# --- MAIN EXECUTION ---
# Use port 8814 for baseline (8813 is used by signalized)
traci.start(Sumo_config, port=8814)
_subscribe_all_detectors()

while traci.simulation.getMinExpectedNumber() > 0 and step_counter < 576000:
    step_counter += 1
    northCurrentPhaseDuration -= stepLength
    southCurrentPhaseDuration -= stepLength

    # ----------------------------------------------------------
    # 1. OBSERVATION PHASE (IMMUTABLE)
    # ----------------------------------------------------------
    obs_north = None
    obs_south = None
    
    # Flags to determine if we are in a decision state (Timer expired & Green Phase)
    north_decision_needed = (northCurrentPhaseDuration <= 0) and (northCurrentPhase % 2 == 0)
    south_decision_needed = (southCurrentPhaseDuration <= 0) and (southCurrentPhase % 2 == 0)
    
    # Capture North State if needed
    if north_decision_needed:
        queue = np.array(_northIntersection_queue())
        n_norm_queue = queue / 2000.0  # Normalize queue - North uses 2000
        n_phase_oh = to_categorical(northCurrentPhase // 2, num_classes=4).flatten()
        obs_north = np.concatenate([n_norm_queue, n_phase_oh]).astype(np.float32)
        # shape: 8 queue + 4 phase-OH = 12

    # Capture South State if needed
    if south_decision_needed:
        queue = np.array(_southIntersection_queue())
        s_norm_queue = queue / 1000.0  # Normalize queue - South uses 1000
        s_phase_oh = to_categorical(southCurrentPhase // 2, num_classes=3).flatten()
        obs_south = np.concatenate([s_norm_queue, s_phase_oh]).astype(np.float32)
        # shape: 5 queue + 3 phase-OH = 8

    # ----------------------------------------------------------
    # 2. STORE TRANSITION
    #    The reward for the PREVIOUS action is the queue we see NOW.
    #    store_transition() appends to agent.states/actions/rewards.
    # ----------------------------------------------------------
    if trainMode == 1:
        if north_decision_needed and northPrevState is not None:
            reward_north = calculate_reward(n_norm_queue)
            NorthAgent.store_transition(northPrevState, northPrevAction, reward_north)
            north_buffer_count += 1

        if south_decision_needed and southPrevState is not None:
            reward_south = calculate_reward(s_norm_queue)
            SouthAgent.store_transition(southPrevState, southPrevAction, reward_south)
            south_buffer_count += 1

    # ----------------------------------------------------------
    # 3. TRAIN ONCE BUFFER HAS A FULL CYCLE OF TRANSITIONS
    #    train_on_episode() clears the buffer after each call,
    #    so we reset the counter too.
    # ----------------------------------------------------------
    if trainMode == 1:
        if north_decision_needed and north_buffer_count >= TRAIN_EVERY_NORTH:
            north_episode += 1
            metrics = _train_agent(NorthAgent, north_episode, "North")
            north_actor_loss_history.append(metrics['actor_loss'])
            north_critic_loss_history.append(metrics['critic_loss'])
            north_entropy_history.append(metrics['entropy'])
            north_reward_history.append(metrics['total_reward'])
            north_episode_steps.append(north_episode)
            north_buffer_count = 0  # buffer cleared by train_on_episode()

        if south_decision_needed and south_buffer_count >= TRAIN_EVERY_SOUTH:
            south_episode += 1
            metrics = _train_agent(SouthAgent, south_episode, "South")
            south_actor_loss_history.append(metrics['actor_loss'])
            south_critic_loss_history.append(metrics['critic_loss'])
            south_entropy_history.append(metrics['entropy'])
            south_reward_history.append(metrics['total_reward'])
            south_episode_steps.append(south_episode)
            south_buffer_count = 0  # buffer cleared by train_on_episode()

    # ----------------------------------------------------------
    # 4. SELECT NEXT ACTION
    # ----------------------------------------------------------
    next_action_N_idx = None
    next_action_S_idx = None

    if north_decision_needed:
        next_action_N_idx = NorthAgent.act(obs_north, training=(trainMode == 1))
        northPrevState = obs_north
        northPrevAction = next_action_N_idx
    elif northCurrentPhaseDuration <= 0:
        # Yellow phase — carry over last green action
        next_action_N_idx = northPrevAction if northPrevAction is not None else 5

    if south_decision_needed:
        next_action_S_idx = SouthAgent.act(obs_south, training=(trainMode == 1))
        southPrevState = obs_south
        southPrevAction = next_action_S_idx
    elif southCurrentPhaseDuration <= 0:
        next_action_S_idx = southPrevAction if southPrevAction is not None else 5

    # ----------------------------------------------------------
    # 5. APPLY PHASE TRANSITIONS TO SIMULATION
    # ----------------------------------------------------------
    
    # Apply North
    if northCurrentPhaseDuration <= 0:
        northCurrentPhase = (northCurrentPhase + 1) % 8
        traci.trafficlight.setPhase("4902876117", northCurrentPhase)

        if northCurrentPhase % 2 == 1:  # yellow/transition
            northCurrentPhaseDuration = 5
        else:  # green
            duration_adj = actionSpace[next_action_N_idx]
            base = {0: 45, 2: 130, 4: 30, 6: 90}.get(northCurrentPhase, 30)
            northCurrentPhaseDuration = max(5, min(180, base + duration_adj))
        
        traci.trafficlight.setPhaseDuration("4902876117", northCurrentPhaseDuration)

    # Apply South (6-phase cycle: 0-5)
    if southCurrentPhaseDuration <= 0:
        southCurrentPhase = (southCurrentPhase + 1) % 6
        traci.trafficlight.setPhase("12188714", southCurrentPhase)

        if southCurrentPhase % 2 == 1:  # yellow/transition
            southCurrentPhaseDuration = 5
        else:  # green
            duration_adj = actionSpace[next_action_S_idx]
            base = {0: 30, 2: 30, 4: 45}.get(southCurrentPhase, 30)
            southCurrentPhaseDuration = max(5, min(180, base + duration_adj))
        
        traci.trafficlight.setPhaseDuration("12188714", southCurrentPhaseDuration)

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
            if not detector_stats: continue
            
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
    if north_buffer_count > 0 and northPrevState is not None:
        north_episode += 1
        metrics = _train_agent(NorthAgent, north_episode, "North [end flush]")
        north_actor_loss_history.append(metrics['actor_loss'])
        north_critic_loss_history.append(metrics['critic_loss'])
        north_entropy_history.append(metrics['entropy'])
        north_reward_history.append(metrics['total_reward'])
        north_episode_steps.append(north_episode)

    if south_buffer_count > 0 and southPrevState is not None:
        south_episode += 1
        metrics = _train_agent(SouthAgent, south_episode, "South [end flush]")
        south_actor_loss_history.append(metrics['actor_loss'])
        south_critic_loss_history.append(metrics['critic_loss'])
        south_entropy_history.append(metrics['entropy'])
        south_reward_history.append(metrics['total_reward'])
        south_episode_steps.append(south_episode)

    print("\nSaving trained models...")
    NorthAgent.save()
    SouthAgent.save()
    print("  OK Models saved to ./Balibago_traci/models_A2C_baseline/")

    print("\nSaving training history...")
    headers = ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy']
    
    save_history('./Balibago_traci/output_A2C/North_A2C_baseline_history.csv', headers,
                 north_episode_steps, north_reward_history,
                 north_actor_loss_history, north_critic_loss_history, north_entropy_history)
    
    save_history('./Balibago_traci/output_A2C/South_A2C_baseline_history.csv', headers,
                 south_episode_steps, south_reward_history,
                 south_actor_loss_history, south_critic_loss_history, south_entropy_history)
    
    print("  OK North_A2C_baseline_history.csv saved.")
    print("  OK South_A2C_baseline_history.csv saved.")

    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    print(f"  North - Total Episodes : {north_episode}")
    recent_n = north_reward_history[-10:] if len(north_reward_history) >= 10 else north_reward_history
    print(f"  North - Last 10 Avg Reward : {np.mean(recent_n):.3f}" if recent_n else "  North - N/A")
    print(f"  South - Total Episodes : {south_episode}")
    recent_s = south_reward_history[-10:] if len(south_reward_history) >= 10 else south_reward_history
    print(f"  South - Last 10 Avg Reward : {np.mean(recent_s):.3f}" if recent_s else "  South - N/A")

traci.close()