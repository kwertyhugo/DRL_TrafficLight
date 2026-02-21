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
os.makedirs('./Balibago_traci/models_A2C', exist_ok=True)

# --- CONFIGURATION ---
GAMMA         = 0.99
LEARNING_RATE = 0.0005
ENTROPY_COEF  = 0.08
VALUE_COEF    = 0.5
MAX_GRAD_NORM = 1.0

# === AGENT INITIALIZATION ===
# North: 8 detectors (e2_0-e2_7) + 1 pedestrian + 4 phase one-hot = 13
NorthAgent = a2c(
    state_size=13,
    action_size=11,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    entropy_coef=ENTROPY_COEF,
    value_coef=VALUE_COEF,
    max_grad_norm=MAX_GRAD_NORM,
    name='North_A2CAgent'
)
# IMPORTANT: Override the hardcoded Olivarez model_dir in A2CAgent
NorthAgent.model_dir = './Balibago_traci/models_A2C/'

# South: 5 detectors (e2_8-e2_12) + 1 pedestrian + 4 phase one-hot = 10
# South intersection has 8 phases (0-7): same as North on the signalized network
SouthAgent = a2c(
    state_size=10,
    action_size=11,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    entropy_coef=ENTROPY_COEF,
    value_coef=VALUE_COEF,
    max_grad_norm=MAX_GRAD_NORM,
    name='South_A2CAgent'
)
# IMPORTANT: Override the hardcoded Olivarez model_dir in A2CAgent
SouthAgent.model_dir = './Balibago_traci/models_A2C/'

# === OPTIONAL: CONTINUE TRAINING FROM SAVED MODELS ===
CONTINUE_TRAINING = True

if CONTINUE_TRAINING:
    try:
        print("\n" + "=" * 70)
        print("Attempting to load existing models for continued training...")
        print("=" * 70)
        from keras.models import load_model

        north_path = './Balibago_traci/models_A2C/North_A2CAgent.keras'
        south_path = './Balibago_traci/models_A2C/South_A2CAgent.keras'

        if os.path.exists(north_path):
            NorthAgent.model = load_model(north_path)
            print(f"  OK Loaded North Agent from {north_path}")
        else:
            print(f"  WARN North model not found - starting fresh")

        if os.path.exists(south_path):
            SouthAgent.model = load_model(south_path)
            print(f"  OK Loaded South Agent from {south_path}")
        else:
            print(f"  WARN South model not found - starting fresh")

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
    '-c', 'Balibago_traci/signalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Balibago_traci/output_A2C/SD_A2C_stats.xml',
    '--tripinfo-output', r'Balibago_traci/output_A2C/SD_A2C_trips.xml'
]

# --- SIMULATION VARIABLES ---
trainMode  = 1
stepLength = 0.1

northCurrentPhase         = 0
northCurrentPhaseDuration = 30
southCurrentPhase         = 0
southCurrentPhaseDuration = 30

actionSpace = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)

detector_ids = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4",
    "e2_5", "e2_6", "e2_7", "e2_8", "e2_9",
    "e2_10", "e2_11", "e2_12"
]
detector_count = len(detector_ids)

step_counter  = 0
north_episode = 0
south_episode = 0

# Previous state/action (reward is assigned at the NEXT green phase)
northPrevState  = None
northPrevAction = None
southPrevState  = None
southPrevAction = None

# Track how many green-phase decisions are in the buffer right now.
# North has 4 green phases per 8-phase cycle → train every 4 decisions = 1 full cycle.
# South has 4 green phases per 8-phase cycle → same.
# This guarantees train_on_episode() always receives exactly 4 transitions,
# never 0 or 1 (which caused the all-zeros bug).
north_buffer_count = 0
south_buffer_count = 0
TRAIN_EVERY = 4   # one full signal cycle = 4 green decisions

# Evaluation metrics (trainMode == 0)
metric_observation_count = 0
throughput_total = 0
jam_length_total = 0

# Training history
north_actor_loss_history  = []
north_critic_loss_history = []
north_entropy_history     = []
north_reward_history      = []
north_episode_steps       = []

south_actor_loss_history  = []
south_critic_loss_history = []
south_entropy_history     = []
south_reward_history      = []
south_episode_steps       = []

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
    north_episode = _get_last_episode('./Balibago_traci/output_A2C/North_A2C_history.csv')
    south_episode = _get_last_episode('./Balibago_traci/output_A2C/South_A2C_history.csv')
    print(f"Resuming - North Episode: {north_episode} | South Episode: {south_episode}")
    print("=" * 70)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id,
        traci.constants.CMD_GET_PERSON_VARIABLE,
        10.0,
        [traci.constants.VAR_WAITING_TIME]
    )

def _subscribe_all_detectors():
    vehicle_context_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(
            det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_context_vars)
        traci.lanearea.subscribe(det_id, vehicle_vars)

def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
    if not vehicle_data:
        return 0
    weights = {"car": 1.0, "jeep": 1.5, "bus": 2.2,
               "truck": 2.5, "motorcycle": 0.3, "tricycle": 0.5}
    for data in vehicle_data.values():
        v_type   = data.get(traci.constants.VAR_TYPE, "car")
        waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
        sumWait += waitTime * weights.get(v_type, 1.0)
    return sumWait

def _northIntersection_queue():
    queues = [_weighted_waits(f"e2_{i}") for i in range(8)]
    pedestrian = 0
    junction_data = traci.junction.getContextSubscriptionResults("4902876117")
    if junction_data:
        for data in junction_data.values():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return queues + [pedestrian]   # length 9

def _southIntersection_queue():
    queues = [_weighted_waits(f"e2_{i}") for i in range(8, 13)]
    pedestrian = 0
    junction_data = traci.junction.getContextSubscriptionResults("12188714")
    if junction_data:
        for data in junction_data.values():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return queues + [pedestrian]   # length 6

def calculate_reward(norm_queue):
    """Negative normalised queue sum, clipped to [-10, 0]."""
    return float(np.clip(-sum(norm_queue), -10.0, 0.0))

def _train_agent(agent, episode_number, agent_label):
    """Run one A2C training update and apply entropy management."""
    actor_loss, critic_loss, entropy, total_reward = agent.train_on_episode()

    # Entropy management — same as Olivarez A2C
    MIN_ENTROPY = 1.0
    if entropy < MIN_ENTROPY:
        agent.entropy_coef = min(0.2, agent.entropy_coef * 1.1)
    else:
        agent.entropy_coef = max(0.01, agent.entropy_coef * 0.995)

    print(
        f"[{agent_label} | Ep {episode_number:4d}] "
        f"Reward: {total_reward:7.3f} | "
        f"Actor Loss: {actor_loss:7.4f} | "
        f"Critic Loss: {critic_loss:9.2f} | "
        f"Entropy: {entropy:5.3f} (coef: {agent.entropy_coef:.4f})"
    )
    return {'actor_loss': actor_loss, 'critic_loss': critic_loss,
            'entropy': entropy, 'total_reward': total_reward}

def save_history(filename, headers, episodes, rewards,
                 actor_losses, critic_losses, entropies):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        for i in range(len(episodes)):
            writer.writerow([episodes[i], rewards[i],
                             actor_losses[i], critic_losses[i], entropies[i]])

# ============================================================
# START SIMULATION
# ============================================================
traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("4902876117")
_junctionSubscription("12188714")

print("\n" + "=" * 70)
print("Starting MULTI-AGENT A2C Training - Balibago Network")
print("=" * 70)
print(f"  North Agent : e2_0-e2_7 + pedestrian | 8-phase cycle | state=13")
print(f"  South Agent : e2_8-e2_12 + pedestrian | 8-phase cycle | state=10")
print(f"  Train every : {TRAIN_EVERY} green decisions (= 1 full cycle per episode)")
print(f"  Models save : ./Balibago_traci/models_A2C/")
print("=" * 70 + "\n")

# ============================================================
# MAIN SIMULATION LOOP
# ============================================================
while traci.simulation.getMinExpectedNumber() > 0 and step_counter < 576000:
    step_counter += 1
    northCurrentPhaseDuration -= stepLength
    southCurrentPhaseDuration -= stepLength

    # Green-phase decision points only (timer expired + even phase index)
    north_decision_needed = (northCurrentPhaseDuration <= 0) and (northCurrentPhase % 2 == 0)
    south_decision_needed = (southCurrentPhaseDuration <= 0) and (southCurrentPhase % 2 == 0)

    obs_north    = None
    obs_south    = None
    norm_q_north = None
    norm_q_south = None

    # ----------------------------------------------------------
    # 1. OBSERVE CURRENT STATE
    # ----------------------------------------------------------
    if north_decision_needed:
        queue        = np.array(_northIntersection_queue())
        norm_q_north = queue / 2000.0  # Balanced for North's 100-12000 queue range
        n_phase_oh   = to_categorical(northCurrentPhase // 2, num_classes=4).flatten()
        obs_north    = np.concatenate([norm_q_north, n_phase_oh]).astype(np.float32)
        # shape: 9 queue + 4 phase-OH = 13 ✓

    if south_decision_needed:
        queue        = np.array(_southIntersection_queue())
        norm_q_south = queue / 1000.0
        s_phase_oh   = to_categorical(southCurrentPhase // 2, num_classes=4).flatten()
        obs_south    = np.concatenate([norm_q_south, s_phase_oh]).astype(np.float32)
        # shape: 6 queue + 4 phase-OH = 10 ✓

    # ----------------------------------------------------------
    # 2. STORE TRANSITION
    #    The reward for the PREVIOUS action is the queue we see NOW.
    #    store_transition() appends to agent.states/actions/rewards.
    # ----------------------------------------------------------
    if trainMode == 1:
        if north_decision_needed and northPrevState is not None:
            reward_north = calculate_reward(norm_q_north)  # norm_q_north already scaled by /3000
            NorthAgent.store_transition(northPrevState, northPrevAction, reward_north)
            north_buffer_count += 1

        if south_decision_needed and southPrevState is not None:
            reward_south = calculate_reward(norm_q_south)  # norm_q_south scaled by /1000
            SouthAgent.store_transition(southPrevState, southPrevAction, reward_south)
            south_buffer_count += 1

    # ----------------------------------------------------------
    # 3. TRAIN ONCE BUFFER HAS A FULL CYCLE OF TRANSITIONS
    #    train_on_episode() clears the buffer after each call,
    #    so we reset the counter too.
    #    Training happens BEFORE act() so the updated policy is
    #    used for the very next action.
    # ----------------------------------------------------------
    if trainMode == 1:
        if north_decision_needed and north_buffer_count >= TRAIN_EVERY:
            north_episode += 1
            metrics = _train_agent(NorthAgent, north_episode, "North")
            north_actor_loss_history.append(metrics['actor_loss'])
            north_critic_loss_history.append(metrics['critic_loss'])
            north_entropy_history.append(metrics['entropy'])
            north_reward_history.append(metrics['total_reward'])
            north_episode_steps.append(north_episode)
            north_buffer_count = 0  # buffer cleared by train_on_episode()

        if south_decision_needed and south_buffer_count >= TRAIN_EVERY:
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
        northPrevState    = obs_north
        northPrevAction   = next_action_N_idx
    elif northCurrentPhaseDuration <= 0:
        # Yellow phase — carry over last green action
        next_action_N_idx = northPrevAction if northPrevAction is not None else 5

    if south_decision_needed:
        next_action_S_idx = SouthAgent.act(obs_south, training=(trainMode == 1))
        southPrevState    = obs_south
        southPrevAction   = next_action_S_idx
    elif southCurrentPhaseDuration <= 0:
        next_action_S_idx = southPrevAction if southPrevAction is not None else 5

    # ----------------------------------------------------------
    # 5. APPLY PHASE TRANSITIONS TO SIMULATION
    # ----------------------------------------------------------

    # ---- North (8-phase cycle: 0-7) ----
    if northCurrentPhaseDuration <= 0:
        northCurrentPhase = (northCurrentPhase + 1) % 8
        traci.trafficlight.setPhase("4902876117", northCurrentPhase)

        if northCurrentPhase % 2 == 1:          # yellow/transition
            northCurrentPhaseDuration = 5
        else:                                    # green
            duration_adj = actionSpace[next_action_N_idx]
            base = {0: 45, 2: 130, 4: 30, 6: 90}.get(northCurrentPhase, 30)
            northCurrentPhaseDuration = max(5, min(180, base + duration_adj))

        traci.trafficlight.setPhaseDuration("4902876117", northCurrentPhaseDuration)

    # ---- South (8-phase cycle: 0-7) ----
    if southCurrentPhaseDuration <= 0:
        southCurrentPhase = (southCurrentPhase + 1) % 8
        traci.trafficlight.setPhase("12188714", southCurrentPhase)

        if southCurrentPhase % 2 == 1:          # yellow/transition
            southCurrentPhaseDuration = 5
        else:                                    # green
            duration_adj = actionSpace[next_action_S_idx]
            base = {0: 25, 2: 30, 4: 40, 6: 45}.get(southCurrentPhase, 30)
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
    print("  OK Models saved to ./Balibago_traci/models_A2C/")

    print("\nSaving training history...")
    headers = ['Episode', 'Total_Reward', 'Actor_Loss', 'Critic_Loss', 'Entropy']
    save_history('./Balibago_traci/output_A2C/North_A2C_history.csv', headers,
                 north_episode_steps, north_reward_history,
                 north_actor_loss_history, north_critic_loss_history, north_entropy_history)
    save_history('./Balibago_traci/output_A2C/South_A2C_history.csv', headers,
                 south_episode_steps, south_reward_history,
                 south_actor_loss_history, south_critic_loss_history, south_entropy_history)
    print("  OK North_A2C_history.csv saved.")
    print("  OK South_A2C_history.csv saved.")

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