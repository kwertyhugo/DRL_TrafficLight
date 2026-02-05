import os
import sys
import traci
import numpy as np
import csv
from models.DDPG import DDPGAgent as ddpg

# ------- Configuration -------
TRAIN_MODE = 1 
reward_scale = 10.0  
noise_decay = 0.9995 
min_noise_std = 0.01

# Normalization tracking (11 dimensions)
state_mean_global = np.zeros(11, dtype=np.float32)
state_std_global = np.ones(11, dtype=np.float32)

detector_ids = ["e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10", "e2_0", "e2_1", "e2_2", "e2_3"]

action_low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

globalAgent = ddpg(
    state_size=11,
    action_size=3,
    action_low=action_low,
    action_high=action_high,
    actor_lr=0.0001,
    critic_lr=0.001,
    name='Global_Olivarez_DDPG'
)

# Global variables for simulation state
mainCurrentPhase = 0
mainCurrentPhaseDuration = 30.0
swCurrentPhase = 0
swCurrentPhaseDuration = 30.0
seCurrentPhase = 0
seCurrentPhaseDuration = 30.0
stepLength = 0.05
step_counter = 0

reward_history = []
actor_loss_history = []
critic_loss_history = []
cumulative_reward = 0
TRAIN_FREQUENCY = 100 # Matches history logic

def normalize_state(state, mean, std_var):
    """Normalize state by running mean/std_var for stability."""
    return (state - mean) / (np.sqrt(std_var) + 1e-8)

def calculate_reward(unnormalized_state):
    """Calculate reward based on total queue (negative waiting time)."""
    if unnormalized_state is None:
        return 0.0
    total_wait = float(np.sum(unnormalized_state))
    return -total_wait

def save_history(filename, headers, reward_hist, actor_loss_hist, critic_loss_hist, train_frequency):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    existing_rows = 0
    
    if file_exists:
        with open(filename, 'r') as f:
            try:
                existing_rows = sum(1 for _ in f) - 1  # minus header line
            except:
                existing_rows = 0 # Handle empty file case
    
    if existing_rows < 0:
        existing_rows = 0

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists or existing_rows == 0:
            writer.writerow(headers)

        # Write only new history data
        start_index = existing_rows
        for i in range(start_index, len(reward_hist)):
            writer.writerow([
                i * train_frequency,
                reward_hist[i],
                actor_loss_hist[i],
                critic_loss_hist[i]
            ])

# ------- Phase Application (Fixed Scoping) -------

def _mainIntersection_phase(action):
    global mainCurrentPhase, mainCurrentPhaseDuration
    mainCurrentPhase = (mainCurrentPhase + 1) % 10
    duration_adjustment = float(np.clip(action[0], -1.0, 1.0) * 5.0)
    
    if mainCurrentPhase in [2, 4]:
        base_duration = 15.0
    elif mainCurrentPhase % 2 == 0:
        base_duration = 30.0
    else:
        base_duration = 3.0

    mainCurrentPhaseDuration = float(max(5.0, min(60.0, base_duration + duration_adjustment)))
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", mainCurrentPhase)
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", mainCurrentPhaseDuration)

def _swPedXing_phase(action):
    global swCurrentPhase, swCurrentPhaseDuration
    swCurrentPhase = (swCurrentPhase + 1) % 10
    duration_adjustment = float(np.clip(action[0], -1.0, 1.0) * 5.0)
    base_duration = 5.0 if swCurrentPhase % 2 == 1 else 30.0
    swCurrentPhaseDuration = float(max(5.0, min(60.0, base_duration + duration_adjustment)))
    traci.trafficlight.setPhase("6401523012", swCurrentPhase)
    traci.trafficlight.setPhaseDuration("6401523012", swCurrentPhaseDuration)

def _sePedXing_phase(action):
    global seCurrentPhase, seCurrentPhaseDuration
    seCurrentPhase = (seCurrentPhase + 1) % 10
    duration_adjustment = float(np.clip(action[0], -1.0, 1.0) * 5.0)
    base_duration = 5.0 if seCurrentPhase % 2 == 1 else 30.0
    seCurrentPhaseDuration = float(max(5.0, min(60.0, base_duration + duration_adjustment)))
    traci.trafficlight.setPhase("3285696417", seCurrentPhase)
    traci.trafficlight.setPhaseDuration("3285696417", seCurrentPhaseDuration)

# ------- State Consolidation & Helpers -------

def _weighted_waits(detector_id):
    sumWait = 0
    try:
        vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
        if not vehicle_data: return 0
        weights = {"car": 1.0, "jeep": 1.5, "bus": 2.2, "truck": 2.5, "motorcycle": 0.3, "tricycle": 0.5}
        for data in vehicle_data.values():
            vtype = data.get(traci.constants.VAR_TYPE, "car")
            waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
            sumWait += waitTime * weights.get(vtype, 1.0)
    except: return 0
    return sumWait

def get_global_raw_state():
    # Main Junction
    m_sw = _weighted_waits("e2_4") + _weighted_waits("e2_5")
    m_se = _weighted_waits("e2_6") + _weighted_waits("e2_7")
    m_ne = _weighted_waits("e2_8")
    m_nw = _weighted_waits("e2_9") + _weighted_waits("e2_10")
    m_ped = 0
    sub_m = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
    if sub_m: 
        for pid, data in sub_m.items(): m_ped += data.get(traci.constants.VAR_WAITING_TIME, 0)

    # SW Crossing
    sw_n = _weighted_waits("e2_0") + _weighted_waits("e2_1")
    sw_s = _weighted_waits("e2_4") + _weighted_waits("e2_5")
    sw_ped = 0
    sub_sw = traci.junction.getContextSubscriptionResults("6401523012")
    if sub_sw:
        for pid, data in sub_sw.items(): sw_ped += data.get(traci.constants.VAR_WAITING_TIME, 0)

    # SE Crossing
    se_w = _weighted_waits("e2_2") + _weighted_waits("e2_3")
    se_e = _weighted_waits("e2_6") + _weighted_waits("e2_7")
    se_ped = 0
    sub_se = traci.junction.getContextSubscriptionResults("3285696417")
    if sub_se:
        for pid, data in sub_se.items(): se_ped += data.get(traci.constants.VAR_WAITING_TIME, 0)

    return np.array([m_sw, m_se, m_ne, m_nw, m_ped, sw_n, sw_s, sw_ped, se_w, se_e, se_ped], dtype=np.float32)

# ------- Simulation Loop -------

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Declare SUMO_HOME")

traci.start(['sumo', '-c', 'Olivarez_traci/signalizedPed.sumocfg', '--step-length', '0.05'])

# Initial Subscriptions
for junc in ["cluster_295373794_3477931123_7465167861", "6401523012", "3285696417"]:
    traci.junction.subscribeContext(junc, traci.constants.CMD_GET_PERSON_VARIABLE, 10.0, [traci.constants.VAR_WAITING_TIME])

vehicle_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
for det in detector_ids:
    traci.lanearea.subscribeContext(det, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_vars)

prev_state = None
prev_action = None
globalPhaseDuration = 30.0



while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1
    globalPhaseDuration -= stepLength

    if globalPhaseDuration <= 0:
        raw_state = get_global_raw_state()
        
        # Normalization logic
        state_mean_global = 0.99 * state_mean_global + 0.01 * raw_state
        state_std_global = 0.99 * state_std_global + 0.01 * np.square(raw_state - state_mean_global)
        current_state = normalize_state(raw_state, state_mean_global, state_std_global)

        # Reward logic
        reward = calculate_reward(raw_state) / reward_scale
        cumulative_reward += reward

        if TRAIN_MODE == 1 and prev_state is not None:
            globalAgent.remember(prev_state, prev_action, reward, current_state, False)
            if len(globalAgent.replay_buffer) >= 128:
                a_loss, c_loss = globalAgent.train()
                if a_loss:
                    actor_loss_history.append(a_loss)
                    critic_loss_history.append(c_loss)
                    reward_history.append(cumulative_reward)
                    cumulative_reward = 0

        # Noise Decay
        if TRAIN_MODE == 1:
            globalAgent.noise.std_dev = max(min_noise_std, globalAgent.noise.std_dev * noise_decay)

        action = globalAgent.get_action(current_state, add_noise=(TRAIN_MODE == 1))
        
        # Apply phases to all 3 junctions using the specific action vector indices
        _mainIntersection_phase([action[0]])
        _swPedXing_phase([action[1]])
        _sePedXing_phase([action[2]])
        
        # Sync the global timer to the main junction's newly calculated duration
        globalPhaseDuration = mainCurrentPhaseDuration 
        
        prev_state = current_state
        prev_action = action
        print(f"Step {step_counter} | Total Queue: {np.sum(raw_state):.1f} | Reward: {reward:.2f}")

    traci.simulationStep()

# ------- Wrap Up & Final Save -------
traci.close()

if TRAIN_MODE == 1:
    globalAgent.save()
    globalAgent.save_replay_buffer()
    save_history('./Olivarez_traci/output_DDPG/global_agent_history.csv', 
                 ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss'],
                 reward_history, actor_loss_history, critic_loss_history, TRAIN_FREQUENCY)

print("Simulation Done!")