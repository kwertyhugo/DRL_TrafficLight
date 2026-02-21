import os
import sys
import traci
import numpy as np
import csv
from keras.utils import to_categorical

# ==========================================
# PATH & MODEL SETUP
# ==========================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create output directories
os.makedirs('./Balibago_traci/output_DDPG', exist_ok=True)

try:
    from models.DDPG import DDPGAgent as ddpg
except ImportError:
    sys.exit("Error: Could not find DDPGAgent in models/DDPG.py")

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 64
TRAIN_FREQUENCY = 1  
GAMMA = 0.99
TAU = 0.001
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
NOISE_STD_INIT = 0.2
NOISE_DECAY = 0.9999

# Action bounds (DDPG outputs -1 to 1)
action_low = np.array([-1.0], dtype=np.float32)
action_high = np.array([1.0], dtype=np.float32)

# ==========================================
# AGENT INITIALIZATION
# ==========================================
# North: 8 Detectors + 4 Phase One-Hot = 12 State Size
NorthAgent = ddpg(
    state_size=12, 
    action_size=1, 
    action_low=action_low,
    action_high=action_high,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    name='North_DDPGAgent'
)

# South: 5 Detectors + 4 Phase One-Hot = 9 State Size
SouthAgent = ddpg(
    state_size=9, 
    action_size=1, 
    action_low=action_low,
    action_high=action_high,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    name='South_DDPGAgent'
)

# ==========================================
# SUMO SETUP
# ==========================================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo', # Use 'sumo-gui' to see the simulation
    '-c', 'Balibago_traci/signalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Balibago_traci/output_DDPG/jam1_stats.xml',
    '--tripinfo-output', r'Balibago_traci/output_DDPG/jam1_trips.xml'
]

# ==========================================
# SIMULATION VARIABLES
# ==========================================
trainMode = 0
stepLength = 0.1

northCurrentPhase = 0
northCurrentPhaseDuration = 30
southCurrentPhase = 0
southCurrentPhaseDuration = 30

# DDPG Action Scaling Factor (Action 1.0 = +25 seconds)
ACTION_SCALE = 25.0 

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
MAX_STEPS = 50000 # Safety limit (1000 seconds)

# History Buffers
reward_history_N, actor_loss_N, critic_loss_N = [], [], []
total_reward_N = 0

reward_history_S, actor_loss_S, critic_loss_S = [], [], []
total_reward_S = 0

# ==========================================
# HELPER FUNCTIONS
# ==========================================
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
    # Detectors 0-7
    queues = [_weighted_waits(f"e2_{i}") for i in range(8)]
    pedestrian = 0
    junction_data = traci.junction.getContextSubscriptionResults("4902876117")
    if junction_data:
        for data in junction_data.values():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return queues + [pedestrian]

def _southIntersection_queue():
    # Detectors 8-12
    queues = [_weighted_waits(f"e2_{i}") for i in range(8, 13)]
    pedestrian = 0
    junction_data = traci.junction.getContextSubscriptionResults("12188714")
    if junction_data:
        for data in junction_data.values():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return queues + [pedestrian]

def calculate_reward(current_state):
    # Negative sum of queues (normalized input)
    if current_state is None: return 0
    return -np.sum(current_state)

def save_history(filename, headers, reward_hist, a_loss_hist, c_loss_hist):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        # Pad shorter lists with None to match length if necessary, 
        # though in this loop structure they should be synced.
        length = len(reward_hist)
        for i in range(length):
            # Safe access in case lists differ slightly due to timing
            r = reward_hist[i] if i < len(reward_hist) else 0
            a = a_loss_hist[i] if i < len(a_loss_hist) else 0
            c = c_loss_hist[i] if i < len(c_loss_hist) else 0
            writer.writerow([i, r, a, c])

# ==========================================
# MAIN EXECUTION
# ==========================================
traci.start(Sumo_config, port=8814)
_subscribe_all_detectors()
_junctionSubscription("4902876117")
_junctionSubscription("12188714")

print(f"STARTING DDPG | Unified Mode")

try:
    while traci.simulation.getMinExpectedNumber() > 0 and step_counter < MAX_STEPS:
        step_counter += 1
        
        northCurrentPhaseDuration -= stepLength
        southCurrentPhaseDuration -= stepLength

        # -------------------------------------------------
        # 1. OBSERVATION PHASE
        # -------------------------------------------------
        obs_north = None
        obs_south = None
        reward_north = 0
        reward_south = 0
        
        # Check Decision Needs
        north_decision_needed = (northCurrentPhaseDuration <= 0) and (northCurrentPhase % 2 == 0)
        south_decision_needed = (southCurrentPhaseDuration <= 0) and (southCurrentPhase % 2 == 0)

        # --- Capture North State ---
        if north_decision_needed:
            raw_q = _northIntersection_queue() 
            queue = np.array(raw_q[:8]) 
            n_norm_queue = queue / 100.0 
            
            n_phase_oh = to_categorical(northCurrentPhase//2, num_classes=4).flatten()
            obs_north = np.concatenate([n_norm_queue, n_phase_oh]).astype(np.float32)

            if trainMode == 1:
                reward_north = calculate_reward(n_norm_queue * 10)
                total_reward_N += reward_north

        # --- Capture South State ---
        if south_decision_needed:
            raw_q = _southIntersection_queue()
            queue = np.array(raw_q[:5]) 
            s_norm_queue = queue / 100.0
            
            s_phase_oh = to_categorical(southCurrentPhase//2, num_classes=4).flatten()
            obs_south = np.concatenate([s_norm_queue, s_phase_oh]).astype(np.float32) 

            if trainMode == 1:
                reward_south = calculate_reward(s_norm_queue * 10)
                total_reward_S += reward_south

        # -------------------------------------------------
        # 2. MEMORY PHASE
        # -------------------------------------------------
        if trainMode == 1:
            if north_decision_needed and northPrevState is not None:
                NorthAgent.remember(northPrevState, northPrevAction, reward_north, obs_north, False)
                if len(NorthAgent.replay_buffer) >= BATCH_SIZE:
                    a_loss, c_loss = NorthAgent.train()
                    if a_loss is not None:
                        actor_loss_N.append(a_loss)
                        critic_loss_N.append(c_loss)
                        reward_history_N.append(total_reward_N)
                        total_reward_N = 0

            if south_decision_needed and southPrevState is not None:
                SouthAgent.remember(southPrevState, southPrevAction, reward_south, obs_south, False)
                if len(SouthAgent.replay_buffer) >= BATCH_SIZE:
                    a_loss, c_loss = SouthAgent.train()
                    if a_loss is not None:
                        actor_loss_S.append(a_loss)
                        critic_loss_S.append(c_loss)
                        reward_history_S.append(total_reward_S)
                        total_reward_S = 0

        # -------------------------------------------------
        # 3. ACTION SELECTION PHASE
        # -------------------------------------------------
        north_action_val = None
        south_action_val = None

        # NORTH
        if north_decision_needed:
            north_action_val = NorthAgent.get_action(obs_north, add_noise=(trainMode==1))
            northPrevState = obs_north
            northPrevAction = north_action_val
            
            if trainMode == 1:
                print(f"North | Q: {np.sum(obs_north[:8]):.2f} | R: {reward_north:.2f} | Act: {north_action_val[0]:.2f}")
        
        elif northCurrentPhaseDuration <= 0:
            north_action_val = northPrevAction if northPrevAction is not None else np.zeros(1)

        # SOUTH
        if south_decision_needed:
            south_action_val = SouthAgent.get_action(obs_south, add_noise=(trainMode==1))
            southPrevState = obs_south
            southPrevAction = south_action_val
            
            if trainMode == 1:
                print(f"South | Q: {np.sum(obs_south[:5]):.2f} | R: {reward_south:.2f} | Act: {south_action_val[0]:.2f}")
        
        elif southCurrentPhaseDuration <= 0:
            south_action_val = southPrevAction if southPrevAction is not None else np.zeros(1)

        # -------------------------------------------------
        # 4. EXECUTION PHASE
        # -------------------------------------------------

        # Apply North
        if northCurrentPhaseDuration <= 0:
            northCurrentPhase = (northCurrentPhase + 1) % 8
            traci.trafficlight.setPhase("4902876117", northCurrentPhase)

            if northCurrentPhase % 2 == 1:
                northCurrentPhaseDuration = 5 # Yellow
            else:
                base = {0: 45, 2: 130, 4: 30, 6: 90}.get(northCurrentPhase, 30)
                adjustment = float(north_action_val[0]) * ACTION_SCALE
                northCurrentPhaseDuration = max(5.0, min(180.0, base + adjustment))
            
            traci.trafficlight.setPhaseDuration("4902876117", float(northCurrentPhaseDuration))

        # Apply South
        if southCurrentPhaseDuration <= 0:
            southCurrentPhase = (southCurrentPhase + 1) % 8
            traci.trafficlight.setPhase("12188714", southCurrentPhase)

            if southCurrentPhase % 2 == 1:
                southCurrentPhaseDuration = 5 # Yellow
            else:
                base = {0: 25, 2: 30, 4: 40, 6: 45}.get(southCurrentPhase, 30)
                adjustment = float(south_action_val[0]) * ACTION_SCALE
                southCurrentPhaseDuration = max(5.0, min(180.0, base + adjustment))
            
            traci.trafficlight.setPhaseDuration("12188714", float(southCurrentPhaseDuration))

        # -------------------------------------------------
        # 5. METRICS & LOGGING
        # -------------------------------------------------
        # Use simple modulo to track every N steps
        if step_counter % 600 == 0: # Check roughly every minute (60s / 0.1s step = 600 steps)
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

        # Simulation Step
        traci.simulationStep()

except Exception as e:
    print(f"Simulation interrupted: {e}")
    import traceback
    traceback.print_exc()

finally:
    # -------------------------------------------------
    # END OF SIMULATION & SAVING
    # -------------------------------------------------
    print("Cleaning up...")
    try:
        traci.close()
    except:
        pass # Ignore close errors if already closed

    if metric_observation_count > 0:
        print("\n Queue Length Avg:", jam_length_total / metric_observation_count)
        print("\n Throughput Avg:", throughput_total / metric_observation_count)

    if trainMode == 1:
        print("Saving trained models...")
        NorthAgent.save()
        SouthAgent.save()

        print("Saving training history...")
        headers = ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss']
        
        save_history('./Balibago_traci/output_DDPG/North_history.csv', headers, 
                    reward_history_N, actor_loss_N, critic_loss_N)
        
        save_history('./Balibago_traci/output_DDPG/South_history.csv', headers, 
                    reward_history_S, actor_loss_S, critic_loss_S)
        
        print("All histories saved successfully!")

    print("Simulation Finished.")