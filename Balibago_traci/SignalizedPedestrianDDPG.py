import os
import sys
import traci
import numpy as np
import csv
from keras.utils import to_categorical

# ==========================================
# PATH & MODEL SETUP
# ==========================================
# Path setup to find models folder (assumes models folder is in parent directory)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create output directories if they don't exist
os.makedirs('./Balibago_traci/output_DDPG', exist_ok=True)

try:
    from models.DDPG import DDPGAgent as ddpg
except ImportError:
    sys.exit("Error: Could not find DDPGAgent in models/DDPG.py. Ensure your directory structure is correct.")

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_MODE = 1          # 1 = Train (Learn), 0 = Test (Run without learning)
REWARD_SCALE = 10.0     # Scale down reward for stability
NOISE_DECAY = 0.9995    # How fast exploration noise decreases
MIN_NOISE_STD = 0.01    # Minimum noise level
BATCH_SIZE = 64
GAMMA = 0.99

# State: 13 vehicle detectors (No phase encoding or ped wait included to keep exactly 13)
STATE_SIZE = 13

# Action: 3 continuous values (Junction North, Junction South, and a spare/West adjustment)
ACTION_SIZE = 3

# Action bounds (DDPG outputs -1 to 1)
action_low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

# IDs from your road network
JUNCTION_NORTH = "4902876117"
JUNCTION_SOUTH = "12188714"
DETECTOR_IDS = [
    "e2_0", "e2_1", "e2_2", "e2_3", "e2_4", "e2_5", "e2_6", "e2_7", 
    "e2_8", "e2_9", "e2_10", "e2_11", "e2_12"
]

# ==========================================
# AGENT INITIALIZATION
# ==========================================
trafficLightAgent = ddpg(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    action_low=action_low,
    action_high=action_high,
    actor_lr=0.0001,
    critic_lr=0.001,
    name='Global_Balibago_DDPG_13_3'
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
    'sumo',
    '-c', 'Balibago_traci/signalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Balibago_traci\output_DDPG\stats.xml',
    '--tripinfo-output', r'Balibago_traci\output_DDPG\trips.xml'
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_state_13():
    """Retrieves the waiting times for exactly 13 detectors."""
    state_vector = []
    weights = {"car": 1.0, "jeep": 1.5, "bus": 2.2, "truck": 2.5, "motorcycle": 0.3, "tricycle": 0.5}
    
    for det in DETECTOR_IDS:
        sum_wait = 0
        try:
            vehicle_data = traci.lanearea.getContextSubscriptionResults(det)
            if vehicle_data:
                for data in vehicle_data.values():
                    vtype = data.get(traci.constants.VAR_TYPE, "car")
                    waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
                    sum_wait += waitTime * weights.get(vtype, 1.0)
        except:
            sum_wait = 0
        state_vector.append(sum_wait / 100.0) # Normalize for DDPG stability
        
    return np.array(state_vector, dtype=np.float32)

def apply_3_actions(action_vals, phase):
    """
    Applies the 3 continuous action outputs to the Green Phase durations.
    action_vals[0] -> North Junction
    action_vals[1] -> South Junction
    action_vals[2] -> Global Offset/Spare
    """
    # Define Base Durations (matching your DQN logic)
    base_n = {0: 45, 2: 130, 4: 30, 6: 90}.get(phase, 30)
    base_s = {0: 25, 2: 30, 4: 40, 6: 45}.get(phase, 30)

    # Scaling: (Action * 25) allows a swing of -25s to +25s
    # Action 2 is used here as an additional global offset for both
    global_offset = action_vals[2] * 5.0 
    
    dur_n = float(np.clip(base_n + (action_vals[0] * 25.0) + global_offset, 10.0, 180.0))
    dur_s = float(np.clip(base_s + (action_vals[1] * 25.0) + global_offset, 10.0, 180.0))

    traci.trafficlight.setPhaseDuration(JUNCTION_NORTH, dur_n)
    traci.trafficlight.setPhaseDuration(JUNCTION_SOUTH, dur_s)
    
    return max(dur_n, dur_s)

def save_training_history(filename, reward_hist, actor_loss_hist, critic_loss_hist):
    headers = ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss']
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        for i in range(len(reward_hist)):
            writer.writerow([i, reward_hist[i], actor_loss_hist[i], critic_loss_hist[i]])

# ==========================================
# MAIN EXECUTION
# ==========================================

traci.start(Sumo_config)

# Subscriptions
for det in DETECTOR_IDS:
    traci.lanearea.subscribeContext(det, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, 
                                   [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME])

# Simulation variables
step_counter = 0
current_phase = 0
timer_until_next_decision = 0
step_length = 0.1

# Buffers for history
reward_history = []
actor_loss_history = []
critic_loss_history = []
cum_reward = 0

prev_state = None
prev_action = None

print("=" * 50)
print(f"STARTING DDPG | State: {STATE_SIZE} | Action: {ACTION_SIZE}")
print("=" * 50)

try:
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step_counter += 1
        timer_until_next_decision -= step_length

        if timer_until_next_decision <= 0:
            # 1. Capture State and Reward
            current_state = get_state_13()
            # Reward is negative sum of waiting times (normalized)
            reward = -np.sum(current_state) / REWARD_SCALE
            cum_reward += reward

            # 2. Memory & Training
            if TRAIN_MODE == 1 and prev_state is not None:
                trafficLightAgent.remember(prev_state, prev_action, reward, current_state, False)
                
                if len(trafficLightAgent.replay_buffer) >= BATCH_SIZE:
                    a_loss, c_loss = trafficLightAgent.train()
                    if a_loss is not None:
                        actor_loss_history.append(a_loss)
                        critic_loss_history.append(c_loss)
                        reward_history.append(cum_reward)
                        cum_reward = 0

            # 3. Phase Transition Logic
            # Transition to next phase (Yellow or Green)
            current_phase = (current_phase + 1) % 8
            traci.trafficlight.setPhase(JUNCTION_NORTH, current_phase)
            traci.trafficlight.setPhase(JUNCTION_SOUTH, current_phase)

            if current_phase % 2 == 1:
                # Yellow Phase (Fixed)
                timer_until_next_decision = 5.0
                traci.trafficlight.setPhaseDuration(JUNCTION_NORTH, 5.0)
                traci.trafficlight.setPhaseDuration(JUNCTION_SOUTH, 5.0)
                # Keep prev_action for yellow phases
                actual_action = prev_action if prev_action is not None else np.zeros(ACTION_SIZE)
            else:
                # Green Phase (Variable via DDPG)
                action = trafficLightAgent.get_action(current_state, add_noise=(TRAIN_MODE == 1))
                timer_until_next_decision = apply_3_actions(action, current_phase)
                
                prev_state = current_state
                prev_action = action
                actual_action = action

            # 4. Decay Noise
            if TRAIN_MODE == 1:
                trafficLightAgent.noise.std_dev = max(MIN_NOISE_STD, trafficLightAgent.noise.std_dev * NOISE_DECAY)

            # Debug Print
            if step_counter % 1000 == 0:
                print(f"Step {step_counter} | Reward: {reward:.2f} | Action: {np.round(actual_action, 2)}")

except Exception as e:
    print(f"Simulation interrupted: {e}")

finally:
    print("Cleaning up...")
    traci.close()

    if TRAIN_MODE == 1:
        print("Saving Model and History...")
        trafficLightAgent.save()
        save_training_history('./Balibago_traci/output_DDPG/training_log.csv', 
                             reward_history, actor_loss_history, critic_loss_history)
        print("Done.")

print("Simulation Finished.")