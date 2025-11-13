import os
import sys
import traci
import numpy as np
import csv

# Import your DDPG implementation
from models.DDPG import DDPGAgent as ddpg

# ------- Configuration / Agents -------

# --- NEW STABILITY FEATURES ---
# Normalization tracking variables
state_mean = np.zeros(5, dtype=np.float32)  # For main intersection (size 5)
state_std = np.ones(5, dtype=np.float32)
reward_scale = 10.0  # reward normalization divisor
noise_decay = 0.9995 # Slower decay for long runs
min_noise_std = 0.01   # Minimum noise level
# ------------------------------

# Define continuous action bounds (vector or scalar)
action_low = np.array([-1.0], dtype=np.float32)
action_high = np.array([1.0], dtype=np.float32)

# --------------------------------------------

# --- TRAIN/TEST TOGGLE ---
# 1 = Train (collect experience, update models, save models, decay noise)
# 0 = Test (load models, no noise, no updates, no saving)
TRAIN_MODE = 1
# -------------------------

# Create agents with proper state sizes
mainIntersectionAgent = ddpg(
    state_size=5,
    action_size=1,
    action_low=action_low,
    action_high=action_high,
    actor_lr = 0.0001,
    critic_lr = 0.001,
    gamma=0.99,
    tau=0.001,
    buffer_size=10000,
    batch_size=128,
    name='MainIntersection_DDPGAgent_one_trafficlight'
)

# --- STARTING FRESH / LOADING ---
if TRAIN_MODE == 0:
    # TEST MODE: Load weights, do not load buffer
    mainIntersectionAgent.load()
    print("[INFO] TEST_MODE: Loading weights for testing.")
else:
    # TRAIN MODE: Decide whether to load or start fresh
    # mainIntersectionAgent.load() # Uncomment to resume training
    # mainIntersectionAgent.load_replay_buffer() # Uncomment to resume training
    print("[INFO] TRAIN_MODE: Starting a fresh training run. Old weights/buffers will NOT be loaded.")
# ----------------------

# Load training histories (this is OK to append to old history)
main_reward_history, main_actor_loss_history, main_critic_loss_history = mainIntersectionAgent.load_history('./Olivarez_traci/output_DDPG/main_ddpg_history_one_trafficlight.csv')


# ------- SUMO start-up -------

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Build sumo command
sumo_cfg_path = os.path.join('Olivarez_traci', 'signalizedPed.sumocfg')
Sumo_config = [
    'sumo', # Use 'sumo' (no GUI) for fast training
    '-c', sumo_cfg_path,
    '--step-length', '0.05',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_DDPG\SD_DDPG_one_trafficlight_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_DDPG\SD_DDPG_one_trafficlight_trips.xml'
]

# ------- Helpers & subscriptions -------

def _junctionSubscription(junction_id):
    """Subscribe to junction for pedestrian data"""
    try:
        traci.junction.subscribeContext(
            junction_id,
            traci.constants.CMD_GET_PERSON_VARIABLE,
            10.0,
            [traci.constants.VAR_WAITING_TIME]
        )
    except Exception as e:
        print(f"[WARN] Junction subscription failed for {junction_id}: {e}")

def _subscribe_all_detectors():
    """Subscribe to all vehicle detectors for efficient data retrieval"""
    detector_ids = [
        "e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10",
        "e2_0", "e2_1", "e2_2", "e2_3"
    ]
    
    vehicle_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    
    for det_id in detector_ids:
        try:
            traci.lanearea.subscribeContext(
                det_id,
                traci.constants.CMD_GET_VEHICLE_VARIABLE,
                3,
                vehicle_vars
            )
        except Exception as e:
            print(f"[WARN] Detector subscription failed for {det_id}: {e}")

# ------- Simulation variables -------

stepLength = 0.05
mainCurrentPhase = 0
mainCurrentPhaseDuration = 30.0

# Keep previous state for reward calculation
mainPrevState = None
mainPrevAction = None

# Training params
BATCH_SIZE = 128 # Use the stable batch size
TRAIN_FREQUENCY = 100  # training every 100 simulation steps
step_counter = 0

# Data storage for plotting
if not main_reward_history:
    main_reward_history = []
    main_actor_loss_history = []
    main_critic_loss_history = []

# Accumulate rewards between training steps
total_main_reward = 0

# ------- Environment state functions -------

def _weighted_waits(detector_id):
    """Calculate weighted waiting times from detector"""
    sumWait = 0
    try:
        vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
        
        if not vehicle_data:
            return 0
        
        for data in vehicle_data.values():
            vtype = data.get(traci.constants.VAR_TYPE, "car")
            waitTime = data.get(traci.constants.VAR_WAITING_TIME, 0)
            
            # Custom weighting logic
            if vtype == "car":
                sumWait += waitTime
            elif vtype == "jeep":
                sumWait += waitTime * 1.5
            elif vtype == "bus":
                sumWait += waitTime * 2.2
            elif vtype == "truck":
                sumWait += waitTime * 2.5
            elif vtype == "motorcycle":
                sumWait += waitTime * 0.3
            elif vtype == "tricycle":
                sumWait += waitTime * 0.5
    except Exception:
        return 0
    
    return sumWait

def _mainIntersection_queue():
    """Get state for main intersection (Returns UN-NORMALIZED data)"""
    southwest = _weighted_waits("e2_4") + _weighted_waits("e2_5")
    southeast = _weighted_waits("e2_6") + _weighted_waits("e2_7")
    northeast = _weighted_waits("e2_8")
    northwest = _weighted_waits("e2_9") + _weighted_waits("e2_10")
    
    pedestrian = 0
    try:
        junction_subscription = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
        if junction_subscription:
            for pid, data in junction_subscription.items():
                pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    except Exception:
        pass
    
    return np.array([southwest, southeast, northeast, northwest, pedestrian], dtype=np.float32)

def normalize_state(state, mean, std):
    """Normalize state by running mean/std for stability."""
    state_len = len(state)
    return (state - mean[:state_len]) / (std[:state_len] + 1e-8)

# ------- Reward calculation -------

def calculate_reward(unnormalized_state):
    """
    Calculate reward based on total queue (negative waiting time).
    Input state should be UN-NORMALIZED.
    """
    if unnormalized_state is None:
        return 0.0
    
    # The state is already the raw waiting times, just sum them
    total_wait = float(np.sum(unnormalized_state))
    
    # Negative reward proportional to queue (we want to minimize waiting)
    # This value is scaled by reward_scale in the main loop
    reward = -total_wait
    
    return reward

# ------- Phase application functions -------

def _mainIntersection_phase(action):
    """Apply continuous action to main intersection phase duration"""
    global mainCurrentPhase, mainCurrentPhaseDuration

    mainCurrentPhase = (mainCurrentPhase + 1) % 10

    # Map action from [-1, 1] to duration adjustment [-5, 5]
    duration_adjustment = float(np.clip(action[0], -1.0, 1.0) * 5.0)
    
    # Compute base duration
    if mainCurrentPhase == 2 or mainCurrentPhase == 4:
        base_duration = 15.0
    elif mainCurrentPhase % 2 == 0:
        base_duration = 30.0
    else:
        base_duration = 3.0

    # Apply adjustment and clamp to [5, 60]
    mainCurrentPhaseDuration = float(max(5.0, min(60.0, base_duration + duration_adjustment)))

    try:
        traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", mainCurrentPhase)
        traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", mainCurrentPhaseDuration)
    except Exception as e:
        print(f"[WARN] setPhase failed for main intersection: {e}")

# ------- History saving function -------

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
            
# ------- Start SUMO and run simulation loop -------

print("Starting SUMO...")
traci.start(Sumo_config)

print("Setting up subscriptions...")
_subscribe_all_detectors()
_junctionSubscription("cluster_295373794_3477931123_7465167861")
_junctionSubscription("6401523012")
_junctionSubscription("3285696417")

if TRAIN_MODE == 1:
    print("Resetting DDPG exploration noise...")
    mainIntersectionAgent.noise.reset()
    # Set initial noise level (std_dev)
    if hasattr(mainIntersectionAgent.noise, 'std_dev'):
         mainIntersectionAgent.noise.std_dev = 0.2 # Starting noise level
    else:
         print("[WARN] Could not set initial noise std_dev. Check DDPGAgent implementation.")

print("Starting simulation loop...")

# Flag for whether to use exploration noise
USE_NOISE = (TRAIN_MODE == 1)

while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1

    # ---- Main intersection logic ----
    mainCurrentPhaseDuration -= stepLength
    if mainCurrentPhaseDuration <= 0:
        
        # --- Get state (raw, un-normalized) ---
        raw_state = _mainIntersection_queue()

        # --- Update running mean/std (for state normalization) ---
        # Use exponential moving average
        state_mean = 0.99 * state_mean + 0.01 * raw_state
        state_std_var = 0.99 * state_std + 0.01 * np.square(raw_state - state_mean) # This is variance
        
        # Normalize the state
        mainCurrentState = normalize_state(raw_state, state_mean, np.sqrt(state_std_var))

        # --- Normalize reward ---
        # Calculate reward on raw state, then scale it
        mainReward = calculate_reward(raw_state) / reward_scale
        total_main_reward += mainReward

        # --- Store experience (Normalized State, Normalized Reward) ---
        # Only remember and train if in TRAIN_MODE
        if TRAIN_MODE == 1 and mainPrevState is not None and mainPrevAction is not None:
            done = False
            mainIntersectionAgent.remember(
                mainPrevState, 
                mainPrevAction, 
                mainReward, 
                mainCurrentState, 
                done
            )

        # --- Action + noise decay ---
        if TRAIN_MODE == 1:
            # Decay the noise std_dev based on the decay rate
            if hasattr(mainIntersectionAgent.noise, 'std_dev'):
                 mainIntersectionAgent.noise.std_dev = max(min_noise_std, mainIntersectionAgent.noise.std_dev * noise_decay)
                 current_noise_std = mainIntersectionAgent.noise.std_dev
            else:
                 current_noise_std = 0.0 # Fallback if attribute missing
        else:
            current_noise_std = 0.0 # No noise in test mode
             
        mainAction = mainIntersectionAgent.get_action(mainCurrentState, add_noise=USE_NOISE)
        _mainIntersection_phase(mainAction)

        mainPrevState = mainCurrentState
        mainPrevAction = mainAction

        print(f"[MAIN] Queue={np.sum(raw_state):.1f}, Reward={mainReward:.3f}, Action={mainAction[0]:.3f}, Noise={current_noise_std:.3f}")

    # ---- Periodic training using replay buffer ----
    # Only train if in TRAIN_MODE
    if TRAIN_MODE == 1 and step_counter % TRAIN_FREQUENCY == 0:
        # Train Main
        if len(mainIntersectionAgent.replay_buffer) >= BATCH_SIZE:
            actor_loss, critic_loss = mainIntersectionAgent.train()
            
            # Check for non-numeric loss values which indicate explosion
            if actor_loss is None or np.isnan(actor_loss) or np.isnan(critic_loss):
                print(f"[ERROR] Training failed at step {step_counter}. Actor/Critic loss is None or NaN. Stopping.")
                break # Stop simulation if training exploded
            
            main_actor_loss_history.append(float(actor_loss))
            main_critic_loss_history.append(float(critic_loss))
            main_reward_history.append(total_main_reward)
            total_main_reward = 0
            print(f"[TRAIN MAIN] Step={step_counter}, Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")

    # Step simulation forward
    traci.simulationStep()

print("\nSimulation completed!")

# Save models and buffers ONLY if in training mode
if TRAIN_MODE == 1:
    # Save trained models
    print("Saving trained models...")
    try:
        mainIntersectionAgent.save()
        print("Models saved successfully!")
    except Exception as e:
        print(f"[ERROR] Could not save models: {e}")
        
    # ------- Save Replay Buffers -------
    print("Saving replay buffers...")
    try:
        mainIntersectionAgent.save_replay_buffer()
        print("Replay buffers saved successfully!")
    except Exception as e:
        print(f"[ERROR] Could not save replay buffers: {e}")

    # Save training history
    print("Saving training history...")
    try:
        save_history(
            './Olivarez_traci/output_DDPG/main_ddpg_history_one_trafficlight.csv',
            ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss'],
            main_reward_history,
            main_actor_loss_history,
            main_critic_loss_history,
            TRAIN_FREQUENCY
        )
        
        print("History saved successfully!")
    except Exception as e:
        print(f"[ERROR] Could not save history: {e}")
else:
    print("[INFO] Test mode complete. No models, buffers, or history saved.")
    

traci.close()
print("Done!")

