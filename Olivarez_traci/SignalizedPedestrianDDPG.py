import os
import sys
import traci
import numpy as np
import csv

# Import your DDPG implementation
from models.DDPG import DDPGAgent as ddpg

# ------- Configuration / Agents -------

# Define continuous action bounds (vector or scalar)
action_low = np.array([-1.0], dtype=np.float32)
action_high = np.array([1.0], dtype=np.float32)

# Create agents with proper state sizes
mainIntersectionAgent = ddpg(
    state_size=5,
    action_size=1,
    action_low=action_low,
    action_high=action_high,
    actor_lr=0.0001,
    critic_lr=0.001,
    gamma=0.99,
    tau=0.0005,
    buffer_size=10000,
    batch_size=128,
    name='MainIntersection_DDPGAgent'
)

swPedXingAgent = ddpg(
    state_size=3,
    action_size=1,
    action_low=action_low,
    action_high=action_high,
    actor_lr=0.0001,
    critic_lr=0.001,
    gamma=0.99,
    tau=0.0005,
    buffer_size=10000,
    batch_size=128,
    name='SW_PedXing_DDPGAgent'
)

sePedXingAgent = ddpg(
    state_size=3,
    action_size=1,
    action_low=action_low,
    action_high=action_high,
    actor_lr=0.0001,
    critic_lr=0.001,
    gamma=0.99,
    tau=0.0005,
    buffer_size=10000,
    batch_size=128,
    name='SE_PedXing_DDPGAgent'
)

mainIntersectionAgent.load()
swPedXingAgent.load()
sePedXingAgent.load()

# Load replay buffers if available
mainIntersectionAgent.load_replay_buffer()
swPedXingAgent.load_replay_buffer()
sePedXingAgent.load_replay_buffer()

# Load training histories if available
main_reward_history, main_actor_loss_history, main_critic_loss_history = mainIntersectionAgent.load_history('./Olivarez_traci/output_DDPG/main_ddpg_history.csv')
sw_reward_history, sw_actor_loss_history, sw_critic_loss_history = swPedXingAgent.load_history('./Olivarez_traci/output_DDPG/sw_ddpg_history.csv')
se_reward_history, se_actor_loss_history, se_critic_loss_history = sePedXingAgent.load_history('./Olivarez_traci/output_DDPG/se_ddpg_history.csv')


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
    'sumo',
    '-c', sumo_cfg_path,
    '--step-length', '0.05',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_DDPG\SD_DDPG_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_DDPG\SD_DDPG_trips.xml'
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
swCurrentPhase = 0
swCurrentPhaseDuration = 30.0
seCurrentPhase = 0
seCurrentPhaseDuration = 30.0

# Keep previous state for reward calculation
mainPrevState = None
mainPrevAction = None
swPrevState = None
swPrevAction = None
sePrevState = None
sePrevAction = None

# Training params
BATCH_SIZE = 128
TRAIN_FREQUENCY = 100  # training every 100 simulation steps
step_counter = 0

# Data storage for plotting
if not main_reward_history:
    main_reward_history = []
    main_actor_loss_history = []
    main_critic_loss_history = []

if not sw_reward_history:
    sw_reward_history = []
    sw_actor_loss_history = []
    sw_critic_loss_history = []
    
if not se_reward_history:
    se_reward_history = []
    se_actor_loss_history = []
    se_critic_loss_history = []

# Accumulate rewards between training steps
total_main_reward = 0
total_sw_reward = 0
total_se_reward = 0

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
    """Get state for main intersection"""
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

def _swPedXing_queue():
    """Get state for SW pedestrian crossing"""
    north = _weighted_waits("e2_0") + _weighted_waits("e2_1")
    south = _weighted_waits("e2_4") + _weighted_waits("e2_5")
    
    pedestrian = 0
    try:
        junction_subscription = traci.junction.getContextSubscriptionResults("6401523012")
        if junction_subscription:
            for pid, data in junction_subscription.items():
                pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    except Exception:
        pass
    
    return np.array([south, north, pedestrian], dtype=np.float32)

def _sePedXing_queue():
    """Get state for SE pedestrian crossing"""
    west = _weighted_waits("e2_2") + _weighted_waits("e2_3")
    east = _weighted_waits("e2_6") + _weighted_waits("e2_7")
    
    pedestrian = 0
    try:
        junction_subscription = traci.junction.getContextSubscriptionResults("3285696417")
        if junction_subscription:
            for pid, data in junction_subscription.items():
                pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    except Exception:
        pass
    
    return np.array([west, east, pedestrian], dtype=np.float32)

# ------- Reward calculation -------

def calculate_reward(current_state):
    """Calculate reward based on total queue (negative waiting time)"""
    if current_state is None:
        return 0.0
    
    # Normalize by dividing by 1000 to keep values in reasonable range
    normalized_total = float(np.sum(current_state)) / 1000.0
    
    # Negative reward proportional to queue (we want to minimize waiting)
    reward = -normalized_total
    
    return reward

# ------- Phase application functions -------

def _mainIntersection_phase(action):
    """Apply continuous action to main intersection phase duration"""
    global mainCurrentPhase, mainCurrentPhaseDuration

    mainCurrentPhase = (mainCurrentPhase + 1) % 10

    # Map action from [-1, 1] to duration adjustment [-15, 15]
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

def _swPedXing_phase(action):
    """Apply continuous action to SW pedestrian crossing phase duration"""
    global swCurrentPhase, swCurrentPhaseDuration

    swCurrentPhase = (swCurrentPhase + 1) % 4
    duration_adjustment = float(np.clip(action[0], -1.0, 1.0) * 5.0)

    if swCurrentPhase % 2 == 1:
        base_duration = 5.0
    else:
        base_duration = 30.0

    swCurrentPhaseDuration = float(max(5.0, min(60.0, base_duration + duration_adjustment)))

    try:
        traci.trafficlight.setPhase("6401523012", swCurrentPhase)
        traci.trafficlight.setPhaseDuration("6401523012", swCurrentPhaseDuration)
    except Exception as e:
        print(f"[WARN] setPhase failed for SW ped crossing: {e}")

def _sePedXing_phase(action):
    """Apply continuous action to SE pedestrian crossing phase duration"""
    global seCurrentPhase, seCurrentPhaseDuration

    seCurrentPhase = (seCurrentPhase + 1) % 4
    duration_adjustment = float(np.clip(action[0], -1.0, 1.0) * 5.0)

    if seCurrentPhase % 2 == 1:
        base_duration = 5.0
    else:
        base_duration = 30.0

    seCurrentPhaseDuration = float(max(5.0, min(60.0, base_duration + duration_adjustment)))

    try:
        traci.trafficlight.setPhase("3285696417", seCurrentPhase)
        traci.trafficlight.setPhaseDuration("3285696417", seCurrentPhaseDuration)
    except Exception as e:
        print(f"[WARN] setPhase failed for SE ped crossing: {e}")

# ------- History saving function -------

def save_history(filename, headers, reward_hist, actor_loss_hist, critic_loss_hist, train_frequency):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    existing_rows = 0
    
    if file_exists:
        with open(filename, 'r') as f:
            existing_rows = sum(1 for _ in f) - 1  # minus header line

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)

        # Write only new history data
        for i in range(existing_rows, len(reward_hist)):
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

print("Resetting DDPG exploration noise...")
mainIntersectionAgent.noise.reset()
swPedXingAgent.noise.reset()
sePedXingAgent.noise.reset()

print("Starting simulation loop...")

while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1

    # ---- Main intersection logic ----
    mainCurrentPhaseDuration -= stepLength
    if mainCurrentPhaseDuration <= 0:
        # Get current state (normalized)
        mainCurrentState = _mainIntersection_queue() / 1000.0  # Normalize
        mainReward = calculate_reward(mainCurrentState * 1000.0)  # Calculate on unnormalized
        total_main_reward += mainReward

        # Store experience if we have previous state
        if mainPrevState is not None and mainPrevAction is not None:
            done = False
            mainIntersectionAgent.remember(
                mainPrevState, 
                mainPrevAction, 
                mainReward, 
                mainCurrentState, 
                done
            )

        # Get action from agent (with exploration noise during training)
        mainAction = mainIntersectionAgent.get_action(mainCurrentState, add_noise=True)
        _mainIntersection_phase(mainAction)

        mainPrevState = mainCurrentState
        mainPrevAction = mainAction

        print(f"[MAIN] Queue={np.sum(mainCurrentState * 1000.0):.1f}, Reward={mainReward:.3f}, Action={mainAction[0]:.3f}")

    # ---- SW pedestrian crossing logic ----
    swCurrentPhaseDuration -= stepLength
    if swCurrentPhaseDuration <= 0:
        swCurrentState = _swPedXing_queue() / 1000.0
        swReward = calculate_reward(swCurrentState * 1000.0)
        total_sw_reward += swReward

        if swPrevState is not None and swPrevAction is not None:
            done = False
            swPedXingAgent.remember(swPrevState, swPrevAction, swReward, swCurrentState, done)

        swAction = swPedXingAgent.get_action(swCurrentState, add_noise=True)
        _swPedXing_phase(swAction)

        swPrevState = swCurrentState
        swPrevAction = swAction

        print(f"[SW] Queue={np.sum(swCurrentState * 1000.0):.1f}, Reward={swReward:.3f}, Action={swAction[0]:.3f}")

    # ---- SE pedestrian crossing logic ----
    seCurrentPhaseDuration -= stepLength
    if seCurrentPhaseDuration <= 0:
        seCurrentState = _sePedXing_queue() / 1000.0
        seReward = calculate_reward(seCurrentState * 1000.0)
        total_se_reward += seReward

        if sePrevState is not None and sePrevAction is not None:
            done = False
            sePedXingAgent.remember(sePrevState, sePrevAction, seReward, seCurrentState, done)

        seAction = sePedXingAgent.get_action(seCurrentState, add_noise=True)
        _sePedXing_phase(seAction)

        sePrevState = seCurrentState
        sePrevAction = seAction

        print(f"[SE] Queue={np.sum(seCurrentState * 1000.0):.1f}, Reward={seReward:.3f}, Action={seAction[0]:.3f}")

    # ---- Periodic training using replay buffer ----
    if step_counter % TRAIN_FREQUENCY == 0:
        # Train Main
        if len(mainIntersectionAgent.replay_buffer) >= BATCH_SIZE:
            actor_loss, critic_loss = mainIntersectionAgent.train()
            main_actor_loss_history.append(float(actor_loss))
            main_critic_loss_history.append(float(critic_loss))
            main_reward_history.append(total_main_reward)
            total_main_reward = 0
            print(f"[TRAIN MAIN] Step={step_counter}, Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")
        
        # Train SW
        if len(swPedXingAgent.replay_buffer) >= BATCH_SIZE:
            actor_loss, critic_loss = swPedXingAgent.train()
            sw_actor_loss_history.append(float(actor_loss))
            sw_critic_loss_history.append(float(critic_loss))
            sw_reward_history.append(total_sw_reward)
            total_sw_reward = 0
            print(f"[TRAIN SW] Step={step_counter}, Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")
        
        # Train SE
        if len(sePedXingAgent.replay_buffer) >= BATCH_SIZE:
            actor_loss, critic_loss = sePedXingAgent.train()
            se_actor_loss_history.append(float(actor_loss))
            se_critic_loss_history.append(float(critic_loss))
            se_reward_history.append(total_se_reward)
            total_se_reward = 0
            print(f"[TRAIN SE] Step={step_counter}, Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")

    # Step simulation forward
    traci.simulationStep()

print("\nSimulation completed!")

# Save trained models
print("Saving trained models...")
try:
    mainIntersectionAgent.save()
    swPedXingAgent.save()
    sePedXingAgent.save()
    print("Models saved successfully!")
except Exception as e:
    print(f"[ERROR] Could not save models: {e}")
    
# ------- Save Replay Buffers -------
print("Saving replay buffers...")
try:
    mainIntersectionAgent.save_replay_buffer()
    swPedXingAgent.save_replay_buffer()
    sePedXingAgent.save_replay_buffer()
    print("Replay buffers saved successfully!")
except Exception as e:
    print(f"[ERROR] Could not save replay buffers: {e}")

# Save training history
print("Saving training history...")
try:
    save_history(
        './Olivarez_traci/output_DDPG/main_ddpg_history.csv',
        ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss'],
        main_reward_history,
        main_actor_loss_history,
        main_critic_loss_history,
        TRAIN_FREQUENCY
    )
    
    save_history(
        './Olivarez_traci/output_DDPG/sw_ddpg_history.csv',
        ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss'],
        sw_reward_history,
        sw_actor_loss_history,
        sw_critic_loss_history,
        TRAIN_FREQUENCY
    )
    
    save_history(
        './Olivarez_traci/output_DDPG/se_ddpg_history.csv',
        ['Step', 'Reward', 'Actor_Loss', 'Critic_Loss'],
        se_reward_history,
        se_actor_loss_history,
        se_critic_loss_history,
        TRAIN_FREQUENCY
    )
    
    print("History saved successfully!")
except Exception as e:
    print(f"[ERROR] Could not save history: {e}")
    

traci.close()
print("Done!")