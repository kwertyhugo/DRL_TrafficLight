import os
import sys
import traci
import numpy as np

# Import your DDPG implementation (ensure models package is on PYTHONPATH)
from models.DDPG import DDPGAgent as ddpg

# ------- Configuration / Agents -------

# Define continuous action bounds (vector or scalar)
action_low = np.array([-1.0], dtype=np.float32)
action_high = np.array([1.0], dtype=np.float32)

# Create agents (use the DDPGAgent signature from your provided implementation)
mainIntersectionAgent = ddpg(
    state_size=5,
    action_size=1,
    action_low=action_low,
    action_high=action_high,
    actor_lr=0.0001,
    critic_lr=0.001,
    gamma=0.99,
    tau=0.001,
    buffer_size=100000,
    batch_size=64,
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
    tau=0.001,
    buffer_size=100000,
    batch_size=64,
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
    tau=0.001,
    buffer_size=100000,
    batch_size=64,
    name='SE_PedXing_DDPGAgent'
)

# ------- SUMO start-up -------

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Build sumo command; make path handling robust
sumo_cfg_path = os.path.join('Olivarez_traciSkeleton', 'map.sumocfg')
Sumo_config = [
    'sumo-gui',
    '-c', sumo_cfg_path,
    '--step-length', '0.05',
    '--delay', '0',
    '--lateral-resolution', '0.1'
]

# ------- Helpers & subscriptions -------

def _junctionSubscription(junction_id):
    # subscription for persons near a junction; wrap in try/except in case ID is invalid
    try:
        traci.junction.subscribeContext(
            junction_id,
            traci.constants.CMD_GET_PERSON_VARIABLE,
            10.0,
            [traci.constants.VAR_SPEED]
        )
    except Exception:
        # If subscription fails (bad id / no persons), ignore silently
        pass

# ------- Simulation variables -------

stepLength = 0.05
mainCurrentPhase = 0
mainCurrentPhaseDuration = 30.0
swCurrentPhase = 0
swCurrentPhaseDuration = 30.0
seCurrentPhase = 0
seCurrentPhaseDuration = 30.0

# Note: actionSpace mapping was used earlier, but DDPG outputs continuous values.
# We'll print the continuous action and clipped duration adjustment instead.
# Keep previous state/action for reward memory
mainPrevState = None
mainPrevAction = None
swPrevState = None
swPrevAction = None
sePrevState = None
sePrevAction = None

# Training params
BATCH_SIZE = 32
TRAIN_FREQUENCY = 1000  # training every TRAIN_FREQUENCY simulation steps
step_counter = 0

# ------- Environment state functions (using your detectors) -------

def _mainIntersection_queue():
    # Ensure detector IDs exist in your network; adjust names to match your detectors.
    southwest = traci.lanearea.getLastStepVehicleNumber("e2_4") + traci.lanearea.getLastStepVehicleNumber("e2_5")
    southeast = traci.lanearea.getLastStepVehicleNumber("e2_6") + traci.lanearea.getLastStepVehicleNumber("e2_7")
    northeast = traci.lanearea.getLastStepVehicleNumber("e2_8")
    northwest = traci.lanearea.getLastStepVehicleNumber("e2_9") + traci.lanearea.getLastStepVehicleNumber("e2_10")
    pedestrian = 0
    return (southwest, southeast, northeast, northwest, pedestrian)

def _swPedXing_queue():
    north = traci.lanearea.getLastStepVehicleNumber("e2_0") + traci.lanearea.getLastStepVehicleNumber("e2_1")
    south = traci.lanearea.getLastStepVehicleNumber("e2_4") + traci.lanearea.getLastStepVehicleNumber("e2_5")
    pedestrian = 0
    junction_id = "6401523012"

    _junctionSubscription(junction_id)
    try:
        junction_subscription = traci.junction.getContextSubscriptionResults(junction_id)
    except Exception:
        junction_subscription = None

    if junction_subscription:
        for pid, data in junction_subscription.items():
            speed = data.get(traci.constants.VAR_SPEED, 0)
            # your previous logic used speed <= 0.5 -> count pedestrian
            if speed <= 0.5:
                pedestrian += 1

    return (south, north, pedestrian)

def _sePedXing_queue():
    west = traci.lanearea.getLastStepVehicleNumber("e2_2") + traci.lanearea.getLastStepVehicleNumber("e2_3")
    east = traci.lanearea.getLastStepVehicleNumber("e2_6") + traci.lanearea.getLastStepVehicleNumber("e2_7")
    pedestrian = 0
    junction_id = "3285696417"

    _junctionSubscription(junction_id)
    try:
        junction_subscription = traci.junction.getContextSubscriptionResults(junction_id)
    except Exception:
        junction_subscription = None

    if junction_subscription:
        for pid, data in junction_subscription.items():
            speed = data.get(traci.constants.VAR_SPEED, 0)
            if speed == 0:
                pedestrian += 1

    return (west, east, pedestrian)

# ------- Reward calc (simple queue reduction) -------
def calculate_reward(current_state, prev_state):
    if prev_state is None:
        return 0.0
    current_total = float(np.sum(current_state))
    prev_total = float(np.sum(prev_state))
    queue_diff = prev_total - current_total
    # normalize / scale
    if queue_diff > 0:
        reward = queue_diff * 2.0
    else:
        reward = queue_diff
    return float(reward)

# ------- Phase application functions (apply continuous action to duration) -------
def _mainIntersection_phase(action):
    global mainCurrentPhase, mainCurrentPhaseDuration

    # increment phase index
    mainCurrentPhase = (mainCurrentPhase + 1) % 10

    # actor returns e.g. np.array([x]) where x in [action_low, action_high]
    duration_adjustment = float(np.clip(action[0], action_low[0], action_high[0]) * 15.0)
    # compute base_duration by original rules
    if mainCurrentPhase == 2 or mainCurrentPhase == 4:
        base_duration = 15.0
    elif mainCurrentPhase % 2 == 0:
        base_duration = 30.0
    else:
        base_duration = 3.0

    # apply adjustment and clamp
    mainCurrentPhaseDuration = float(max(5.0, min(60.0, base_duration + duration_adjustment)))

    # set phase and duration (IDs must match your network)
    try:
        traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", mainCurrentPhase)
        traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", mainCurrentPhaseDuration)
    except Exception as e:
        # Log but don't crash if ID invalid
        print(f"[WARN] setPhase failed for main intersection: {e}")

def _swPedXing_phase(action):
    global swCurrentPhase, swCurrentPhaseDuration

    swCurrentPhase = (swCurrentPhase + 1) % 4
    duration_adjustment = float(np.clip(action[0], action_low[0], action_high[0]) * 15.0)

    # base durations
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
    global seCurrentPhase, seCurrentPhaseDuration

    seCurrentPhase = (seCurrentPhase + 1) % 4
    duration_adjustment = float(np.clip(action[0], action_low[0], action_high[0]) * 15.0)

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

# ------- Start SUMO and run simulation loop -------

traci.start(Sumo_config)
# detector_ids = traci.lanearea.getIDList()  # optional: inspect detector ids

while traci.simulation.getMinExpectedNumber() > 0:
    step_counter += 1

    # ---- Main intersection logic ----
    mainCurrentPhaseDuration -= stepLength
    if mainCurrentPhaseDuration <= 0:
        mainCurrentState = np.array(_mainIntersection_queue(), dtype=np.float32)
        mainReward = calculate_reward(mainCurrentState, mainPrevState)

        if mainPrevState is not None and mainPrevAction is not None:
            done = False
            mainIntersectionAgent.remember(mainPrevState, mainPrevAction, mainReward, mainCurrentState, done)

        mainAction = mainIntersectionAgent.get_action(mainCurrentState, add_noise=True)
        _mainIntersection_phase(mainAction)

        mainPrevState = mainCurrentState
        mainPrevAction = mainAction

        print(f"[MAIN] Queue={int(np.sum(mainCurrentState))}, Reward={mainReward:.3f}, Action={mainAction}")

    # ---- SW pedestrian crossing logic ----
    swCurrentPhaseDuration -= stepLength
    if swCurrentPhaseDuration <= 0:
        swCurrentState = np.array(_swPedXing_queue(), dtype=np.float32)
        swReward = calculate_reward(swCurrentState, swPrevState)

        if swPrevState is not None and swPrevAction is not None:
            done = False
            swPedXingAgent.remember(swPrevState, swPrevAction, swReward, swCurrentState, done)

        swAction = swPedXingAgent.get_action(swCurrentState, add_noise=True)
        _swPedXing_phase(swAction)

        swPrevState = swCurrentState
        swPrevAction = swAction

        print(f"[SW] Queue={int(np.sum(swCurrentState))}, Reward={swReward:.3f}, Action={swAction}")

    # ---- SE pedestrian crossing logic ----
    seCurrentPhaseDuration -= stepLength
    if seCurrentPhaseDuration <= 0:
        seCurrentState = np.array(_sePedXing_queue(), dtype=np.float32)
        seReward = calculate_reward(seCurrentState, sePrevState)

        if sePrevState is not None and sePrevAction is not None:
            done = False
            sePedXingAgent.remember(sePrevState, sePrevAction, seReward, seCurrentState, done)

        seAction = sePedXingAgent.get_action(seCurrentState, add_noise=True)
        _sePedXing_phase(seAction)

        sePrevState = seCurrentState
        sePrevAction = seAction

        print(f"[SE] Queue={int(np.sum(seCurrentState))}, Reward={seReward:.3f}, Action={seAction}")

    # ---- Periodic training using replay buffer and train() method ----
    if step_counter % TRAIN_FREQUENCY == 0:
        # Main
        if len(mainIntersectionAgent.replay_buffer) >= BATCH_SIZE:
            actor_loss, critic_loss = mainIntersectionAgent.train()
            print(f"[TRAIN MAIN] actor_loss={actor_loss}, critic_loss={critic_loss}, buffer={len(mainIntersectionAgent.replay_buffer)}")
        # SW
        if len(swPedXingAgent.replay_buffer) >= BATCH_SIZE:
            actor_loss, critic_loss = swPedXingAgent.train()
            print(f"[TRAIN SW] actor_loss={actor_loss}, critic_loss={critic_loss}, buffer={len(swPedXingAgent.replay_buffer)}")
        # SE
        if len(sePedXingAgent.replay_buffer) >= BATCH_SIZE:
            actor_loss, critic_loss = sePedXingAgent.train()
            print(f"[TRAIN SE] actor_loss={actor_loss}, critic_loss={critic_loss}, buffer={len(sePedXingAgent.replay_buffer)}")

    # Step simulation forward
    traci.simulationStep()

# Save trained models (DDPGAgent.save exists in your agent)
print("Saving trained models...")
try:
    mainIntersectionAgent.save(folder='models_saved')
    swPedXingAgent.save(folder='models_saved')
    sePedXingAgent.save(folder='models_saved')
    print("Models saved successfully!")
except Exception as e:
    print(f"[WARN] could not save models: {e}")

traci.close()
