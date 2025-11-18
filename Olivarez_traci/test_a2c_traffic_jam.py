import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create output directory if it doesn't exist
os.makedirs('./Olivarez_traci/output_A2C', exist_ok=True)

import traci
import numpy as np
from keras.utils import to_categorical
from models.A2C import A2CAgent as a2c
from keras.models import load_model

# === LOAD TRAINED AGENTS ===
print("\n" + "=" * 70)
print("LOADING TRAINED A2C MODELS FOR TESTING")
print("=" * 70)

# Initialize agents (same architecture as training)
mainIntersectionAgent = a2c(
    state_size=26, action_size=7,
    gamma=0.99,
    learning_rate=0.0005,
    entropy_coef=0.08,
    value_coef=0.5,
    max_grad_norm=1.0,
    name='A2C_Main_Agent'
)

swPedXingAgent = a2c(
    state_size=23, action_size=7,
    gamma=0.99,
    learning_rate=0.0005,
    entropy_coef=0.04,
    value_coef=0.5,
    max_grad_norm=1.0,
    name='A2C_SW_PedXing_Agent'
)

sePedXingAgent = a2c(
    state_size=23, action_size=7,
    gamma=0.99,
    learning_rate=0.0005,
    entropy_coef=0.04,
    value_coef=0.5,
    max_grad_norm=1.0,
    name='A2C_SE_PedXing_Agent'
)

# Load trained models
try:
    main_model_path = './Olivarez_traci/models_A2C/A2C_Main_Agent.keras'
    sw_model_path = './Olivarez_traci/models_A2C/A2C_SW_PedXing_Agent.keras'
    se_model_path = './Olivarez_traci/models_A2C/A2C_SE_PedXing_Agent.keras'
    
    mainIntersectionAgent.model = load_model(main_model_path)
    print(f"✓ Loaded Main Agent from {main_model_path}")
    
    swPedXingAgent.model = load_model(sw_model_path)
    print(f"✓ Loaded SW Agent from {sw_model_path}")
    
    sePedXingAgent.model = load_model(se_model_path)
    print(f"✓ Loaded SE Agent from {se_model_path}")
    
    print("=" * 70)
    
except Exception as e:
    print(f"ERROR: Could not load trained models!")
    print(f"Details: {e}")
    print("Please ensure you have trained models in ./Olivarez_traci/models_A2C/")
    sys.exit(1)

# === SUMO CONFIGURATION ===
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo',
    '-c', r'Olivarez_traci\signalizedPed.sumocfg',
    '--route-files', r'Olivarez_traci\demand\flows_traffic_jam.rou.xml',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_A2C\SP_A2C_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_A2C\SP_A2C_trips.xml'
]
# === SIMULATION VARIABLES ===
stepLength = 0.1
mainCurrentPhase = 0
mainCurrentPhaseDuration = 30
swCurrentPhase = 0
swCurrentPhaseDuration = 30
seCurrentPhase = 0
seCurrentPhaseDuration = 30
actionSpace = (-15, -10, -5, 0, 5, 10, 15)

# === HELPER FUNCTIONS (same as training) ===
def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id, traci.constants.CMD_GET_PERSON_VARIABLE,
        10.0, [traci.constants.VAR_WAITING_TIME]
    )
    
def _subscribe_all_detectors():
    detector_ids = [
        "e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10",
        "e2_0", "e2_1", "e2_2", "e2_3"
    ]
    vehicle_vars = [traci.constants.VAR_TYPE, traci.constants.VAR_WAITING_TIME]
    for det_id in detector_ids:
        traci.lanearea.subscribeContext(
            det_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 3, vehicle_vars
        )

def _weighted_waits(detector_id):
    sumWait = 0
    vehicle_data = traci.lanearea.getContextSubscriptionResults(detector_id)
    if not vehicle_data:
        return 0
    
    weight_map = {"car": 1.0, "jeep": 1.5, "bus": 2.2, 
                  "truck": 2.5, "motorcycle": 0.3, "tricycle": 0.5}
    
    for data in vehicle_data.values():
        vtype = data.get(traci.constants.VAR_TYPE, "car")
        wait = data.get(traci.constants.VAR_WAITING_TIME, 0)
        sumWait += wait * weight_map.get(vtype, 1.0)
    return sumWait

def _mainIntersection_queue():
    e2_4 = _weighted_waits("e2_4")
    e2_5 = _weighted_waits("e2_5")
    e2_6 = _weighted_waits("e2_6")
    e2_7 = _weighted_waits("e2_7")
    e2_8 = _weighted_waits("e2_8")
    e2_9 = _weighted_waits("e2_9")
    e2_10 = _weighted_waits("e2_10")
    
    pedestrian = 0
    junction_sub = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
    if junction_sub:
        for pid, data in junction_sub.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return [e2_4, e2_5, e2_6, e2_7, e2_8, e2_9, e2_10, pedestrian]

def _swPedXing_queue():
    e2_0 = _weighted_waits("e2_0")
    e2_1 = _weighted_waits("e2_1")
    e2_4 = _weighted_waits("e2_4")
    e2_5 = _weighted_waits("e2_5")
    
    pedestrian = 0
    junction_sub = traci.junction.getContextSubscriptionResults("6401523012")
    if junction_sub:
        for pid, data in junction_sub.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return [e2_0, e2_1, e2_4, e2_5, pedestrian]

def _sePedXing_queue():
    e2_2 = _weighted_waits("e2_2")
    e2_3 = _weighted_waits("e2_3")
    e2_6 = _weighted_waits("e2_6")
    e2_7 = _weighted_waits("e2_7")
    
    pedestrian = 0
    junction_sub = traci.junction.getContextSubscriptionResults("3285696417")
    if junction_sub:
        for pid, data in junction_sub.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    return [e2_2, e2_3, e2_6, e2_7, pedestrian]

def _mainIntersection_phase(action_index):
    global mainCurrentPhase, mainCurrentPhaseDuration
    mainCurrentPhase += 1
    mainCurrentPhase = mainCurrentPhase % 10
    duration_adjustment = actionSpace[action_index]
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", mainCurrentPhase)
    
    if mainCurrentPhase == 2 or mainCurrentPhase == 4:
        base_duration = 15
    elif mainCurrentPhase % 2 == 0:
        base_duration = 30
    else:
        base_duration = 3
    
    mainCurrentPhaseDuration = max(5, min(60, base_duration + duration_adjustment))
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", mainCurrentPhaseDuration)
    
def _swPedXing_phase(action_index):
    global swCurrentPhase, swCurrentPhaseDuration
    swCurrentPhase += 1
    swCurrentPhase = swCurrentPhase % 4
    duration_adjustment = actionSpace[action_index]
    traci.trafficlight.setPhase("6401523012", swCurrentPhase)
    
    if swCurrentPhase % 2 == 1:
        base_duration = 5
    elif swCurrentPhase % 2 == 0:
        base_duration = 30
    
    swCurrentPhaseDuration = max(5, min(60, base_duration + duration_adjustment))
    traci.trafficlight.setPhaseDuration("6401523012", swCurrentPhaseDuration)
    
def _sePedXing_phase(action_index):
    global seCurrentPhase, seCurrentPhaseDuration
    seCurrentPhase += 1
    seCurrentPhase = seCurrentPhase % 4
    duration_adjustment = actionSpace[action_index]
    traci.trafficlight.setPhase("3285696417", seCurrentPhase)
    
    if seCurrentPhase % 2 == 1:
        base_duration = 5
    elif seCurrentPhase % 2 == 0:
        base_duration = 30
    
    seCurrentPhaseDuration = max(5, min(60, base_duration + duration_adjustment))
    traci.trafficlight.setPhaseDuration("3285696417", seCurrentPhaseDuration)

# === START SUMO ===
traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("cluster_295373794_3477931123_7465167861")
_junctionSubscription("6401523012")
_junctionSubscription("3285696417")

print("\n" + "=" * 70)
print("TESTING A2C AGENTS ON TRAFFIC JAM SCENARIO")
print("=" * 70)
print("Configuration:")
print("  - Traffic Flow: flows_traffic_jam.rou.xml")
print("  - Step Length: 0.1s")
print("  - Mode: INFERENCE (training=False)")
print("  - Outputs: SP_A2C_stats.xml, SP_A2C_trips.xml")
print("=" * 70 + "\n")

# === TEST LOOP ===
step_count = 0
while traci.simulation.getMinExpectedNumber() > 0:
    
    mainCurrentPhaseDuration -= stepLength
    swCurrentPhaseDuration -= stepLength
    seCurrentPhaseDuration -= stepLength
    
    # Agent decision points (when phase expires)
    if mainCurrentPhaseDuration <= 0:
        
        # Get current phase encodings
        main_phase = to_categorical(mainCurrentPhase, num_classes=10).flatten()
        swPed_phase = to_categorical(swCurrentPhase, num_classes=4).flatten()
        sePed_phase = to_categorical(seCurrentPhase, num_classes=4).flatten()
        
        # MAIN AGENT STATE
        main_queue = np.array(_mainIntersection_queue())
        normalized_main_queue = main_queue / 1000.0
        mainCurrentState = np.concatenate([
            normalized_main_queue, main_phase, swPed_phase, sePed_phase
        ]).astype(np.float32)

        # SW AGENT STATE
        swPed_queue = np.array(_swPedXing_queue())
        normalized_swPed_queue = swPed_queue / 1000.0
        swCurrentState = np.concatenate([
            normalized_swPed_queue, main_phase, swPed_phase, sePed_phase
        ]).astype(np.float32)
        
        # SE AGENT STATE
        sePed_queue = np.array(_sePedXing_queue())
        normalized_sePed_queue = sePed_queue / 1000.0
        seCurrentState = np.concatenate([
            normalized_sePed_queue, main_phase, swPed_phase, sePed_phase
        ]).astype(np.float32)
        
        # GET ACTIONS FROM TRAINED MODELS (inference mode)
        mainActionIndex = mainIntersectionAgent.act(mainCurrentState, training=False)
        swActionIndex = swPedXingAgent.act(swCurrentState, training=False)
        seActionIndex = sePedXingAgent.act(seCurrentState, training=False)
        
        # APPLY ACTIONS
        _mainIntersection_phase(mainActionIndex)
        _swPedXing_phase(swActionIndex)
        _sePedXing_phase(seActionIndex)
        
        # Optional: Print progress every 100 steps
        if step_count % 100 == 0:
            sim_time = traci.simulation.getTime()
            print(f"[Step {step_count:5d}] Time: {sim_time:7.1f}s | "
                  f"Main Queue: {sum(main_queue):7.1f} | "
                  f"SW Queue: {sum(swPed_queue):6.1f} | "
                  f"SE Queue: {sum(sePed_queue):6.1f}")
        
        step_count += 1
    
    traci.simulationStep()

# === CLOSE SIMULATION ===
print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
print(f"Total steps executed: {step_count}")
print(f"\nOutput files saved:")
print(f"  - Statistics: Olivarez_traci/output_A2C/SP_A2C_stats.xml")
print(f"  - Trip Info: Olivarez_traci/output_A2C/SP_A2C_trips.xml")
print("\nCompare these with your baseline (SP_NoDRL_*.xml) to evaluate performance!")
print("=" * 70)

traci.close()