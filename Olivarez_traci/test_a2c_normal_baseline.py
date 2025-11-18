import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create output directory if it doesn't exist
os.makedirs('./Olivarez_traci/output_A2C_Baseline', exist_ok=True)

import traci
import numpy as np
from keras.utils import to_categorical
from models.A2C import A2CAgent as a2c
from keras.models import load_model

# === LOAD TRAINED AGENT ===
print("\n" + "=" * 70)
print("LOADING TRAINED BASELINE A2C MODEL FOR TESTING")
print("=" * 70)

# Initialize agent (same architecture as training)
mainIntersectionAgent = a2c(
    state_size=13,
    action_size=7,
    gamma=0.99,
    learning_rate=0.0005,
    entropy_coef=0.08,
    value_coef=0.5,
    max_grad_norm=1.0,
    name='A2C_Baseline_Agent'
)

# Load trained model
try:
    model_path = './Olivarez_traci/models_A2C_Baseline/A2C_Baseline_Agent.keras'
    mainIntersectionAgent.model = load_model(model_path)
    print(f"✓ Loaded Baseline Agent from {model_path}")
    print("=" * 70)
except Exception as e:
    print(f"ERROR: Could not load trained model!")
    print(f"Details: {e}")
    print("Please ensure you have a trained model in ./Olivarez_traci/models_A2C_Baseline/")
    sys.exit(1)

# === SUMO CONFIGURATION ===
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo',
    '-c', r'Olivarez_traci\baselinePed.sumocfg',
    '--route-files', r'Olivarez_traci\demand\flows_normal_traffic.rou.xml',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_A2C_Baseline\SP_A2C_Baseline_Normal_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_A2C_Baseline\SP_A2C_Baseline_Normal_trips.xml'
]

# === SIMULATION VARIABLES ===
stepLength = 0.1
mainCurrentPhase = 0
mainCurrentPhaseDuration = 30
actionSpace = (-15, -10, -5, 0, 5, 10, 15)

# === HELPER FUNCTIONS ===
def _junctionSubscription(junction_id):
    traci.junction.subscribeContext(
        junction_id, traci.constants.CMD_GET_PERSON_VARIABLE,
        10.0, [traci.constants.VAR_WAITING_TIME]
    )
    
def _subscribe_all_detectors():
    detector_ids = ["e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10"]
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
    e2_values = [_weighted_waits(f"e2_{i}") for i in range(4, 11)]
    
    pedestrian = 0
    junction_sub = traci.junction.getContextSubscriptionResults("cluster_295373794_3477931123_7465167861")
    if junction_sub:
        for pid, data in junction_sub.items():
            pedestrian += data.get(traci.constants.VAR_WAITING_TIME, 0)
    
    return e2_values + [pedestrian]

def _mainIntersection_phase(action_index):
    global mainCurrentPhase, mainCurrentPhaseDuration
    mainCurrentPhase = (mainCurrentPhase + 1) % 10
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

# === START SUMO ===
traci.start(Sumo_config)
_subscribe_all_detectors()
_junctionSubscription("cluster_295373794_3477931123_7465167861")

print("\n" + "=" * 70)
print("TESTING BASELINE A2C AGENT ON NORMAL TRAFFIC SCENARIO")
print("=" * 70)
print("Configuration:")
print("  - Traffic Flow: flows_normal_traffic.rou.xml")
print("  - Step Length: 0.1s")
print("  - Mode: INFERENCE (training=False)")
print("  - System: Baseline (No Pedestrian Crossing Signals)")
print("  - Outputs: SP_A2C_Baseline_Normal_stats.xml, SP_A2C_Baseline_Normal_trips.xml")
print("=" * 70 + "\n")

# === TEST LOOP ===
step_count = 0
while traci.simulation.getMinExpectedNumber() > 0:
    
    mainCurrentPhaseDuration -= stepLength
    
    if mainCurrentPhaseDuration <= 0:
        
        # Get state
        main_queue = np.array(_mainIntersection_queue())
        normalized_main_queue = main_queue / 1000.0
        main_phase = to_categorical(mainCurrentPhase // 2, num_classes=5).flatten()
        mainCurrentState = np.concatenate([
            normalized_main_queue, main_phase
        ]).astype(np.float32)
        
        # Get action from trained model (inference mode)
        mainActionIndex = mainIntersectionAgent.act(mainCurrentState, training=False)
        
        # Apply action
        _mainIntersection_phase(mainActionIndex)
        
        # Print progress every 100 steps
        if step_count % 100 == 0:
            sim_time = traci.simulation.getTime()
            print(f"[Step {step_count:5d}] Time: {sim_time:7.1f}s | "
                  f"Queue Total: {sum(main_queue):7.1f}")
        
        step_count += 1
    
    traci.simulationStep()

# === CLOSE SIMULATION ===
print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
print(f"Total steps executed: {step_count}")
print(f"\nOutput files saved:")
print(f"  - Statistics: Olivarez_traci/output_A2C_Baseline/SP_A2C_Baseline_Normal_stats.xml")
print(f"  - Trip Info: Olivarez_traci/output_A2C_Baseline/SP_A2C_Baseline_Normal_trips.xml")
print("\nCompare these with your signalized A2C results!")
print("=" * 70)

traci.close()