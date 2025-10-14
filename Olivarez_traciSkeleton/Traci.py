# Step 1: Add modules to provide access to specific libraries and functions
import os # Module provides functions to handle file paths, directories, environment variables
import sys # Module provides access to Python-specific system parameters and functions

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci # Static network information (such as reading and analyzing network files)

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'map.sumocfg',
    '--step-length', '0.05',
    '--delay', '100',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)

# Step 6: Define Variables
stepLength = 0.05
currentPhase = 0
currentPhaseDuration = 30+stepLength

while traci.simulation.getMinExpectedNumber() > 0:
    currentPhaseDuration -= 1*stepLength
    if currentPhaseDuration <= 0:
        lane1 = traci.lanearea.getLastStepVehicleNumber("e2_6")
        print(f"{lane1}")
    
        lane2 = traci.lanearea.getLastStepVehicleNumber("e2_4")
        print(f"{lane2}")
        
        currentPhase += 1
        currentPhase = currentPhase%10
        print("PHASE: ", currentPhase)
        
        traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", currentPhase)
        if currentPhase == 2 or currentPhase == 4:
            currentPhaseDuration = 15 + stepLength
        elif currentPhase%2 == 0:
            currentPhaseDuration = 30 + stepLength
        else:
            currentPhaseDuration = 3 + stepLength

        traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", currentPhaseDuration)
    
    traci.simulationStep()

# Step 7: Close connection

traci.close()
