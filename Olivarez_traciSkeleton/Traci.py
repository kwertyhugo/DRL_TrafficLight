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
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)

# Step 6: Define Variables
vehicle_speed = 0
total_speed = 0

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    # # Get and print current traffic light state of J1
    # tls_state = traci.trafficlight.getRedYellowGreenState("cluster_295373794_3477931123_7465167861")
    # print(f"Current TLS state at TLS: {tls_state}")
    
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", 0)
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", 3)
    
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", 1)
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", 3)
    
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", 2)
    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", 3)
    
    tls_state = traci.trafficlight.getPhase("cluster_295373794_3477931123_7465167861")
    print(f"Current TLS state at main TLS: {tls_state}")
    
    lane1 = traci.lanearea.getLastStepVehicleNumber("e2_6")
    print(f"{lane1}")
    
    lane2 = traci.lanearea.getLastStepVehicleNumber("e2_4")
    print(f"{lane2}")

# Step 7: Close connection

traci.close()
