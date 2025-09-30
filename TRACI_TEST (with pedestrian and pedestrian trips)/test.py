import os
import sys
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")



Sumo_config = [
    'sumo-gui',
    '-c', 'run_simulation.sumocfg',
    '--step-length', '0.1',
    '--delay', '1000',
    '--lateral-resolution', '1'
]

traci.start(Sumo_config)

# Define Variables
trafficLightID = "J1"
trafficLightPhase = -1



# # Step 7: Define Functions

# # Step 8: Take simulation steps until there are no more vehicles in the network
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    if trafficLightID in traci.trafficlight.getIDList():
        if trafficLightPhase != traci.trafficlight.getPhase(trafficLightID):
            trafficLightPhase = traci.trafficlight.getPhase(trafficLightID)
            print(trafficLightPhase, "\n")


#     # Here you can decide what to do with simulation data at each step
#     if 'J1' in traci.vehicle.getIDList():
#         vehicle_speed = traci.vehicle.getSpeed('veh1')
#         total_speed = total_speed + vehicle_speed
#     # step_count = step_count + 1
#     print(f"Vehicle speed: {vehicle_speed} m/s")

# Step 9: Close connection between SUMO and Traci
traci.close()