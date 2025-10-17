import os
import sys 
import traci

from models.DQN import dqnClass as dqn

#Select DRL Agent
mainIntersectionAgent = dqn()
swPedXingAgent = dqn()
sePedXingAgent = dqn()


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo-gui',
    '-c', 'Olivarez_traciSkeleton\map.sumocfg',
    '--step-length', '0.05',
    '--delay', '100',
    '--lateral-resolution', '0.1'
]

#Simulation Variables
stepLength = 0.05
currentPhase = 0
currentPhaseDuration = 30

#Inputs to the Model
def _mainIntersection_queue():
    #vehicle detectors
    southwest = traci.lanearea.getLastStepVehicleNumber("e2_4") + traci.lanearea.getLastStepVehicleNumber("e2_5")
    southeast = traci.lanearea.getLastStepVehicleNumber("e2_6") + traci.lanearea.getLastStepVehicleNumber("e2_7")
    northeast = traci.lanearea.getLastStepVehicleNumber("e2_8")
    northwest = traci.lanearea.getLastStepVehicleNumber("e2_9") + traci.lanearea.getLastStepVehicleNumber("e2_10")
    
    #pedestrian detectors
    pedestrian = 0
    

    return (southwest, southeast, northeast, northwest, pedestrian)

def _swPedXing_queue():
    north = traci.lanearea.getLastStepVehicleNumber("e2_0") + traci.lanearea.getLastStepVehicleNumber("e2_1")
    south = traci.lanearea.getLastStepVehicleNumber("e2_4") + traci.lanearea.getLastStepVehicleNumber("e2_5")
    pedestrian = 0
    
    return (north, south, pedestrian)

def _sePedXing_queue():
    west = traci.lanearea.getLastStepVehicleNumber("e2_2") + traci.lanearea.getLastStepVehicleNumber("e2_3")
    east = traci.lanearea.getLastStepVehicleNumber("e2_6") + traci.lanearea.getLastStepVehicleNumber("e2_7")
    pedestrian = 0
    
    return (west, east, pedestrian)
    


#Output of the model
def _mainIntersection_phase():
    global currentPhase, currentPhaseDuration
    
    currentPhase += 1
    currentPhase = currentPhase%10
    print("MAIN PHASE: ", currentPhase)
        
    #Placeholder logic for now, fixed time logic
    traci.trafficlight.setPhase("cluster_295373794_3477931123_7465167861", currentPhase)
    if currentPhase == 2 or currentPhase == 4:
        currentPhaseDuration = 15
    elif currentPhase%2 == 0:
        currentPhaseDuration = 30
    else:
        currentPhaseDuration = 3

    traci.trafficlight.setPhaseDuration("cluster_295373794_3477931123_7465167861", currentPhaseDuration)
    
def _swPedXing_phase():
    global currentPhase, currentPhaseDuration
    
    currentPhase += 1
    currentPhase = currentPhase%4
    print("SW PHASE: ", currentPhase)
        
    #Placeholder logic for now, fixed time logic
    traci.trafficlight.setPhase("6401523012", currentPhase)
    if currentPhase%2 == 1:
        currentPhaseDuration = 5
    elif currentPhase%2 == 0:
        currentPhaseDuration = 30

    traci.trafficlight.setPhaseDuration("6401523012", currentPhaseDuration)
    
def _sePedXing_phase():
    global currentPhase, currentPhaseDuration
    
    currentPhase += 1
    currentPhase = currentPhase%4
    print("SE PHASE: ", currentPhase)
        
    #Placeholder logic for now, fixed time logic
    traci.trafficlight.setPhase("3285696417", currentPhase)
    if currentPhase%2 == 1:
        currentPhaseDuration = 5
    elif currentPhase%2 == 0:
        currentPhaseDuration = 30

    traci.trafficlight.setPhaseDuration("3285696417", currentPhaseDuration)



traci.start(Sumo_config)
detector_ids = traci.lanearea.getIDList()
    
#Simulation Loop
while traci.simulation.getMinExpectedNumber() > 0:
    currentPhaseDuration -= 1*stepLength
    if currentPhaseDuration <= 0:
        _mainIntersection_phase()
    
    #agent reward
    total = sum(_mainIntersection_queue, _swPedXing_queue, _sePedXing_queue)
    
    
    traci.simulationStep()


traci.close()
