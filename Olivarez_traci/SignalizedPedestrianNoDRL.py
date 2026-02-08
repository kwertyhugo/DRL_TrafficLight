import os
import sys 
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo',
    '-c', r'Olivarez_traci\SignalizedPed.sumocfg',
    '--step-length', '0.1',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_NoDRL\SP_NoDRL_stats_trafficjam.xml',
    '--tripinfo-output', r'Olivarez_traci\output_NoDRL\SP_NoDRL_trips_trafficjam.xml'
]

#Metrics
detector_count = 7
jam_length_average = 0
jam_length_total = 0
metric_observation_count = 0

#Object Context Subscription in SUMO
detector_ids = ["e2_4", "e2_5", "e2_6", "e2_7", "e2_8", "e2_9", "e2_10"]

def _subscribe_all_detectors():
    global detector_ids 
    vehicle_vars = [traci.constants.JAM_LENGTH_METERS, traci.constants.VAR_INTERVAL_NUMBER]
    
    for det_id in detector_ids:
        traci.lanearea.subscribe(
            det_id,
            vehicle_vars
        )
        
traci.start(Sumo_config)
_subscribe_all_detectors()

while traci.simulation.getMinExpectedNumber() > 0:
    jam_length = 0
    metric_observation_count += 1
        
    for det_id in detector_ids:
        detector_stats = traci.lanearea.getSubscriptionResults(det_id)

        if not detector_stats:
            print("Lane Data Error: Undetected")
            break
            
        jam_length += detector_stats.get(traci.constants.JAM_LENGTH_METERS, 0)
                
    jam_length /= detector_count
    jam_length_total += jam_length
    traci.simulationStep()

traci.close()

jam_length_average = jam_length_total / metric_observation_count

print("\n Queue Length:", jam_length_average)