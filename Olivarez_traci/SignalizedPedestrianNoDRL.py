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
    '-c', r'Olivarez_traci\signalizedPed.sumocfg',
    '--step-length', '0.05',
    '--delay', '0',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_NoDRL\SP_NoDRL_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_NoDRL\SP_NoDRL_trips.xml'
]

traci.start(Sumo_config)

while traci.simulation.getMinExpectedNumber() > 0:    
    traci.simulationStep()

traci.close()