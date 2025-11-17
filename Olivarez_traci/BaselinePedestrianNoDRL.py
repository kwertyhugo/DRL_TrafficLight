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
    '-c', r'Olivarez_traci\baselinePed.sumocfg',
    '--step-length', '0.1',
    '--delay', '20',
    '--lateral-resolution', '0.1',
    '--statistic-output', r'Olivarez_traci\output_NoDRL\BP_NoDRL_stats.xml',
    '--tripinfo-output', r'Olivarez_traci\output_NoDRL\BP_NoDRL_trips.xml'
]

traci.start(Sumo_config)

steps = 0
while steps < 7000:    
    steps += 0.1
    traci.simulationStep()

traci.close()