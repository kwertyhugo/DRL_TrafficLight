# AI-Driven Traffic Light Optimization: Evaluating Performance for Vehicles and Pedestrians in the Philippines
This repository contains the simulation environment and Deep Reinforcement Learning (DRL) agents for optimizing traffic signal timing in Laguna. The study utilizes SUMO (Simulation of Urban Mobility) to evaluate the performance of DQN, A2C, and DDPG algorithms under "Baseline" (unsignalized) and "Signalized" pedestrian crossing scenarios.

# Running Instructions
- python -m venv venv
- venv\Scripts\activate
- pip install -r requirements.txt
- python Olivarez_traci\SignalizedPedestrianDiscrete_A2C.py

# Project Overview 
Traffic congestion in the Philippines results in significant economic loss, yet most AI-driven solutions focus on metropolitan areas like Metro Manila. [cite_start]This study addresses the gap by focusing on single intersections in regional cities (specifically Laguna).

## Key Features
- Three DRL Algorithms: Comparison of Deep Q-Network (DQN), Advantage Actor-Critic (A2C), and Deep Deterministic Policy Gradient (DDPG).
- Dual Scenarios:
  + Baseline: Standard traffic lights; pedestrians cross during gaps (unsignalized).
  + Signalized: Dedicated pedestrian phases integrated into the signal cycle.
- Real-World Data Integration: Network topology from OpenStreetMap (OSM) and traffic demand estimated via carriageway width.
- Diverse Traffic Conditions: Simulations cover Normal (26% road saturation), Slow (48%), and Traffic Jam (81%) conditions.
