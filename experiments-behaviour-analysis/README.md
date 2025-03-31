# experiments-behaviour-analysis
Experiments for generating metaheuristic search process data. Uses MAHF for configuring algorithms.


Algorithms used:
- Random Search (RS)
- Genetic Algorithm (GA)
- Particle Swarm Optimisation (PSO)

Note that for GA, the configurations are split according to the selection parameters and the results are logged
in subordinate folders. This is to load data for subsequent analysis in batches to reduce running out of memory.

## Exploratory Experiments

Generate data for all algorithms on all 24 BBOB functions (but only in limited dimensions, though this can be expanded
as necessary) to
- test general setup
- compare dimensionality reduction techniques applied to data

Logging is simplified, but includes several diversity measures. These are reduced to the most relevant one(s) for the 
experiments.
