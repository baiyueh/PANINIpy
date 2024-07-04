import time
import sys
import ScholarCodeCollective
from ScholarCodeCollective.hypergraph_binning_main import Hypergraph_binning
from ScholarCodeCollective.MDL_regionalization_main import MDL_regionalization
from ScholarCodeCollective.Network_hubs_main import Network_hubs
from ScholarCodeCollective.MDL_network_population_clustering_main import MDL_populations
from ScholarCodeCollective.urban_boundary_delineation_main import greedy_opt




hypergraph_instance = ScholarCodeCollective.hypergraph_binning_main.Hypergraph_binning()

# Example call to a method that uses logchoose
hypergraph_instance.logOmega([5], [2])
hypergraph_instance.logchoose(4,3)

#example event dataset X and time step width dt
X = [('a1','b2',1,1.1),('a1','b3',1,1.5),('a1','b2',1,1.6),('a2','b2',1,1.7),('a2','b3',1,1.9),\
    ('a4','b5',1,5.5),('a1','b3',1,150),('a1','b3',1,160),('a4','b6',1,170),('a2','b3',1,190)]
dt = 1

start_exact = time.time()
results_exact = hypergraph_instance.MDL_hypergraph_binning(X,dt,exact=True)

runtime_exact = time.time() - start_exact
runtime_exact

import numpy as np
import matplotlib.pyplot as plt
import time
import ScholarCodeCollective as SCC
from ScholarCodeCollective.MDL_network_population_clustering_main import generate_synthetic, MDL_populations
import networkx as nx
import random

mode_example = [{(1,2), (1,3), (1,4), (2,1), (2,3), (2,4), (3,1), (3,2), (3,4), (3,5), (4,1), (4,2), (4,3), (5,3), (6,8), (7,8), (8,6), (8,7)}, 
        {(1,2), (1,3), (2,1), (3,1), (3,4), (3,5), (3,6), (4,3), (4,5), (4,6), (5,3), (5,4), (5,6), (5,7), (6,3), (6,4), (6,5)}, 
        {(2,4), (3,4), (4,2), (4,3), (4,6), (5,6), (5,7), (5,8), (6,4), (6,5), (6,7), (6,8), (7,5), (7,6), (7,8), (8,5), (8,6), (8,7)}]
nets, cluster_labels = generate_synthetic(S=10, N=20, modes=mode_example,
                                          alphas=[0.9, 0.8, 0.7], betas=[0.1, 0.1, 0.1], pis=[0.33, 0.33, 0.34])

mdl_pop = MDL_populations(edgesets=nets, N=20, K0=1, n_fails=100, directed=False, max_runs=100)
mdl_pop.initialize_clusters()
clusters, modes, L = mdl_pop.run_sims()
