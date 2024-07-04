import time
import sys
import ScholarCodeCollective
from ScholarCodeCollective.hypergraph_binning_main import Hypergraph_binning
from ScholarCodeCollective.MDL_regionalization_main import MDL_regionalization
from ScholarCodeCollective.Network_hubs_main import Network_hubs
from ScholarCodeCollective.MDL_network_population_clustering_main import MDL_populations



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
