# MDL-network-population-clustering

**Inputs:** \
**edgesets:** list of sets. the s-th set contains all the edges (i,j) in the s-th network in the sample (do not include the other direction (j,i) if network is undirected). 
    the order of edgesets within D only matters for contiguous clustering, where we want the edgesets to be in order of the samples in time
**N:** number of nodes in each network \
**K0:** initial number of clusters (for discontiguous clustering, usually K0 = 1 works well; for contiguous clustering it doesn't matter) \
**n_fails:** number of failed reassign/merge/split/merge-split moves before terminating algorithm \
**bipartite:** 'None' for unipartite network populations, array [# of nodes of type 1, # of nodes of type 2] otherwise \
**directed:** Set to True when sets of edges input are directed 
**max_runs:** Maximum number of allowed moves, independent of number of failed moves

**Outputs of 'run_sims' (unconstrained description length optimization) and 'dynamic_contiguous' (restriction to contiguous clusters):** \
**C:** dictionary with items (cluster label):(set of indices corresponding to networks in cluster) \
**A:** dictionary with items (cluster label):(set of edges corresponding to mode of cluster) \
**L:** inverse compression ratio (description length after clustering)/(description length of naive transmission) 

**For discontiguous clustering, use:** \
MDLobj = MDL_populations(edgesets,N,K0,n_fails,bipartite,directed,max_runs) \
MDLobj.initialize_clusters() \
C,A,L = MDLobj.run_sims() 

**For contiguous clustering, use:** \
MDLobj = MDL_populations(edgesets,N,K0=(anything),n_fails=(anything),bipartite,directed) \
C,A,L = MDLobj.dynamic_contiguous() 

If you use this algorithm please cite:

A. Kirkley, A. Rojas, M. Rosvall, and J-G. Young, Compressing network populations with modal
networks reveals structural diversity. Communications Physics 6, 148 (2023).

