def partition_clustering(g,partitions,energies,num_mcs,num_init_modes,initial_runs,\
                        manual_mode_scale,dist_percentile,fix_K,consecutive_rejects,max_local_moves,\
                         Lambda,Kprior,MItype,mode_info_type,\
                         graph_plots,node_size = 5):
    
    """
    partition clustering algorithm (wrapper for Cython code)
    inputs:
        'g': igraph network object
        'graph_plots': plot output partitions using sknetwork (True/False)
        'node_size': size of nodes in these plots
         all other variables defined in Cython class and function definitions
         
    returns:
        modes and mode_labels (as in Cython function)
        g
        length S list of length N partitions
        partition log-probabilities 
    """
    
    partitions = np.array(partitions).astype('int')
    energies = np.array(energies).astype('float')
    num_mcs,num_init_modes,initial_runs,manual_mode_scale,dist_percentile,fix_K,consecutive_rejects,\
                max_local_moves,Lambda,Kprior = \
         int(num_mcs),int(num_init_modes),int(initial_runs),float(manual_mode_scale),float(dist_percentile),\
        int(fix_K),int(consecutive_rejects),int(max_local_moves),float(Lambda),int(Kprior)
    res = partition_clustering_cython(partitions,energies,num_mcs,num_init_modes,initial_runs,\
                               manual_mode_scale,dist_percentile,fix_K,consecutive_rejects,\
                                 max_local_moves,Lambda,Kprior,MItype,mode_info_type)
    
    modes = list(res[0])
    if graph_plots == True: 
        g2 = pyg.igraph2gt(g)
        pos = sfdp_layout(g2, multilevel=True, cooling_step=0.99)
        x, y = ungroup_vector_property(pos, [0, 1])
        g.vs['x'] = list(x.a)
        g.vs['y'] = list(y.a)
        print('-----------------------------------------------------------')
        print('Plotting '+str(len(modes))+' representative partitions...')
        print('----------------------------------------------------------- \n')
        weights = Counter(res[-1])
        adjacency = ss.csr_matrix(g.get_adjacency().data)
        for m in modes:
            print('Partition index:',m,', Partition log-probability:',energies[m],', Weight:',weights[m]/len(energies))
            partition_labels = np.array(gt.inference.partition_modes.align_partition_labels(partitions[m],\
                                                                partitions[modes[0]])).astype('int')
            image = sknetwork.visualization.svg_graph(adjacency,position = np.array([g.vs['x'],g.vs['y']]).T,scale=1, \
                                                      node_size=node_size,labels = partition_labels)
                
            display(SVG(image))
    
    mode_labels = list(res[2])
    return modes,mode_labels,g,partitions,energies