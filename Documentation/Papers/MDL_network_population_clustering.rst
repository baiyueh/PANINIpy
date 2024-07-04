MDL network population clustering
+++++++++

Tutorial 
===============
.. _equation1:

.. math::

    \mathcal{L}_k\left(\mathcal{A}^{(k)}, C_k\right) = \mathcal{L}\left(\mathcal{A}^{(k)}\right) + S \log\left(\frac{S}{S_k}\right) + \ell_k \tag{1}

.. _equation2:
.. math::

    \mathcal{L}(\mathcal{D}) = \sum_{k=1}^{K} \mathcal{L}_k\left(\mathcal{A}^{(k)}, C_k\right). \tag{2}

Equation :eq:`(2)` gives the total description length of the data :math:`\mathcal{D}` under our multi-part transmission scheme. By minimizing this objective function we identify the best configurations of modes :math:`\mathcal{A}` and clusters :math:`\mathcal{C}`. A good configuration :math:`\{\mathcal{A}, \mathcal{C}\}` will allow us to transmit a large portion of the information in :math:`\mathcal{D}` through the modes alone. If we use too many modes, the description length will increase as these are costly to communicate in full. And if we use too few, the description length will also increase because we will have to send lengthy messages describing how mismatched networks and modes differ. Hence, through the principle of parsimony, Eq. :eq:`(2)` favors descriptions with the number of clusters :math:`K` as small as possible but not smaller.


**Inputs:**

- **edgesets:** list of sets. The s-th set contains all the edges (i, j) in the s-th network in the sample (do not include the other direction (j, i) if the network is undirected). The order of edgesets within D only matters for contiguous clustering, where we want the edgesets to be in order of the samples in time.
- **N:** number of nodes in each network
- **K0:** initial number of clusters (for discontiguous clustering, usually K0 = 1 works well; for contiguous clustering it does not matter)
- **n_fails:** number of failed reassign/merge/split/merge-split moves before terminating the algorithm
- **bipartite:** 'None' for unipartite network populations, array [# of nodes of type 1, # of nodes of type 2] otherwise
- **directed:** Set to True when sets of edges input are directed
- **max_runs:** Maximum number of allowed moves, independent of number of failed moves

**Outputs of 'run_sims' (unconstrained description length optimization) and 'dynamic_contiguous' (restriction to contiguous clusters):**

- **C:** dictionary with items (cluster label):(set of indices corresponding to networks in cluster)
- **A:** dictionary with items (cluster label):(set of edges corresponding to mode of cluster)
- **L:** inverse compression ratio (description length after clustering)/(description length of naive transmission)

**For discontiguous clustering, use:**

.. code-block:: python

    MDLobj = MDL_populations(edgesets, N, K0, n_fails, bipartite, directed, max_runs)
    MDLobj.initialize_clusters()
    C, A, L = MDLobj.run_sims()

**For contiguous clustering, use:**

.. code-block:: python

    MDLobj = MDL_populations(edgesets, N, K0=(anything), n_fails=(anything), bipartite, directed)
    C, A, L = MDLobj.dynamic_contiguous()

If you use this algorithm, please cite:

A. Kirkley, A. Rojas, M. Rosvall, and J-G. Young, Compressing network populations with modal networks reveals structural diversity. Communications Physics 6, 148 (2023).


MDL Population Clustering
==========================

This module contains the code for the MDL (Minimum Description Length) population clustering algorithm.

Functions
---------

All of the following functions are provided in this module and have the same general usage as described below.

.. list-table:: Functions
   :header-rows: 1

   * - Function
     - Description
   * - `generate_synthetic(S, N, modes, alphas, betas, pis) <#generate_synthetic>`_
     - Generate synthetic networks from the heterogeneous population model.
   * - `generate_synthetic.ind2ij(ind, N) <#ind2ij>`_
     - Convert index to edge indices.
   * - `remap_keys(Dict) <#remap_keys>`_
     - Remap dict keys to first K integers.
   * - `MDL_populations.__init__(edgesets, N, K0=1, n_fails=100, bipartite=None, directed=False, max_runs=np.inf) <#MDL_populations_init>`_
     - Initialize the MDL_populations class.
   * - `MDL_populations.initialize_clusters() <#MDL_populations_initialize_clusters>`_
     - Initialize K0 random clusters and find their modes as well as the total description length of this configuration.
   * - `MDL_populations.random_key() <#MDL_populations_random_key>`_
     - Generate random key for new cluster.
   * - `MDL_populations.logchoose(N, K) <#MDL_populations_logchoose>`_
     - Compute the logarithm of the binomial coefficient.
   * - `MDL_populations.logmult(Ns) <#MDL_populations_logmult>`_
     - Compute the logarithm of the multinomial coefficient.
   * - `MDL_populations.generate_Ek(cluster) <#MDL_populations_generate_Ek>`_
     - Tally edge counts for networks in the cluster.
   * - `MDL_populations.update_mode(Ek, Sk) <#MDL_populations_update_mode>`_
     - Generate mode from cluster edge counts by greedily removing least common edges in the cluster.
   * - `MDL_populations.Lk(Ak, Ek, Sk) <#MDL_populations_Lk>`_
     - Compute cluster description length as a function of mode, edge counts, and size of the cluster.
   * - `MDL_populations.move1(k=None) <#MDL_populations_move1>`_
     - Reassign randomly chosen network to the best cluster.
   * - `MDL_populations.move2() <#MDL_populations_move2>`_
     - Merge two randomly chosen clusters.
   * - `MDL_populations.move3() <#MDL_populations_move3>`_
     - Split randomly chosen cluster in two and perform K-means type algorithm to get these clusters and modes.
   * - `MDL_populations.move4() <#MDL_populations_move4>`_
     - Merge two randomly chosen clusters then split them.
   * - `MDL_populations.run_sims() <#MDL_populations_run_sims>`_
     - Run discontiguous (unconstrained) merge split simulations to identify modes and clusters that minimize the description length.
   * - `MDL_populations.dynamic_contiguous() <#MDL_populations_dynamic_contiguous>`_
     - Minimize description length while constraining clusters to be contiguous in time.
   * - `MDL_populations.evaluate_partition(partition, contiguous=False) <#MDL_populations_evaluate_partition>`_
     - Evaluate description length of partition.

Reference
---------

.. _generate_synthetic:

.. raw:: html

   <div id="generate_synthetic" class="function-header">
       <span class="class-name">function</span> <span class="function-name">generate_synthetic(S, N, modes, alphas, betas, pis)</span> <a href="#generate_synthetic" class="source-link">[source]</a>
   </div>

**Description**:
Generate synthetic networks from the heterogeneous population model.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (S, N, modes, alphas, betas, pis)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">S</span>: Number of synthetic networks to generate.</li>
       <li><span class="param-name">N</span>: Number of nodes in each network.</li>
       <li><span class="param-name">modes</span>: List of modes for the population model.</li>
       <li><span class="param-name">alphas</span>: List of probabilities for true positive edges in each mode.</li>
       <li><span class="param-name">betas</span>: List of probabilities for false positive edges in each mode.</li>
       <li><span class="param-name">pis</span>: List of probabilities for each mode.</li>
   </ul>

**Returns**:
  - **nets**: List of generated networks.
  - **cluster_labels**: List of cluster labels for the generated networks.

.. _ind2ij:

.. raw:: html

   <div id="ind2ij" class="function-header">
       <span class="class-name">function</span> <span class="function-name">generate_synthetic.ind2ij(ind, N)</span> <a href="#ind2ij" class="source-link">[source]</a>
   </div>

**Description**:
Convert index to edge indices.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (ind, N)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">ind</span>: Index of the edge.</li>
       <li><span class="param-name">N</span>: Number of nodes in the network.</li>
   </ul>

**Returns**:
  - **tuple**: Edge indices (i, j).

.. _remap_keys:

.. raw:: html

   <div id="remap_keys" class="function-header">
       <span class="class-name">function</span> <span class="function-name">remap_keys(Dict)</span> <a href="#remap_keys" class="source-link">[source]</a>
   </div>

**Description**:
Remap dict keys to first K integers.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (Dict)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">Dict</span>: Dictionary to remap.</li>
   </ul>

**Returns**:
  - **Dict**: Remapped dictionary.

.. _MDL_populations_init:

.. raw:: html

   <div id="MDL_populations_init" class="function-header">
       <span class="class-name">class</span> <span class="function-name">MDL_populations.__init__(edgesets, N, K0=1, n_fails=100, bipartite=None, directed=False, max_runs=np.inf)</span> <a href="#MDL_populations_init" class="source-link">[source]</a>
   </div>

**Description**:
Initialize the MDL_populations class.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (edgesets, N, K0=1, n_fails=100, bipartite=None, directed=False, max_runs=np.inf)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">edgesets</span>: List of sets. The s-th set contains all the edges (i, j) in the s-th network in the sample (do not include the other direction (j, i) if the network is undirected).</li>
       <li><span class="param-name">N</span>: Number of nodes in each network.</li>
       <li><span class="param-name">K0</span>: Initial number of clusters (for discontiguous clustering, usually K0 = 1 works well; for contiguous clustering, it does not matter).</li>
       <li><span class="param-name">n_fails</span>: Number of failed reassign/merge/split/merge-split moves before terminating the algorithm.</li>
       <li><span class="param-name">bipartite</span>: 'None' for unipartite network populations, array [# of nodes of type 1, # of nodes of type 2] otherwise.</li>
       <li><span class="param-name">directed</span>: Boolean indicating whether edgesets contain directed edges.</li>
       <li><span class="param-name">max_runs</span>: Maximum number of allowed moves, regardless of the number of fails.</li>
   </ul>

.. _MDL_populations_initialize_clusters:

.. raw:: html

   <div id="MDL_populations_initialize_clusters" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.initialize_clusters()</span> <a href="#MDL_populations_initialize_clusters" class="source-link">[source]</a>
   </div>

**Description**:
Initialize K0 random clusters and find their modes as well as the total description length of this configuration.

.. _MDL_populations_random_key:

.. raw:: html

   <div id="MDL_populations_random_key" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.random_key()</span> <a href="#MDL_populations_random_key" class="source-link">[source]</a>
   </div>

**Description**:
Generate random key for new cluster.

.. _MDL_populations_logchoose:

.. raw:: html

   <div id="MDL_populations_logchoose" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.logchoose(N, K)</span> <a href="#MDL_populations_logchoose" class="source-link">[source]</a>
   </div>

**Description**:
Compute the logarithm of the binomial coefficient.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (N, K)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">N</span>: Total number of items.</li>
       <li><span class="param-name">K</span>: Number of chosen items.</li>
   </ul>

**Returns**:
  - **float**: Logarithm of the binomial coefficient.

.. _MDL_populations_logmult:

.. raw:: html

   <div id="MDL_populations_logmult" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.logmult(Ns)</span> <a href="#MDL_populations_logmult" class="source-link">[source]</a>
   </div>

**Description**:
Compute the logarithm of the multinomial coefficient with the denominator Ns[0]!Ns[1]!....

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (Ns)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">Ns</span>: List of counts for the multinomial coefficient.</li>
   </ul>

**Returns**:
  - **float**: Logarithm of the multinomial coefficient.

.. _MDL_populations_generate_Ek:

.. raw:: html

   <div id="MDL_populations_generate_Ek" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.generate_Ek(cluster)</span> <a href="#MDL_populations_generate_Ek" class="source-link">[source]</a>
   </div>

**Description**:
Tally edge counts for networks in the cluster.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (cluster)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">cluster</span>: Set of network indices in the cluster.</li>
   </ul>

**Returns**:
  - **Ek**: Dictionary of edge counts for the cluster.

.. _MDL_populations_update_mode:

.. raw:: html

   <div id="MDL_populations_update_mode" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.update_mode(Ek, Sk)</span> <a href="#MDL_populations_update_mode" class="source-link">[source]</a>
   </div>

**Description**:
Generate mode from cluster edge counts by greedily removing least common edges in the cluster.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (Ek, Sk)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">Ek</span>: Dictionary of edge counts for the cluster.</li>
       <li><span class="param-name">Sk</span>: Size of the cluster.</li>
   </ul>

**Returns**:
  - **Ak**: Set of edges corresponding to the mode of the cluster.

.. _MDL_populations_Lk:

.. raw:: html

   <div id="MDL_populations_Lk" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.Lk(Ak, Ek, Sk)</span> <a href="#MDL_populations_Lk" class="source-link">[source]</a>
   </div>

**Description**:
Compute cluster description length as a function of mode, edge counts, and size of the cluster.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (Ak, Ek, Sk)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">Ak</span>: Set of edges corresponding to the mode of the cluster.</li>
       <li><span class="param-name">Ek</span>: Dictionary of edge counts for the cluster.</li>
       <li><span class="param-name">Sk</span>: Size of the cluster.</li>
   </ul>

**Returns**:
  - **float**: Cluster description length.

.. _MDL_populations_move1:

.. raw:: html

   <div id="MDL_populations_move1" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.move1(k=None)</span> <a href="#MDL_populations_move1" class="source-link">[source]</a>
   </div>

**Description**:
Reassign randomly chosen network to the best cluster.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (k=None)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">k</span>: Cluster index (optional).</li>
   </ul>

**Returns**:
  - **bool**: Whether the move was accepted.
  - **float**: Change in description length.

.. _MDL_populations_move2:

.. raw:: html

   <div id="MDL_populations_move2" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.move2()</span> <a href="#MDL_populations_move2" class="source-link">[source]</a>
   </div>

**Description**:
Merge two randomly chosen clusters.

**Returns**:
  - **bool**: Whether the move was accepted.
  - **float**: Change in description length.

.. _MDL_populations_move3:

.. raw:: html

   <div id="MDL_populations_move3" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.move3()</span> <a href="#MDL_populations_move3" class="source-link">[source]</a>
   </div>

**Description**:
Split randomly chosen cluster in two and perform K-means type algorithm to get these clusters and modes.

**Returns**:
  - **bool**: Whether the move was accepted.
  - **float**: Change in description length.

.. _MDL_populations_move4:

.. raw:: html

   <div id="MDL_populations_move4" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.move4()</span> <a href="#MDL_populations_move4" class="source-link">[source]</a>
   </div>

**Description**:
Merge two randomly chosen clusters then split them.

**Returns**:
  - **bool**: Whether the move was accepted.
  - **float**: Change in description length.

.. _MDL_populations_run_sims:

.. raw:: html

   <div id="MDL_populations_run_sims" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.run_sims()</span> <a href="#MDL_populations_run_sims" class="source-link">[source]</a>
   </div>

**Description**:
Run discontiguous (unconstrained) merge split simulations to identify modes and clusters that minimize the description length.

**Returns**:
  - **C**: Dictionary with items (cluster label):(set of indices corresponding to networks in the cluster).
  - **A**: Dictionary with items (cluster label):(set of edges corresponding to the mode of the cluster).
  - **L**: Inverse compression ratio (description length after clustering)/(description length of naive transmission).

.. _MDL_populations_dynamic_contiguous:

.. raw:: html

   <div id="MDL_populations_dynamic_contiguous" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.dynamic_contiguous()</span> <a href="#MDL_populations_dynamic_contiguous" class="source-link">[source]</a>
   </div>

**Description**:
Minimize description length while constraining clusters to be contiguous in time.

**Returns**:
  - **C**: Dictionary with items (cluster label):(set of indices corresponding to networks in the cluster).
  - **A**: Dictionary with items (cluster label):(set of edges corresponding to the mode of the cluster).
  - **L**: Inverse compression ratio (description length after clustering)/(description length of naive transmission).

.. _MDL_populations_evaluate_partition:

.. raw:: html

   <div id="MDL_populations_evaluate_partition" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_populations.evaluate_partition(partition, contiguous=False)</span> <a href="#MDL_populations_evaluate_partition" class="source-link">[source]</a>
   </div>

**Description**:
Evaluate description length of partition. Contiguous option removes cluster label entropy term from description length.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (partition, contiguous=False)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">partition</span>: List of cluster labels for each network.</li>
       <li><span class="param-name">contiguous</span>: Boolean indicating whether to remove cluster label entropy term.</li>
   </ul>

**Returns**:
  - **float**: Description length of the partition.



Demo 
=======
Example Code
------------
.. code:: python

  import numpy as np
  import matplotlib.pyplot as plt
  import time
  import ScholarCodeCollective as SCC
  from ScholarCodeCollective.MDL_network_population_clustering_main import generate_synthetic, MDL_populations
  import networkx as nx
  import random

  def visualize_synthetic_clusters(nets, cluster_labels, N):
      num_plots = len(nets)
      cols = 3
      rows = (num_plots // cols) + (num_plots % cols > 0)
      fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

      for i, (net, cluster_label) in enumerate(zip(nets, cluster_labels)):
          row, col = divmod(i, cols)
          ax = axes[row, col] if rows > 1 else axes[col]
          G = nx.DiGraph()
          G.add_nodes_from(range(N))
          G.add_edges_from(net)
          
          pos = nx.circular_layout(G)  # Layout for visualization
          nx.draw(G, pos, with_labels=True, ax=ax, node_size=300, node_color='skyblue', font_weight='bold', arrows=True)
          ax.set_title(f'Network {i+1} (Mode {cluster_label})')
      
      for j in range(i + 1, rows * cols):
          fig.delaxes(axes.flatten()[j])

      plt.tight_layout()
      plt.show()

  mode_example = [{(1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (3,5), (6,8), (7,8)}, 
          {(1,2), (1,3), (3,4), (3,5), (3,6), (4,5), (4,6), (5,6), (5,7)}, 
          {(2,4), (3,4), (4,6), (5,6), (5,7), (5,8), (6,7), (6,8), (7,8)}]

  node_num = 8
  nets, cluster_labels = generate_synthetic(
      S=9, 
      N=node_num, 
      modes=mode_example,
      alphas=[1, 1, 1], 
      betas=[0.1, 0.1, 0.1], 
      pis=[0.33, 0.33, 0.34]
  )

  # Visualize the synthetic networks
  visualize_synthetic_clusters(nets, cluster_labels, N=8)

  mdl_pop = MDL_populations(edgesets=nets, N=node_num, K0=1, n_fails=100, directed=False, max_runs=100)
  mdl_pop.initialize_clusters()
  clusters, modes, L = mdl_pop.run_sims()

  def visualize_clusters(nets, clusters, modes, L, N, filename='MDL_population_clusters.png'):
      num_clusters = len(clusters)
      fig, ax = plt.subplots(1, num_clusters, figsize=(15, 8))  

      if num_clusters == 1:
          ax = [ax]

      for i, (k, v) in enumerate(clusters.items()):
          G = nx.Graph()
          G.add_nodes_from(range(N))
          for idx in v:
              G.add_edges_from(nets[idx])
          
          degrees = dict(G.degree())
          max_degree = max(degrees.values()) if degrees else 1
          node_sizes = [100 + 400 * degrees[node] / max_degree for node in G.nodes()]
          
          pos = nx.circular_layout(G)  
          
          edge_colors = []
          for node in G.nodes():
              node_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
              for edge in G.edges(node):
                  edge_colors.append(node_color)
          
          nx.draw(G, pos, ax=ax[i], with_labels=True, node_size=node_sizes, node_color='lightblue', font_size=8, font_weight='bold', edge_color=edge_colors)
          
          num_networks = len(v)
          mode_size = len(modes[k])
          ax[i].set_title(f'Cluster {k}: {num_networks} networks, Mode size: {mode_size}, Inverse compression ratio: {L:.3f}', fontsize=10)
          ax[i].axis('off') 

      plt.tight_layout()
      plt.subplots_adjust(top=0.85, wspace=0.5)  
      plt.savefig(filename, bbox_inches='tight', dpi=200)
      plt.show()

  # Visualize the unconstrained description length optimization networks
  visualize_clusters(nets, clusters, modes, L, node_num)


Example Output
--------------
.. image:: synthetic_networks_population_example.png
    :alt: Example output showing the synthetic networks structure.
Example output showing the synthetic networks structure.

.. image:: MDL_population_clusters_example.png
    :alt: Example output showing the MDL population clustering results for synthetic networks.
Example output showing the MDL population clustering results for synthetic networks.

Link 
====

Paper source
------------

https://arxiv.org/abs/2209.13827