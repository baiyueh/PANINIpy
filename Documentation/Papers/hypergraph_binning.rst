Binning Temporal Hypergraphs
+++++++++

Tutorial 
===============

Code to perform hypergraph binning method derived in "Inference of dynamic hypergraph representations in temporal interaction data" (Kirkley, 2024, https://arxiv.org/abs/2308.16546). 

Inputs a list containing events between two disjoint node sets, with entries of the form (source node, destination node, weight, time), where:

- **source node:** Node from first set (i.e. a user).
- **destination:** Node from second set (i.e. an item purchased by the user).
- **weight:** The weight of the event, representing its significance or frequency (i.e. the cost of the item).
- **time:** The time at which the event occurs.

Also inputs a desired resolution dt for time discretzation, and whether to use exact dynamic programming or an approximate greedy method for the inference.

Outputs compression ratio, event clusters, and number of time steps, where:

- **compression ratio:** The ratio of the description length after event binning to the description length of naive transmission.
- **event clusters:** A list assigning each event to a temporal cluster label where the events can be aggregated into a weighted hypergraph or bipartite graph.
- **number of time steps:** The number of time steps corresponding to the specified width (dt).

The method minimizes the following nonparametric Minimum Description Length (MDL) clustering objective over 1D segmentations :math:`\tau`  of the time interval:

.. _equation1:

.. math::
    :nowrap:

    \[
    \mathcal{L}_{\text{total}}(\mathcal{X}, \tau) = \sum_{k=1}^{K} \mathcal{L}_{\text{cluster}}^{(k)},
    \]

where

.. _equation2:

.. math::
    :nowrap:

    \[
    \mathcal{L}_{\text{cluster}}^{(k)} = \log(N - 1)(T - 1) + \log\left(\binom{S}{m_k}\right)\left(\binom{D}{m_k}\right)\left (\binom{\tau_k}{m_k}\right) + \left[\log \Omega(s^{(k)}, d^{(k)}) + \log \Omega(G^{(k)}, n^{(k)}) \right]
    \]

is the description length of the k-th temporal cluster according to Eq. 14 in https://arxiv.org/pdf/2308.16546. 

MDL Hypergraph Binning
======================

This module provides functions to calculate the logarithm of binomial and multinomial coefficients, as well as to identify the Minimum Description Length (MDL) configuration for hypergraph binning.

Functions
---------

All of the following functions are provided in this module and have the same general usage as described below.

.. list-table:: Functions
   :header-rows: 1

   * - Function
     - Description
   * - `logchoose(n, k) <#logchoose>`_
     - Compute the logarithm of the binomial coefficient.
   * - `logmult(counts) <#logmult>`_
     - Compute the logarithm of the multinomial coefficient over count data.
   * - `logOmega(rs, cs, swap=True) <#logOmega>`_
     - Compute the logarithm of the number of non-negative integer matrices with specified row and column sums.
   * - `MDL_hypergraph_binning(X, dt, exact=True) <#MDL_hypergraph_binning>`_
     - Identify the MDL-optimal temporally contiguous partition of event data.
   * - `MDL_hypergraph_binning.DL(i, j) <#DL>`_
     - Compute the cluster-level description length.

Reference
---------

.. _logchoose:

.. raw:: html

   <div id="logchoose" class="function-header">
       <span class="class-name">function</span> <span class="function-name">logchoose(n, k)</span> 
       <a class="source-link" href="../Code/hypergraph_binning.html#logchoose">[source]</a>
   </div>

**Description**:
Compute the logarithm of the binomial coefficient.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (n, k)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">n</span>: Total number of items.</li>
       <li><span class="param-name">k</span>: Number of chosen items.</li>
   </ul>

**Returns**:
  - **float**: Logarithm of the binomial coefficient.

.. _logmult:

.. raw:: html

   <div id="logmult" class="function-header">
       <span class="class-name">function</span> <span class="function-name">logmult(counts)</span> 
       <a href="../Code/hypergraph_binning.html#logmult" class="source-link">[source]</a>
   </div>

**Description**:
Compute the logarithm of the multinomial coefficient over count data.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (counts)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">counts</span>: Count data for which the multinomial coefficient is calculated.</li>
   </ul>

**Returns**:
  - **float**: Logarithm of the multinomial coefficient.

.. _logOmega:

.. raw:: html

   <div id="logOmega" class="function-header">
       <span class="class-name">function</span> <span class="function-name">logOmega(rs, cs, swap=True)</span> 
       <a href="../Code/hypergraph_binning.html#logomega" class="source-link">[source]</a>
   </div>

**Description**:
Compute the logarithm of the number of non-negative integer matrices with specified row and column sums.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (rs, cs, swap=True)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">rs</span>: Array of row sums.</li>
       <li><span class="param-name">cs</span>: Array of column sums.</li>
       <li><span class="param-name">swap</span>: Boolean to swap the definition of rows and columns for a minor accuracy improvement.</li>
   </ul>

**Returns**:
  - **float**: Logarithm of the number of non-negative integer matrices.

.. _MDL_hypergraph_binning:

.. raw:: html

   <div id="MDL_hypergraph_binning" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_hypergraph_binning(X, dt, exact=True)</span> 
       <a href="../Code/hypergraph_binning.html#mdl-hypergraph-binning" class="source-link">[source]</a>
   </div>

**Description**:
Identify the MDL-optimal temporally contiguous partition of event data X at resolution dt.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (X, dt, exact=True)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">X</span>: List of event data entries, each in the form [source, destination, weight, time].</li>
       <li><span class="param-name">dt</span>: Time discretization width.</li>
       <li><span class="param-name">exact</span>: Boolean to indicate whether to use the exact dynamic programming solution or the faster approximate greedy solution.</li>
   </ul>

**Returns**:
  - **best_MDL/L0**: Compression ratio eta for MDL-optimal temporally contiguous partition of event data X.
  - **labels**: Partition of the event data into event clusters.
  - **T**: Number of time steps corresponding to width dt.

.. _DL:

.. raw:: html

   <div id="DL" class="function-header">
       <span class="class-name">function</span> <span class="function-name">MDL_hypergraph_binning.DL(i, j)</span> 
       <a href="../Code/hypergraph_binning.html#mdl-hypergraph-binning" class="source-link">[source]</a>
   </div>

**Description**:
Compute the description length for cluster corresponding to the time interval [i,j].

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (i, j)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">i</span>: Interval start index.</li>
       <li><span class="param-name">j</span>: Interval end index.</li>
   </ul>

**Returns**:
  - **float**: Cluster-level description length.

Demo
====

The following example demonstrates how to use the ``MDL_hypergraph_binning`` function on an event dataset ``X`` with a time step width ``dt``.

Example Code
------------

**Step 1: Import necessary libraries**

.. code-block:: python

    import time
    from paninipy.hypergraph_binning import MDL_hypergraph_binning, logOmega, logchoose
    import matplotlib.pyplot as plt

**Step 2: Generate synthetic event dataset and choose time discretization**

.. code-block:: python

    X = [('A','1',1,0.1),('B','1',1,0.2),('B','1',1,0.3),('A','1',1,0.4),('B','1',1,0.5),
         ('B','1',1,1),('C','2',1,2),('A','2',1,3),('B','2',1,4),('C','2',1,5)]
    dt = 0.1

**Step 3: Run the exact dynamic programming algorithm**

.. code-block:: python

    start_exact = time.time()
    results_exact = MDL_hypergraph_binning(X, dt, exact=True)
    runtime_exact = time.time() - start_exact

**Step 4: Run the fast greedy agglomerative algorithm**

.. code-block:: python

    start_greedy = time.time()
    results_greedy = MDL_hypergraph_binning(X, dt, exact=False)
    runtime_greedy = time.time() - start_greedy

**Step 5: Display results**

.. code-block:: python

    print('Exact Dynamic Program Results: ')
    print('     compression ratio =', round(results_exact[0], 4))
    print('     MDL-optimal event partition =', results_exact[1])
    print('     number of time steps =', results_exact[2])
    print('     runtime =', round(runtime_exact, 4))

    print('Greedy Algorithm Results: ')
    print('     compression ratio =', round(results_greedy[0], 4))
    print('     MDL-optimal event partition =', results_greedy[1])
    print('     number of time steps =', results_greedy[2])
    print('     runtime =', round(runtime_greedy, 4))


**Step 6: Function to visualize the binning results**

.. code-block:: python

    def visualize_binning(X, result, method):
        labels = result[1]
        times = [t for _, _, _, t in X]
        pairs = [(src, dest) for src, dest, _, _ in X]
        times_transformed = times
        unique_nodes = sorted(list(set([src for src, dest in pairs] + [dest for src, dest in pairs])))
        node_pos = {node: i for i, node in enumerate(unique_nodes)}

        colors = ['skyblue' if label == 0 else 'lightcoral' for label in labels]

        fig, ax = plt.subplots(figsize=(12, 4))
        for (src, dest), time, color in zip(pairs, times_transformed, colors):
            src_pos = node_pos[src]
            dest_pos = node_pos[dest]
            ax.plot([time, time], [src_pos, dest_pos], color=color, marker='o', markersize=11, linestyle='-', zorder=1)
            ax.text(time, src_pos, f"{src}", ha='center', va='center', fontsize=9, fontweight='bold', zorder=2, color='black')
            ax.text(time, dest_pos, f"{dest}", ha='center', va='center', fontsize=9, fontweight='bold', zorder=2, color='black')

        ax.set_xlim(min(times_transformed) - 0.5, max(times_transformed) + 0.5)
        ax.set_ylim(min(node_pos.values()) - 1, max(node_pos.values()) + 1)
        ax.set_yticks([])
        ax.set_xlabel('Time')
        ax.set_title(f'Event Partition in Time Scale with {method} Dynamic Solution')
        plt.savefig(f"timeline_plot_with_log_transform_{method}.png", bbox_inches='tight', dpi=200)
        plt.show()

**Step 7: Visualize the binning results for the exact dynamic programming**

.. code-block:: python

    visualize_binning(X, results_exact, 'exact dynamic programming')

**Step 8: Visualize the binning results for the fast greedy agglomerative**

.. code-block:: python

    visualize_binning(X, results_greedy, 'fast greedy agglomerative')

Example Output
--------------

.. code-block:: text

    Exact Dynamic Program Results: 
        compression ratio = 0.9842
        MDL-optimal event partition = [0 0 0 0 0 0 1 1 1 1]
        number of time steps = 50
        runtime = 0.006
    Greedy Algorithm Results: 
        compression ratio = 0.9913
        MDL-optimal event partition = [0 0 0 0 0 1 1 1 1 1]
        number of time steps = 50
        runtime = 0.0036

.. figure:: Figures/timeline_plot_with_exact_dynamic_programming.png
    :alt: Hypergraph binning results for synthetic dataset with exact dynamic programming solution.
    
Hypergraph binning results for synthetic dataset with exact dynamic programming solution. The x-axis represents time and the events are plotted with colors indicating event clusters. Events labeled with "0" are partitioned into the first group (light blue), and events labeled with "1" are partitioned into the second group (light red). Each group forms a cohesive hypergraph structure involving the two sets of nodes.

.. figure:: Figures/timeline_plot_with_fast_greedy_agglomerative.png
    :alt: Hypergraph binning results for synthetic dataset with greedy agglomerative solution.
    
Hypergraph binning results for synthetic dataset with greedy agglomerative solution. 

Paper source
====

If you use this algorithm in your work, please cite:

A. Kirkley, Inference of dynamic hypergraph representations in temporal interaction data. Physical Review E 109, 054306 (2024).
Paper: https://arxiv.org/abs/2308.16546
