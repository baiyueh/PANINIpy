Regionalization with Community Detection
+++++++++

Tutorial 
===============

Code to perform spatial regionalization based on the greedy agglomerative algorithm derived in "Urban Boundary Delineation from Commuting Data with Bayesian Stochastic Blockmodeling: Scale, Contiguity, and Hierarchy" (Morel-Balbi and Kirkley, 2024, https://arxiv.org/pdf/2405.04911). 

Inputs are:

- **N:** The number of nodes in the network.
- **spatial_elist:** A list of tuples (i, j) encoding the spatial adjacencies of the fundamental spatial units, where a tuple (i, j) indicates that unit i and unit j are spatially adjacent.
- **flow_elist:** A list of tuples (i, j, w) encoding the flows between the fundamental spatial units, where a tuple (i, j, w) indicates a flow of w going from unit i to unit j.

Outputs a regionalization result DLs, partitions, where:

- **DLs:** A list of description length values (in nats) at each iteration of the algorithm.
- **partitions:** A list of partitions at each iteration of the algorithm, where each partition is described by a list containing the node identifiers of the nodes belonging to the partition.

Algorithm minimizes the following clustering objective over contiguous partitions of the input spatial network:

.. math::

    C(B) + \sum_r g(r) + \sum_{r,s} f(r, s)

as specified in Eq. 9 of https://arxiv.org/pdf/2405.04911. The function C(B) only depends on the number of clusters B and other network-level constants, while g(r) only depends on the r-th cluster and f(r,s) only depends on the r-th and s-th clusters. This general functional form can accommodate most stochastic block model variants as well as modularity or infomap. 

The current code implementation provided here uses an objective function corresponding to the description length of a weighted stochastic block model, as described in the paper, but this can be easily modified for other objective functions as described above.


Input parameters:
^^^^^^^^^^^^^^^^^

`greedy_opt` takes in the following parameters:

**N**
    The number of nodes in the network.

**spatial_elist**
    A list of tuples :math:`(i, j)` encoding the spatial adjacencies of the fundamental spatial units, where a tuple :math:`(i, j)` indicates that unit :math:`i` and unit :math:`j` are spatially adjacent.

**flow_elist**
    A list of tuples :math:`(i, j, w)` encoding the flows between the fundamental spatial units, where a tuple :math:`(i, j, w)` indicates a flow of :math:`w` going from unit :math:`i` to unit :math:`j`.

The file `Baton Rouge, LA.pkl` contains a pickled dictionary with the input parameters for the example case of Baton Rouge in Louisiana, as shown in `example.ipynb`.

Outputs
^^^^^^^

Once the input parameters are available, the algorithm can be run by calling the function `greedy_opt(N, spatial_elist, flow_elist)`. The outputs of the function are:

**DLs**
    A list of description length values (in nats) at each iteration of the algorithm.

**partitions**
    A list of partitions at each iteration of the algorithm, where each partition is described by a list containing the node identifiers of the nodes belonging to the partition.

Notes
-----

The algorithm is set up by default to stop and return when no merge further decreases the description length, but this behavior can be overridden by commenting out/modifying the appropriate sections of the code.


Greedy Regionalization Algorithm
================================

This module contains the code for the greedy regionalization algorithm.

All of the following functions are provided in this module and have the same general usage as described below.

.. list-table:: Functions
   :header-rows: 1

   * - Function
     - Description
   * - `DefaultDict.__init__(default_factory, **kwargs) <#init>`_
     - Initialize the DefaultDict class.
   * - `DefaultDict.__getitem__(key) <#getitem>`_
     - Get an item from the DefaultDict.
   * - `logchoose(N, K) <#logchoose>`_
     - Compute the logarithm of the binomial coefficient.
   * - `logmultiset(N, K) <#logmultiset>`_
     - Compute the logarithm of the multiset coefficient.
   * - `greedy_opt(N, spatial_elist, flow_elist) <#greedy_opt>`_
     - Perform fast greedy regionalization for objective functions.
   * - `greedy_opt.C(B) <#C>`_
     - Compute the global contribution to the description length.
   * - `greedy_opt.g(r) <#g>`_
     - Compute the cluster-level contribution to the description length.
   * - `greedy_opt.f(r, s) <#f>`_
     - Compute the cluster-to-cluster contribution to the description length.
   * - `greedy_opt.total_dl() <#total_dl>`_
     - Compute the total description length.
   * - `greedy_opt.delta_dl(r, s) <#delta_dl>`_
     - Compute the change in description length after merging clusters.
   * - `greedy_opt.merge_updates(r, s, DL) <#merge_updates>`_
     - Merge clusters and update the description length.

Reference
---------

.. _init:

.. raw:: html

   <div id="init" class="function-header">
       <span class="class-name">class</span> <span class="function-name">DefaultDict.__init__(default_factory, **kwargs)</span> <a href="#init" class="source-link">[source]</a>
   </div>

**Description**:
Initialize the DefaultDict class.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (default_factory, **kwargs)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">default_factory</span>: The default factory function.</li>
       <li><span class="param-name">**kwargs</span>: Additional keyword arguments.</li>
   </ul>

.. _getitem:

.. raw:: html

   <div id="getitem" class="function-header">
       <span class="class-name">class</span> <span class="function-name">DefaultDict.__getitem__(key)</span> <a href="#getitem" class="source-link">[source]</a>
   </div>

**Description**:
Get an item from the DefaultDict.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (key)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">key</span>: The key to get the item for.</li>
   </ul>

**Returns**:
  - **value**: The value associated with the key.

.. _logchoose:

.. raw:: html

   <div id="logchoose" class="function-header">
       <span class="class-name">function</span> <span class="function-name">logchoose(N, K)</span> <a href="#logchoose" class="source-link">[source]</a>
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

.. _logmultiset:

.. raw:: html

   <div id="logmultiset" class="function-header">
       <span class="class-name">function</span> <span class="function-name">logmultiset(N, K)</span> <a href="#logmultiset" class="source-link">[source]</a>
   </div>

**Description**:
Compute the logarithm of the multiset coefficient.

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
  - **float**: Logarithm of the multiset coefficient.

.. _greedy_opt:

.. raw:: html

   <div id="greedy_opt" class="function-header">
       <span class="class-name">function</span> <span class="function-name">greedy_opt(N, spatial_elist, flow_elist)</span> <a href="#greedy_opt" class="source-link">[source]</a>
   </div>

**Description**:
Perform fast greedy agglomerative regionalization.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (N, spatial_elist, flow_elist)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">N</span>: Number of nodes.</li>
       <li><span class="param-name">spatial_elist</span>: List of edges (i,j) defined by the spatial adjacency between i and j (no repeats).</li>
       <li><span class="param-name">flow_elist</span>: List of weighted edges defined by flows (i,j,w), where flow is from i --> j and has weight w (no repeats).</li>
   </ul>

**Returns**:
  - **DLs**: List of description length values at each iteration.
  - **partitions**: List of partitions at each iteration.

**Notes**:
Make sure nodes are indexed as 0,....,N-1 so as to handle nodes with no flows.

.. _C:

.. raw:: html

   <div id="C" class="function-header">
       <span class="class-name">function</span> <span class="function-name">greedy_opt.C(B)</span> <a href="#C" class="source-link">[source]</a>
   </div>

**Description**:
Compute the global contribution to the description length.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (B)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">B</span>: Number of clusters.</li>
   </ul>

**Returns**:
  - **float**: Global contribution to the description length.

.. _g:

.. raw:: html

   <div id="g" class="function-header">
       <span class="class-name">function</span> <span class="function-name">greedy_opt.g(r)</span> <a href="#g" class="source-link">[source]</a>
   </div>

**Description**:
Compute the cluster-level contribution to the description length.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (r)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">r</span>: Cluster index or tuple of cluster indices.</li>
   </ul>

**Returns**:
  - **float**: Cluster-level contribution to the description length.

.. _f:

.. raw:: html

   <div id="f" class="function-header">
       <span class="class-name">function</span> <span class="function-name">greedy_opt.f(r, s)</span> <a href="#f" class="source-link">[source]</a>
   </div>

**Description**:
Compute the cluster-to-cluster contribution to the description length.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (r, s)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">r</span>: Cluster index or tuple of cluster indices.</li>
       <li><span class="param-name">s</span>: Cluster index or tuple of cluster indices.</li>
   </ul>

**Returns**:
  - **float**: Cluster-to-cluster contribution to the description length.

.. _total_dl:

.. raw:: html

   <div id="total_dl" class="function-header">
       <span class="class-name">function</span> <span class="function-name">greedy_opt.total_dl()</span> <a href="#total_dl" class="source-link">[source]</a>
   </div>

**Description**:
Compute the total description length.

**Returns**:
  - **float**: Total description length.

.. _delta_dl:

.. raw:: html

   <div id="delta_dl" class="function-header">
       <span class="class-name">function</span> <span class="function-name">greedy_opt.delta_dl(r, s)</span> <a href="#delta_dl" class="source-link">[source]</a>
   </div>

**Description**:
Compute the change in description length after merging clusters.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (r, s)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">r</span>: Cluster index.</li>
       <li><span class="param-name">s</span>: Cluster index.</li>
   </ul>

**Returns**:
  - **float**: Total change in description length after merging clusters.

.. _merge_updates:

.. raw:: html

   <div id="merge_updates" class="function-header">
       <span class="class-name">function</span> <span class="function-name">greedy_opt.merge_updates(r, s, DL)</span> <a href="#merge_updates" class="source-link">[source]</a>
   </div>

**Description**:
Merge clusters and update the description length.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (r, s, DL)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">r</span>: Cluster index.</li>
       <li><span class="param-name">s</span>: Cluster index.</li>
       <li><span class="param-name">DL</span>: Current description length.</li>
   </ul>

**Returns**:
  - **float**: Updated description length.



Demo 
=======
Example Code
------------

Step 1: Import necessary libraries and configure LaTeX settings

.. code:: python

    import pickle
    import pandas as pd
    import geopandas as gpd
    import matplotlib
    import matplotlib.pyplot as plt
    from ScholarCodeCollective.community_regionalization import greedy_opt

    # LaTeX preamble
    matplotlib.rcParams.update({'text.usetex': True})
    matplotlib.rcParams.update({'text.latex.preamble': r"",
                                "font.serif": "Times"})
    matplotlib.rcParams.update({'font.family':'serif'})

Step 2: Load pickled data for the Baton Rouge CBSA

.. code:: python

    # Load pickled data for the Baton Rouge CBSA. Data is stored in a dictionary with the following keys:
    #   'Name' - Name of the MSA
    #   'Class' - Statistical classification of the MSA (one of "CBSA", "CSA", or "state")
    #   'N' - Number of nodes in the MSA
    #   'Spatial Edgelist' - List of tuples representing the spatial edges in the MSA
    #   'Flow Edgelist' - List of tuples representing the flow edges in the MSA
    fName = "Baton Rouge, LA.pkl"
    with open(fName, "rb") as f:
        data = pickle.load(f)

Step 3: Load the geospatial data for plotting

.. code:: python

    # Load the geospatial data (needed for plotting)
    fName = "geo_gdf.pkl"
    with open(fName, "rb") as f:
        geo_gdf = pickle.load(f)

Step 4: Create a dictionary mapping indices to 'GEOID10' values

.. code:: python

    # Create a dictionary mapping indices to 'GEOID10' values (needed for plotting)
    index_to_geoid = {i: geoid for i, geoid in enumerate(geo_gdf["GEOID10"])}
    # Unpack the data
    name = data["Name"]
    N = data["N"]
    spatial_elist = data["Spatial Edgelist"]
    flow_elist = data["Flow Edgelist"]

Step 5: Run the greedy algorithm and map node indices back to GEOIDs

.. code:: python

    # Run the greedy algorithm
    print(f"Running greedy algorithm on {name}...")
    DLs, partitions = greedy_opt(N, spatial_elist, flow_elist)
    # Map node indices back to GEOIDs
    clusters = []
    for partition in partitions[-1]:
        fips_set = set()
        for node_idx in partition:
            fips_set.add(index_to_geoid[node_idx])
        clusters.append(fips_set)

Step 6: Map group labels to tracts

.. code:: python

    # Map group labels to tracts
    geo_gdf["group_label"] = pd.Series()
    for i, cluster in enumerate(clusters):
        geo_gdf.loc[geo_gdf["GEOID10"].isin(cluster), "group_label"] = i

Step 7: Plot the results

.. code:: python

    # Plot: Colors indicate the inferred partitions, lines demarcate the underlying tract subdivisions
    B = len(clusters)
    fig, ax = plt.subplots(figsize=(12, 12))
    geo_gdf.plot(column="group_label", ax=ax, edgecolor="black", linewidth=0.5, cmap="tab20")
    ax.set_title(f"Description Length (nats) = {DLs[-1]:.2f} \n B = {B}", fontsize=20)
    plt.suptitle(name, fontsize=24)
    plt.axis(False)
    plt.show()

Example Output
--------------

.. figure:: output.png
    :alt: Example output showing the spatial regionalization results for Baton Rouge, LA.

Spatial regionalization Rrsults for Baton Rouge, LA. Colors indicate the inferred clusters, while the black lines show the underlying tract subdivisions. The optimal description length in nats and the optimal number of clusters B are shown above the figure.

Paper source
====

If you use this algorithm in your work, please cite:

S. Morel-Balbi and A. Kirkley*, Urban Boundary Delineation from Commuting Data with Bayesian Stochastic Blockmodeling: Scale, Contiguity, and Hierarchy. Preprint arXiv:2405.04911 (2024). 
Paper: https://arxiv.org/abs/2405.04911 
