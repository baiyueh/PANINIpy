Identifying Network Hubs
+++++++++

Tutorial 
===============
Code to perform hub node classification in networks derived in ‘Identifying hubs in directed networks’. Inputs an event dataset of the form \[(data, N, degree_list, out_degrees, weighted)\], where:

- **data:** List of tuples representing the edgelist or a list of degrees for a network.
- **N:** Number of nodes in the network.
- **degree_list:** Boolean indicating if the data is a list of degrees.
- **out_degrees:** Boolean indicating if hubs should be computed using out-degree values.
- **weighted:** Boolean indicating if multigraph encoding should be used.

Outputs a classification result of the form \[(hub nodes, their degrees, description length, compression ratio)\], where:

- **hub nodes:** List of identified hub nodes.
- **degrees:** Degrees of the identified hub nodes.
- **description length:** The corresponding description length for the identified hub nodes.
- **compression ratio:** The compression ratio for the identified hub nodes.

using the following clustering objectives:

Simple ER encoding: 

.. _equation1:

.. math::
    :nowrap:

    \[
    \mathcal{L}^{(\text{ERs})}(V_h) = \log NM + \log \binom{N}{h} + \log \binom{h(N - 1)}{M_h} + \log \binom{(N - h)(N - 1)}{M - M_h} \tag{1}
    \]

.. _equation2:

.. math::
    :nowrap:

    \[
    \mathcal{L}^{(\text{CMs})}(V_h) = \log NM + \log \binom{N}{h} + \log \binom{M_h + h - 1}{h - 1} + \sum_{i \in V_h} \log \binom{N - 1}{k_i} + \log \binom{(N - h)(N - 1)}{M - M_h} \tag{2}
    \]

Multigraph ER encoding:

.. _equation3:

.. math::
    :nowrap:

    \[
    \mathcal{L}^{(\text{ERm})}(V_h) = \log NM + \log \binom{N}{h} + \log \left(\frac{hN}{M_h}\right) + \log \left(\frac{(N - h)N}{M - M_h}\right) \tag{3}
    \]

.. _equation4:

.. math::
    :nowrap:

    \[
    \mathcal{L}^{(\text{CMm})}(V_h) = \log NM + \log \binom{N}{h} + \log \binom{M_h + h - 1}{h - 1} + \sum_{i \in V_h} \log \binom{N}{k_i} + \log \left(\frac{(N - h)N}{M - M_h}\right) \tag{4}
    \]

This method optimizes the Minimum Description Length (MDL) objective for identifying hub nodes in networks.

# Network-hubs
MDL algorithm for classifying hub nodes in networks. 

The "hubs" function in functions.py inputs either an edgelist or a degree list for a network and returns the hub nodes, their degrees, and the corresponding description length/compression ratio for the methods described in the paper below.

If you use this algorithm in your work please cite:

A. Kirkley, Identifying hubs in directed networks. Physical Review E 109, 034310 (2024).


Network Hubs
============

This module provides a class and functions for identifying network hubs using various encoding methods.

All of the following functions are provided in this module and have the same general usage as described below.

.. list-table:: Functions
   :header-rows: 1

   * - Function
     - Description
   * - `Network_hubs.__init__(name) <#init>`_
     - Initialize the Network_hubs class.
   * - `Network_hubs.logchoose(N, K) <#logchoose>`_
     - Compute the logarithm of the binomial coefficient.
   * - `Network_hubs.logmultiset(N, K) <#logmultiset>`_
     - Compute the logarithm of the multiset coefficient.
   * - `Network_hubs.hubs(data, N, degree_list=False, out_degrees=False, weighted=False) <#hubs>`_
     - Identify hub nodes in the network.

Reference
---------

.. _init:

.. raw:: html

   <div id="init" class="function-header">
       <span class="class-name">class</span> <span class="function-name">Network_hubs.__init__(name)</span> <a href="#__init__" class="source-link">[source]</a>
   </div>

**Description**:
Initialize the Network_hubs class.

.. _logchoose:

.. raw:: html

   <div id="logchoose" class="function-header">
       <span class="class-name">function</span> <span class="function-name">Network_hubs.logchoose(N, K)</span> <a href="#logchoose" class="source-link">[source]</a>
   </div>

**Description**:
Compute the logarithm of the binomial coefficient.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (N, K)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">N</span>: Total number of elements.</li>
       <li><span class="param-name">K</span>: Number of elements to choose.</li>
   </ul>

**Returns**:
  - **float**: Logarithm of the binomial coefficient.

.. _logmultiset:

.. raw:: html

   <div id="logmultiset" class="function-header">
       <span class="class-name">function</span> <span class="function-name">Network_hubs.logmultiset(N, K)</span> <a href="#logmultiset" class="source-link">[source]</a>
   </div>

**Description**:
Compute the logarithm of the multiset coefficient.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (N, K)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">N</span>: Total number of elements.</li>
       <li><span class="param-name">K</span>: Number of elements to choose.</li>
   </ul>

**Returns**:
  - **float**: Logarithm of the multiset coefficient.

.. _hubs:

.. raw:: html

   <div id="hubs" class="function-header">
       <span class="class-name">function</span> <span class="function-name">Network_hubs.hubs(data, N, degree_list=False, out_degrees=False, weighted=False)</span> <a href="#hubs" class="source-link">[source]</a>
   </div>

**Description**:
Identify hub nodes in the network.

**Parameters**:

.. raw:: html

   <div class="parameter-block">
       (data, N, degree_list=False, out_degrees=False, weighted=False)
   </div>

   <ul class="parameter-list">
       <li><span class="param-name">data</span>: List of tuples or list of degrees.</li>
       <li><span class="param-name">N</span>: Number of nodes in the network.</li>
       <li><span class="param-name">degree_list</span>: Boolean indicating if data is a list of degrees.</li>
       <li><span class="param-name">out_degrees</span>: Boolean indicating if hubs should be computed using out-degree values.</li>
       <li><span class="param-name">weighted</span>: Boolean indicating if multigraph encoding should be used.</li>
   </ul>

**Returns**:
  - **dict**: Dictionary of results for the ER and CM encodings.


Demo 
=======
Example Code
------------

**Step 1: Import necessary libraries**

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import ScholarCodeCollective as SCC
    from ScholarCodeCollective.hub_identification import Network_hubs
    import networkx as nx
    import random

**Step 2: Function to convert GML position format**

.. code-block:: python

    def convert_gml_pos(gml_file, output_file):
        with open(f'{gml_file}.gml', 'r') as file:
            lines = file.readlines()

        with open(f'{output_file}.gml', 'w') as file:
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith('_pos "'):
                    coords = stripped_line.split('"')[1]
                    x, y = coords.split(',')
                    file.write(f'      x {x.strip()}\n')
                    file.write(f'      y {y.strip()}\n')
                else:
                    file.write(line)

    gml_file = 'kidnappings'
    output_file = f'{gml_file}_converted'
    convert_gml_pos(gml_file, output_file)

**Step 3: Function to visualize network with hubs**

.. code-block:: python

    def visualize_network_with_hubs(net, N, pos, hub_results, filename, color):
        fig, ax = plt.subplots(figsize=(14, 16))  
        G = nx.DiGraph() 
        G.add_nodes_from(range(N))
        G.add_edges_from([(e[0], e[1]) for e in net])
        
        largest_cc = max(nx.weakly_connected_components(G), key=len)   # Extract the giant component
        G_sub = G.subgraph(largest_cc).copy()
        in_degrees = dict(G_sub.in_degree())  
        max_in_degree = max(in_degrees.values()) if in_degrees else 1
        node_sizes = [100 + 1000 * in_degrees[node] / max_in_degree for node in G_sub.nodes()]    
        nx.draw(G_sub, pos, ax=ax, with_labels=False, node_size=node_sizes, node_color='lightblue', font_size=3, font_weight='bold', alpha=0.4, arrows=True)
        nx.draw_networkx_labels(G_sub, pos, labels=in_degrees, font_color='black', font_size=6, font_weight='bold')
        
        hub_nodes = [node for node in hub_results['hub_nodes'] if node in G_sub]
        hub_node_size = [100 + 1000 * in_degrees[node] / max_in_degree for node in hub_nodes]
        nx.draw_networkx_nodes(G_sub, pos, ax=ax, nodelist=hub_nodes, node_color=color, node_size=hub_node_size, edgecolors='black', linewidths=2, alpha=1)
        
        dl = hub_results['description_length']
        cr = hub_results['compression_ratio']
        ax.set_title(f'{filename}, DL: {dl:.2f}, CR: {cr:.2f}', fontsize=10)
        ax.axis('off') 
        plt.tight_layout()
        plt.savefig(f'{filename}.png', bbox_inches='tight', dpi=200)
        plt.show()

**Step 4: Read the GML file and prepare data**

.. code-block:: python

    G = nx.read_gml(f'{output_file}.gml', label='id')
    pos_x = nx.get_node_attributes(G,'x')
    pos_y = nx.get_node_attributes(G,'y')
    pos = {node: (pos_x[node], pos_y[node]) for node in G.nodes()}
    print(G.nodes(data=True))
    print("\nEdges with attributes:")
    print(G.edges(data=True))
    net = set((u, v, 1) for u, v in G.edges())
    num_node = len(G.nodes(data=True))
    nh = Network_hubs("Example Network")

**Step 5: Run the network hubs algorithm for ER encoding**

.. code-block:: python

    hub_results_er = nh.hubs(net, N=num_node, degree_list=False, weighted=False)

**Step 6: Run the network hubs algorithm for CM encoding**

.. code-block:: python

    hub_results_cm = nh.hubs(net, N=num_node, degree_list=False, weighted=False)
    pos = {i: (pos_x[i], pos_y[i]) for i in range(num_node)}

**Step 7: Visualize and save the ER hubs network**

.. code-block:: python

    visualize_network_with_hubs(net, N=num_node, pos=pos, hub_results=hub_results_er['ER'], filename=f'{gml_file}_network_er_hubs', color='red')

**Step 8: Visualize and save the CM hubs network**

.. code-block:: python

    visualize_network_with_hubs(net, N=num_node, pos=pos, hub_results=hub_results_cm['CM'], filename=f'{gml_file}_network_cm_hubs', color='green')

Example Output
--------------

.. figure:: kidnappings_Network_ERs_hubs.png
    :alt: Example output showing the kidnappings network hubs identifying through ER encoding.

    Graph Representation of the Kidnappings Network (Giant Component Only) with Simple Version of ER (ERs) Encoding Hubs. Nodes are colored light blue, with hub nodes in red. The nodes are labelled with in-degree values, the edges are directed with arrows. The size of each node corresponds to its degree. The title includes the description length (DL) and compression ratio (CR).

.. figure:: kidnappings_Network_CMs_hubs.png
    :alt: Example output showing the kidnappings network hubs identifying through CM encoding.

    Graph Representation of the Kidnappings Network (Giant Component Only) with Simple Version of CM (CMs) Encoding Hubs. Nodes are colored light blue, with hub nodes in green. The nodes are labelled with in-degree values, the edges are directed with arrows. The size of each node corresponds to its degree. The title includes the description length (DL) and compression ratio (CR).

Paper source
====

If you use this algorithm in your work, please cite:

A. Kirkley, Identifying hubs in directed networks. Physical Review E [Editor’s Suggestion] 109, 034310 (2024). 
Paper: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.109.034310/