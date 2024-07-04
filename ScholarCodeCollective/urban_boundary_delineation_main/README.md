# Urban Boundary Delineation from Commuting Data with Bayesian Stochastic Blockmodeling: Scale, Contiguity, and Hierarchy

This repository contains the code implementing the greedy agglomerative algorithm used to perform spatial regionalization described in the paper "Urban Boundary Delineation from Commuting Data with Bayesian Stochastic Blockmodeling: Scale, Contiguity, and Hierarchy" by Sebastian Morel-Balbi and Alec Kirkley.

The files are:

**greedy_algorithm.py**: Python module containing the code for implementing the greedy agglomerative algorithm.

**example.ipynb**: Jupyter notebook containing an example use case of the greedy algorithm when used to perform spatial regionalization of the Baton Rouge, LA CBSA.

**Baton Rouge, LA.pkl**: Pickled dictionary containing the spatial and flow edge lists for the Baton Rouge, LA CBSA. See example.ipynb for more information.

**geo_gdf.pkl**: Pickled GeoPandas geodataframe containing the geometries for the Baton Rouge, LA CBSA. Used for plotting in example.ipynb.

**README.md**: This file.

## Typical usage

The fast greedy agglomerative algorithm can be used to optimize any objective function of the form

$$
    C(B) + \sum_r g(r) + \sum_{r,s} f(r, s).
$$

The code implementation provided here uses an objective function corresponding to the description length of a weighted stochastic block model, as described in the paper "Urban Boundary Delineation from Commuting Data with Bayesian Stochastic Blockmodeling: Scale, Contiguity, and Hierarchy", but can be modified at will.

The algorithm is implemented by the function `greedy_opt` in the `greedy_algorithm.py` module.

### Input parameters:

`greedy_opt` takes in the following parameters:

**N**: The number of nodes in the network.

**spatial_elist**: A list of tuples $(i, j)$ encoding the spatial adjacencies of the fundamental spatial units, where a tuple $(i, j)$ indicates that unit $i$ and unit $j$ are spatially adjacent.

**flow_elist**: A list of tuples $(i, j,w)$ encoding the flows between the fundamental spatial units, where a tuple $(i, j, w)$ indicates a flow of $w$ going from unit $i$ to unit $j$.

The file `Baton Rouge, LA.pkl` contains a pickled dictionary with the input parameters for the example case of Baton Rouge in Louisiana, as shown in `example.ipynb`.

### Outputs

Once the input parameters are available, the algorithm can be run by calling the function `greedy_opt(N, spatial_elist, flow_elist)`. The outputs of the function are

**DLs**: A list of description length values (in nats) at each iteration of the algorithm.

**partitions**: A list of partitions at each iteration of the algorithm, where each partition is described by a list containing the node identifiers of the nodes belonging to the partition.

## Notes

The algorithm is set up by default to stop and return when no merge further decreases the description length, but this behaviour can be overridden by commenting out/modifying the appropriate sections of the code.
