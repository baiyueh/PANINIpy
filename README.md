<meta name="google-site-verification" content="QBsxqO0wWO-ZZl0YBi0pC_PCSEgF9Z671lLwrUavwyE" />

[![PyPI version](https://badge.fury.io/py/paninipy.svg)](https://pypi.org/project/paninipy/)
[![ReadTheDocs](https://img.shields.io/readthedocs/paninipy.svg)](
    https://paninipy.readthedocs.io/en/latest/)
[![CI](https://github.com/baiyueh/PANINIpy/actions/workflows/pages/pages-build-deployment/badge.svg?branch=gh-pages)](https://baiyueh.github.io/PANINIpy/)

# [PANINIpy](https://github.com/baiyueh/PANINIpy)

PANINIpy: Package of Algorithms for Nonparametric Inference with Networks in Python is a package designed for nonparametric inference with complex network data, with methods for identifying hubs in networks, regionalizing mobility or distributional data over spatial networks, clustering network populations, and constructing hypergraphs from temporal data among other features. 

## Table of Contents

- [Installation](#installation)
- [Modules](#modules)
  - [Binning Temporal Hypergraphs](#binning-temporal-hypergraphs)
  - [Clustering Network Populations](#clustering-network-populations)
  - [Regionalization with Distributional Data](#regionalization-with-distributional-data)
  - [Identifying Network Hubs](#identifying-network-hubs)
  - [Regionalization with Community Detection](#regionalization-with-community-detection)
  - [MDL Network Backbones](#mdl-network-backbones)
- [Documentation](#documentation)
- [License](#license)

## Installation

pip install paninipy
### [PyPI](https://pypi.org/project/paninipy/)

## Modules
### [Binning Temporal Hypergraphs](https://paninipy.readthedocs.io/en/latest/Papers/hypergraph_binning.html)

Identify MDL-optimal temporally contiguous partitions of event data between distinct node sets (e.g. users and products).\
Utilizes method derived in “Inference of dynamic hypergraph representations in temporal interaction data” (Kirkley, 2024, https://arxiv.org/abs/2308.16546).


### [Clustering Network Populations](https://paninipy.readthedocs.io/en/latest/Papers/population_clustering.html)

Generate synthetic network population datasets and perform clustering of observed network populations, multilayer network layers, or temporal networks.\
Utilizes method derived in “Compressing network populations with modal networks reveals structural diversity” (Kirkley et al., 2023, https://arxiv.org/pdf/2209.13827).

### [Regionalization with Distributional Data](https://paninipy.readthedocs.io/en/latest/Papers/distributional_regionalization.html)

Perform MDL-based regionalization on distributional (e.g. census) data over space.\
Utilizes method derived in “Spatial regionalization as optimal data compression” (Kirkley, 2022, https://arxiv.org/pdf/2111.01813).

### [Identifying Network Hubs](https://paninipy.readthedocs.io/en/latest/Papers/hub_identification.html)

Identify hub nodes in a network using different information theoretic criteria.\
Utilizes methods derived in “Identifying hubs in directed networks” (Kirkley, 2024, https://arxiv.org/pdf/2312.03347).


### [Regionalization with Community Detection](https://paninipy.readthedocs.io/en/latest/Papers/community_regionalization.html)

Perform community detection-based regionalization on network data.\
Utilizes method derived in “Urban Boundary Delineation from Commuting Data with Bayesian Stochastic Blockmodeling: Scale, Contiguity, and Hierarchy” (Morel-Balbi and Kirkley, 2024, https://arxiv.org/pdf/2405.04911).

### [MDL Network Backbones](https://paninipy.readthedocs.io/en/latest/Papers/mdl_backboning.html)

Infer global and local backbones of a network using the minimum description length principle .\
Utilizes method derived in “Fast nonparametric inference of network backbones for graph sparsification” (Kirkley, 2024, https://arxiv.org/abs/2409.06417).

## Documentation 

Detailed documentation for each module and function is available at the link below:

### [PANINIpy Documentation](https://paninipy.readthedocs.io/en/latest/)

## License 
Distributed under the MIT License. See LICENSE for more information.
