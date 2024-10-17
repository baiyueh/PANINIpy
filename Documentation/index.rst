.. meta::
    :description lang=en:
        PANINIpy: Package of Algorithms for Nonparametric Inference with Networks in Python is a package designed for nonparametric inference with complex network data, 
        with methods for identifying hubs in networks, regionalizing mobility or distributional data over spatial networks, clustering network populations, and constructing hypergraphs from temporal data among other features.

.. image:: https://imgur.com/RPhHG5W.png
    :width: 250px

-------------------------------------


PANINIpy: Package of Algorithms for Nonparametric Inference with Networks in Python
=============================================

*PANINIpy*: *PANINIpy* is a Python package providing a collection of nonparametric statistical inference methods for unsupervised learning with network data. Many of these methods are motivated by the Minimum Description Length (MDL) principle and aim to make network inference accessible and robust without the need for parameter tuning. Some benefits of using *PANINIpy* are listed below:

- **Easy Installation**: PANINIpy can be easily installed through pip.
- **Pure Python Implementation**: All modules are written in pure Python, ensuring compatibility with standard Python environments and minimal dependencies.
- **Parameter-Free**: *PANINIpy*'s nonparametric algorithms allow users to extract meaningful summaries of network data without the need to manually adjust free parameters. This reduces potential biases from human aesthetic preferences and implicit algorithmic constraints, and makes the network inference process more user-friendly.
- **Statistical Robustness**: The methods are designed to separate significant structural regularities from noise in network data, making them suitable for real-world datasets that may contain statistical fluctuations and measurement error.
- **Focus on Network Inference**: Unlike broader network analysis packages, *PANINIpy* specifically focuses on nonparametric inference, filling a gap for specialized statistical analysis without the overhead of broader functionalities.


.. toctree::
   :maxdepth: 2
   :caption: Installation and Contributing

   Papers/installation
   Papers/contributing


.. toctree::
   :maxdepth: 2
   :caption: Methods

   Papers/hypergraph_binning
   Papers/population_clustering
   Papers/distributional_regionalization
   Papers/hub_identification
   Papers/community_regionalization
   Papers/mdl_backboning

