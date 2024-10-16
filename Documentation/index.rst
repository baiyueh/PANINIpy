.. meta::
    :description lang=en:
        PANINIpy: Package of Algorithms for Nonparametric Inference with Networks in Python is a package designed for nonparametric inference with complex network data, 
        with methods for identifying hubs in networks, regionalizing mobility or distributional data over spatial networks, clustering network populations, and constructing hypergraphs from temporal data among other features.

.. image:: https://imgur.com/RPhHG5W.png
    :width: 250px

-------------------------------------


PANINIpy: Package of Algorithms for Nonparametric Inference with Networks in Python
=============================================

*PANINIpy*: *PANINIpy* is a pure Python written package that provides a collection of nonparametric statistical inference methods for unsupervised learning with network data. It is motivated by the Minimum Description Length (MDL) principle and aims to make network inference accessible and robust without the need for parameter tuning. PANINIpy is designed for extracting meaningful structural and dynamical patterns from complex networks in a flexible, nonparametric way, particularly useful for researchers and practitioners in the field of network analysis. The benefits of using *PANINIpy* as below:

- **Easy Installation**: Collection of nonparametric statistical inference methods can be easily installed through pip in few seconds.
- **Pure Python Implementation**: All modules are written in pure Python, ensuring compatibility with standard Python environments and minimal dependencies.
- **No Manual Parameter Tuning**: *PANINIpy*'s nonparametric approach means users do not need to manually adjust parameters, reducing potential biases and making the process more user-friendly.
- **Ease of Use**: PANINIpy's design prioritizes simplicity, allowing users to easily integrate its methods into their workflows without advanced statistical techniques/library.
- **Unified Nonparametric Methods**: *PANINIpy* provides a collection of nonparametric methods unified under the MDL principle, making it a comprehensive tool for network inference.
- **Robustness to Noise**: The methods are designed to be robust in the presence of noise, making them suitable for real-world data that may contain measurement errors or sampling biases.
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

