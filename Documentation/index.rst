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

# Statement of Need

Due to their discrete, relational, and heterogeneous nature, complex networks present new obstacles for statistical inference. Many inference objectives on networks are intrinsically combinatorial and produce complex summaries in the form of sets or partitions. These factors make scalability and interpretability of critical importance for practical algorithms, which are not often easily accommodated within learning frameworks that focus on continuous ordered data. There are also a number of ways uncertainty can be introduced in the collection of a network dataset, whether through measurement error, sampling bias, or fluctuations across experimental settings in longitudinal or cross-sectional studies. These factors underscore the importance of developing new principled and flexible methods for extracting structural and dynamical regularities in networks that do not rely on ad hoc parameter choices or heuristics, allowing them to be robust in the presence of noise. 

*PANINIpy* is a flexible and easy-to-use collection of nonparametric statistical inference methods for unsupervised learning with network data. These methods are unified in their motivation from fundamental principles–currently, the Minimum Description Length (MDL) principle underlies all the methods in *PANINIpy*—and their lack of dependence on arbitrary parameter choices that can impose unwanted biases in inference results. *PANINIpy* is highly accessible for practitioners as its methods do not require the user to manually tune any input parameters and everything is written from scratch in pure Python to be optimized for each task of interest without reliance on existing packages. *PANINIpy* therefore provides an important complement to existing network analysis packages in Python such as NetworkX that focus primarily on network metrics, network visualization, and community detection. 

There are number of existing Python packages containing individual methods that perform nonparametric inference with networks, but *PANINIpy* is unique as it is unified under this scope and performs a wide variety of network inference tasks beyond community detection or network reconstruction. The Graph-Tool package, a general-purpose network analysis package, includes a number of principled Bayesian methods for community detection and network reconstruction. Other popular packages such as NetworkX and iGraph also have methods for complex network inference—largely for the task of community detection—and are used primarily for network summary statistics and visualization. *PANINIpy* fills an important gap in the software space for network inference methods with very simple dependencies in pure Python.


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

