Installation Steps
+++++++++

PANINIpy is available on PyPI and can be installed using ``pip``, which will include the dependencies automatically:

.. code-block:: console

    pip install paninipy

Or, install the latest version along with necessary build dependencies from the GitHub repository:

.. code-block:: console

    pip install git+https://github.com/baiyueh/PANINIpy.git

For users who prefer installing dependencies manually, we provide a requirements.txt file on GitHub. To install PANINIpy and its dependencies using this file:

.. code-block:: console
    
    git clone https://github.com/baiyueh/PANINIpy.git
    cd PANINIpy
    pip install -r requirements.txt


**Source Code and Discussion Channel**

Available on Github, `baiyueh/PANINIpy <https://github.com/baiyueh/PANINIpy/>`_.
Please report bugs, issues and feature extensions there. We also have `discussion channel <https://github.com/baiyueh/PANINIpy/discussions>`_ available to discuss anything related to *PANINIpy*:


**Minimal Working Example**

To ensure running in a working environment please make sure the python version ``>=3.9, <3.12``, and here is a simple example using the ``network_backbones`` module to verify the functionalities:

.. code-block:: python

    from paninipy.mdl_backboning import MDL_backboning

    # Define a weighted edge list
    elist = [
        (0, 1, 12), (0, 3, 20), (0, 4, 8),
        (1, 2, 1), (1, 4, 3),
        (2, 0, 1), (2, 1, 3),
        (3, 2, 3), (3, 4, 1),
        (4, 3, 1)
    ]

    # Compute backbones using out-edges
    backbone_global, backbone_local, compression_global, compression_local = MDL_backboning(
        elist, directed=True, out_edges=True
    )

    # Display results
    print('Global Backbone:', backbone_global)
    print('Inverse Compression Ratio (Global):', compression_global)
    print('Local Backbone:', backbone_local)
    print('Inverse Compression Ratio (Local):', compression_local)

If the installation was successful, running this code should display the global and local backbones along with their inverse compression ratios.

**Testing and Continuous Integration**

*PANINIpy* is equipped with automated testing on GitHub Actions to ensure that all functionalities work as expected. As part of CI/CD pipeline, this process helps maintain code quality and quickly catches any issues.

To run the tests manually, users need to clone the *PANINIpy* GitHub repository as the tests are not included in the PyPI package installation.

1. **Clone the Repository**: First, clone the *PANINIpy* GitHub repository to access the `Tests` folder.

   .. code-block:: console

       git clone https://github.com/baiyueh/PANINIpy.git
       cd PANINIpy

2. **Install the Required Packages**: Ensure all dependencies, including `pytest`, are installed by running the following commands:

   .. code-block:: console

       pip install -r requirements.txt
       pip install pytest

3. **Run the Tests**: Use `pytest` to execute all tests in the `Tests` folder.

   .. code-block:: console

       pytest Tests/

4. **View the Results**: The output will display any failed tests and provide detailed information on each failed test case.

GitHub Actions runs these tests automatically with the latest stable versions of Python and relevant dependencies each time code is pushed to the repository or a pull request is made. For more information about workflows runs, see `Workflows <https://github.com/baiyueh/PANINIpy/actions>`_ in the repository.

