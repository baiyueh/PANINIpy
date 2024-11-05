import pytest
import numpy as np
from paninipy.distributional_regionalization import MDL_regionalization

def test_MDL_regionalization():
    # Initialize the MDL regionalization instance
    name = "Test MDL Regionalization"
    mdl = MDL_regionalization(name)

    # Define adjacency list 
    adjlist = [
        [1, 2],  
        [0, 2, 3],  
        [0, 1, 3],  
        [1, 2, 4], 
        [3]  
    ]

    # Define distributions as a normalized probability matrix 
    dists = np.array([
        [0.3, 0.7],
        [0.5, 0.5],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.4, 0.6]
    ])

    # Define population for each node
    pops = [100, 150, 120, 130, 110]

    # Run the MDL regionalization
    compression_ratio, labels, output_dists = mdl.MDL_regionalization(adjlist, dists, pops)

    # Assertions to validate outputs
    assert isinstance(compression_ratio, float), "Compression ratio should be a float"
    assert 0 <= compression_ratio <= 1, "Compression ratio should be between 0 and 1"
    assert compression_ratio == 1.0, "Compression ratio did not match expected value of 1.0"

    assert isinstance(labels, list), "Labels should be a list"
    assert len(labels) == len(adjlist), "Labels length should match number of nodes"
    assert all(isinstance(label, int) for label in labels), "Each label should be an integer"

    # Check that output distributions match input distributions
    assert np.allclose(output_dists, dists), "Output distributions should match input distributions"

if __name__ == "__main__":
    pytest.main()
