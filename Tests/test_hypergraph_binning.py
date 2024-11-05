import time
import numpy as np
import pytest
from paninipy.hypergraph_binning import MDL_hypergraph_binning

def test_hypergraph_binning():
    # Input data and parameters
    X = [
        ('A', '1', 1, 0.1), ('B', '1', 1, 0.2), ('B', '1', 1, 0.3), 
        ('A', '1', 1, 0.4), ('B', '1', 1, 0.5), ('B', '1', 1, 1), 
        ('C', '2', 1, 2), ('A', '2', 1, 3), ('B', '2', 1, 4), ('C', '2', 1, 5)
    ]
    dt = 0.1
    
    # Run exact algorithm and record runtime
    start_exact = time.time()
    results_exact = MDL_hypergraph_binning(X, dt, exact=True)
    runtime_exact = time.time() - start_exact

    # Run greedy algorithm and record runtime
    start_greedy = time.time()
    results_greedy = MDL_hypergraph_binning(X, dt, exact=False)
    runtime_greedy = time.time() - start_greedy

    # Assertions for exact algorithm results
    assert isinstance(results_exact, tuple), "Exact: Results should be a tuple"
    assert len(results_exact) == 3, "Exact: Expected three elements in the result tuple"
    assert isinstance(results_exact[0], float), "Exact: Compression ratio should be a float"
    assert isinstance(results_exact[1], np.ndarray), "Exact: Event partition should be an array"
    assert isinstance(results_exact[2], int), "Exact: Number of time steps should be an integer"
    assert runtime_exact < 1, "Exact: Runtime should be reasonably fast (<1s for small inputs)"

    # Expected values based on sample output
    assert round(results_exact[0], 4) == 0.9842, "Exact: Unexpected compression ratio"
    assert np.array_equal(results_exact[1], np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])), "Exact: Unexpected event partition"
    assert results_exact[2] == 50, "Exact: Unexpected number of time steps"

    # Assertions for greedy algorithm results
    assert isinstance(results_greedy, tuple), "Greedy: Results should be a tuple"
    assert len(results_greedy) == 3, "Greedy: Expected three elements in the result tuple"
    assert isinstance(results_greedy[0], float), "Greedy: Compression ratio should be a float"
    assert isinstance(results_greedy[1], np.ndarray), "Greedy: Event partition should be an array"
    assert isinstance(results_greedy[2], int), "Greedy: Number of time steps should be an integer"
    assert runtime_greedy < 1, "Greedy: Runtime should be reasonably fast (<1s for small inputs)"

    # Expected values based on sample output
    assert round(results_greedy[0], 4) == 0.9913, "Greedy: Unexpected compression ratio"
    assert np.array_equal(results_greedy[1], np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])), "Greedy: Unexpected event partition"
    assert results_greedy[2] == 50, "Greedy: Unexpected number of time steps"

if __name__ == "__main__":
    pytest.main()
