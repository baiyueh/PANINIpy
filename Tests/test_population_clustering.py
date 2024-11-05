import pytest
import numpy as np
from paninipy.population_clustering import generate_synthetic, MDL_populations

def test_population_clustering():
    # Define parameters for synthetic generation
    mode_example = [
        {(0, 1), (0, 4), (0, 5), (1, 4), (1, 5), (1, 2), (3, 7), (6, 7)},
        {(0, 1), (0, 4), (1, 2), (1, 5), (1, 6), (2, 5), (2, 6), (2, 3), (5, 6)},
        {(1, 5), (4, 5), (5, 6), (2, 3), (2, 6), (3, 6), (3, 7), (2, 7), (6, 7)}
    ]
    node_num = 8
    network_num = 10

    # Generate synthetic data
    nets, cluster_labels = generate_synthetic(
        S=network_num,
        N=node_num,
        modes=mode_example,
        alphas=[1, 1, 1],
        betas=[0.1, 0.1, 0.1],
        pis=[0.33, 0.33, 0.34]
    )

    # Assertions to check structure and data validity
    assert isinstance(nets, list), "Generated networks should be a list"
    assert len(nets) == network_num, f"Expected {network_num} networks, got {len(nets)}"
    assert isinstance(cluster_labels, list), "Cluster labels should be a list"
    assert len(cluster_labels) == network_num, "Cluster labels length should match the number of networks"

    # Initialize MDL populations model
    mdl_pop = MDL_populations(edgesets=nets, N=node_num, K0=1, n_fails=100, directed=False, max_runs=np.inf)
    mdl_pop.initialize_clusters()
    
    # Run clustering
    clusters, modes, L = mdl_pop.run_sims()

    # Assertions to check clustering output structure and values
    assert isinstance(clusters, dict), "Clusters should be a dictionary"
    assert isinstance(modes, dict), "Modes should be a dictionary"
    assert isinstance(L, float), "Inverse compression ratio should be a float"
    assert 0 <= L <= 1, "Inverse compression ratio should be between 0 and 1"

    # Check if clustering output aligns with sample output for values
    assert len(clusters) > 0, "Clusters dictionary should not be empty"
    assert len(modes) > 0, "Modes dictionary should not be empty"

if __name__ == "__main__":
    pytest.main()
