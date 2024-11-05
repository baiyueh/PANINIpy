import pytest
from paninipy.community_regionalization import greedy_opt

def test_community_regionalization():
    N = 5  
    # Define spatial adjacency as a list of tuples (no repeats)
    spatial_elist = [
        (0, 1), (0, 2), (1, 2),
        (1, 3), (2, 3), (3, 4)
    ]

    # Define flow edges with weights
    flow_elist = [
        (0, 1, 10), (1, 2, 5), (2, 0, 3),
        (1, 3, 8), (3, 4, 6)
    ]

    # Run the greedy_opt function
    DLs, partitions = greedy_opt(N, spatial_elist, flow_elist)

    # Expected values based on the given output example
    expected_DLs = [
        54.09185576642525, 
        46.163863508956936, 
        36.97257159073716, 
        29.22840733112082, 
        22.84656602784012
    ]
    
    expected_partitions = [
        {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}},
        [{0}, {1}, {2}, {3, 4}],
        [{1}, {3, 4}, {0, 2}],
        [{0, 2}, {1, 3, 4}],
        [{0, 1, 2, 3, 4}]
    ]

    # Verify output types and structure
    assert isinstance(DLs, list), "DLs should be a list"
    assert all(isinstance(dl, float) for dl in DLs), "Each item in DLs should be a float"
    assert isinstance(partitions, list), "Partitions should be a list"
    assert all(isinstance(partition, (dict, list)) for partition in partitions), "Each partition should be a dict or list of clusters"

    # Check that DLs and partitions match the expected values
    assert DLs == expected_DLs, f"Expected DLs {expected_DLs}, but got {DLs}"
    assert partitions == expected_partitions, f"Expected partitions {expected_partitions}, but got {partitions}"

if __name__ == "__main__":
    pytest.main()
