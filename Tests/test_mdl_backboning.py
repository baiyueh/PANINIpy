import pytest
from paninipy.mdl_backboning import MDL_backboning

def test_mdl_backboning():
    # Define a weighted edge list
    elist = [
        (0, 1, 12), (0, 3, 20), (0, 4, 8),
        (1, 2, 1), (1, 4, 3),
        (2, 0, 1), (2, 1, 3),
        (3, 2, 3), (3, 4, 1),
        (4, 3, 1)
    ]

    # Expected results
    expected_global_backbone_directed = [(0, 3, 20), (0, 1, 12), (0, 4, 8)]
    expected_local_backbone_directed  = [(0, 3, 20), (1, 4, 3), (2, 1, 3), (3, 2, 3)]

    # Compute backbones using out-edges
    backbone_global_directed, backbone_local_directed, compression_global_directed, compression_local_directed = MDL_backboning(
        elist, directed=True, out_edges=True
    )

    # Assertions to check if results match expected values
    assert backbone_global_directed  == expected_global_backbone_directed, "Global Backbone does not match expected result"
    assert backbone_local_directed  == expected_local_backbone_directed, "Local Backbone does not match expected result"
    assert isinstance(compression_global_directed, float), "Global Compression is not a float"
    assert isinstance(compression_local_directed, float), "Local Compression is not a float"

    # Compute backbones with undirected edges
    expected_global_backbone_undirected = [(0, 1, 12), (0, 4, 8), (0, 3, 20)]
    expected_local_backbone_undirected = [(0, 1, 12), (0, 4, 8), (1, 2, 3), (0, 3, 20)]

    backbone_global_undirected, backbone_local_undirected, compression_global_undirected, compression_local_undirected = MDL_backboning(
        elist, directed=True, out_edges=True
    )

    assert backbone_global_undirected == expected_global_backbone_undirected, "Global Backbone does not match expected result"
    assert backbone_local_undirected == expected_local_backbone_undirected, "Local Backbone does not match expected result"
    assert isinstance(compression_global_undirected, float), "Global Compression is not a float"
    assert isinstance(compression_local_undirected, float), "Local Compression is not a float"

if __name__ == "__main__":
    pytest.main()
