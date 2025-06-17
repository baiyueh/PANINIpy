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
    expected_global_backbone = [(0, 3, 20), (0, 1, 12), (0, 4, 8)]
    expected_global_compression = 0.7114680371853184
    expected_local_backbone = [(0, 3, 20), (1, 4, 3), (2, 1, 3), (3, 2, 3)]
    expected_local_compression = 0.8897699752517169

    # Compute backbones using out-edges
    backbone_global, backbone_local, compression_global, compression_local = MDL_backboning(
        elist, directed=True, out_edges=True
    )

    # Assertions to check if results match expected values
    assert backbone_global == expected_global_backbone, "Global Backbone does not match expected result"
    assert backbone_local == expected_local_backbone, "Local Backbone does not match expected result"
    assert pytest.approx(compression_global, rel=1e-6) == expected_global_compression, "Global Compression does not match expected result"
    assert pytest.approx(compression_local, rel=1e-6) == expected_local_compression, "Local Compression does not match expected result"

    # Compute backbones with undirected edges
    expected_global_backbone = [(0, 1, 12), (0, 4, 8), (0, 3, 20)]
    expected_global_compression = 0.7348799446767645
    expected_local_backbone = [(0, 1, 12), (0, 4, 8), (1, 2, 3), (0, 3, 20)]
    expected_local_compression = 0.8223220603580707

    backbone_global, backbone_local, compression_global, compression_local = MDL_backboning(
        elist, directed=True, out_edges=True
    )

    assert backbone_global == expected_global_backbone, "Global Backbone does not match expected result"
    assert backbone_local == expected_local_backbone, "Local Backbone does not match expected result"
    assert pytest.approx(compression_global, rel=1e-6) == expected_global_compression, "Global Compression does not match expected result"
    assert pytest.approx(compression_local, rel=1e-6) == expected_local_compression, "Local Compression does not match expected result"

if __name__ == "__main__":
    pytest.main()
