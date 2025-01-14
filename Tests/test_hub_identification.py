import pytest
from paninipy.hub_identification import Network_hubs

def test_hub_identification():
    name = "Test Network Hubs"
    nh = Network_hubs(name)

    # Define parameters for the test
    N = 5 
    data = [
        (0, 1, 10), (1, 2, 5), (2, 0, 3),
        (1, 3, 8), (3, 4, 6)
    ]

    # Run the hubs method
    results = nh.hubs(data, N, degree_list=False, out_degrees=False, weighted=True)

    # Assertions for ER model
    assert results['ER']['hub_nodes'].size == 0, "ER model: Expected no hub nodes"
    assert results['ER']['hub_degrees'] == [], "ER model: Expected no hub degrees"
    assert isinstance(results['ER']['description_length'], float), "ER model: Description length should be a float"
    assert 0 < results['ER']['compression_ratio'] <= 1, "ER model: Compression ratio should be in (0, 1]"

    # Assertions for CM model
    assert results['CM']['hub_nodes'].size == 0, "CM model: Expected no hub nodes"
    assert results['CM']['hub_degrees'] == [], "CM model: Expected no hub degrees"
    assert isinstance(results['CM']['description_length'], float), "CM model: Description length should be a float"
    assert 0 < results['CM']['compression_ratio'] <= 1, "CM model: Compression ratio should be in (0, 1]"

    # Assertions for Average Model
    assert results['AVG']['hub_nodes'] == [1, 3], "AVG model: Unexpected hub nodes"
    assert results['AVG']['hub_degrees'] == [10.0, 8.0], "AVG model: Unexpected hub degrees"

    # Assertions for Loubar Model
    assert results['LOUBAR']['hub_nodes'] == [1, 3, 4], "Loubar model: Unexpected hub nodes"
    assert results['LOUBAR']['hub_degrees'] == [10.0, 8.0], "Loubar model: Unexpected hub degrees for AVG"

if __name__ == "__main__":
    pytest.main()
