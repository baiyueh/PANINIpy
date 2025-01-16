import pytest
import numpy as np
from collections import Counter
from paninipy.partial_rankings import partial_rankings, get_N, get_M, get_edges

# Test cases for the Partial Rankings module
def test_preprocessing():
    # Input data
    matchlist = np.array([
        ["A", "B"], ["A", "C"], ["B", "C"],
        ["C", "A"], ["B", "A"], ["C", "B"]
    ])

    # Test get_N
    N = get_N(matchlist)
    assert N == 3, "The number of unique items should be 3"

    # Test get_M
    M = get_M(matchlist)
    assert M == 6, "The number of matches should be 6"

    # Test get_edges
    e_out, e_int = get_edges(matchlist)
    total_out_edges = sum(len(targets) for targets in e_out.values())
    total_in_edges = sum(len(targets) for targets in e_int.values())

    assert total_out_edges == 6, "The number of out-edges should match the number of matches"
    assert total_in_edges == 6, "The number of in-edges should match the number of matches"

def test_partial_rankings():
    # Input data
    matchlist = np.array([
        ["A", "B"], ["A", "C"], ["B", "C"],
        ["C", "A"], ["B", "A"], ["C", "B"]
    ])

    # Preprocess data
    N = get_N(matchlist)
    M = get_M(matchlist)
    e_out, e_int = get_edges(matchlist)

    # Compute partial rankings
    results = partial_rankings(N, M, e_out, e_int, full_trace=True)

    # Test result structure
    assert isinstance(results, list), "Results should be a list"
    assert all(isinstance(res, dict) for res in results), "Each result should be a dictionary"

    # Test specific result values
    best_result = results[np.argmin([res['DL'] for res in results])]
    assert 'Strengths' in best_result, "Best result should include Strengths"
    assert 'Clusters' in best_result, "Best result should include Clusters"

def test_graph_creation():
    from networkx import DiGraph

    # Input data
    matchlist = np.array([
        ["A", "B"], ["A", "C"], ["B", "C"],
        ["C", "A"], ["B", "A"], ["C", "B"]
    ])

    # Count match frequencies
    match_count = Counter([tuple(match) for match in matchlist])
    weighted_matchlist = np.array([
        [match[0], match[1], count] for match, count in match_count.items()
    ])

    # Create graph
    def _nx_from_match_list(matchlist):
        g = DiGraph()
        for match in matchlist:
            i, j = match[0], match[1]
            weight = int(match[2]) if len(match) == 3 else 1
            g.add_edge(i, j, weight=weight)
        return g

    graph = _nx_from_match_list(weighted_matchlist)

    # Test graph structure
    assert isinstance(graph, DiGraph), "The graph should be a NetworkX DiGraph"
    assert len(graph.nodes) == 3, "The graph should have 3 nodes"
    assert len(graph.edges) == 6, "The graph should have 6 edges"

if __name__ == "__main__":
    pytest.main()
