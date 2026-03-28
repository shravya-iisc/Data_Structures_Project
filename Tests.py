# =========================
# Shravya's Section
# =========================

# TODO: Add property-based tests for SCC detection in NetworkX using Hypothesis



import networkx as nx
from hypothesis import given, strategies as st

# =========================
# Manju's Section
# =========================


def directed_graphs(min_nodes=1, max_nodes=20):
    return st.builds(
        lambda edges, n: _build_digraph(edges, n),
        edges=st.lists(
            st.tuples(st.integers(0, max_nodes - 1),
                      st.integers(0, max_nodes - 1)),
            min_size=0, max_size=300
        ),
        n=st.integers(min_nodes, max_nodes)
    )


def _build_digraph(edge_list, n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, v in edge_list:
        G.add_edge(u, v)
    return G


@given(directed_graphs())
def test_scc_reachability(G):
    """
    Property: Nodes in the same SCC must be mutually reachable.

    Mathematical Basis:
    Two nodes u and v belong to the same SCC iff:
        u → v  AND  v → u
    (both directions reachable)

    Why Important:
    This is *the definition* of SCC. Any violation indicates a core bug in
    SCC detection or in reachability routines.
    """
    sccs = list(nx.strongly_connected_components(G))

    for comp in sccs:
        comp = list(comp)
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                u, v = comp[i], comp[j]
                assert nx.has_path(G, u, v), f"FAIL: Path not found from {u} to {v}"
                assert nx.has_path(G, v, u), f"FAIL: Path not found from {v} to {u}"
    
    print(f"PASS: All {len(sccs)} SCC(s) satisfy mutual reachability property")
