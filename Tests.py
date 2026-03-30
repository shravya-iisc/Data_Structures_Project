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


# =========================
# Shravya's Section
# =========================

# Property-based tests on Maximum Flow NetworkX functions, using the Hypothesis library

# ---------- Property test input generator ----------
@st.composite
def generate_flow_input(draw):
    n = draw(st.integers(min_value=2, max_value=20))
    edges = draw(
        st.lists(
            st.tuples(
                st.integers(0, n - 1),
                st.integers(0, n - 1),
                st.integers(1, 20)
            ),
            min_size=1,
            max_size=n * n
        )
    )
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for u, v, cap in edges:
        # Avoid self loops
        if u != v:
            G.add_edge(u, v, capacity=cap)

    # ensure s != t 
    s = draw(st.integers(0, n - 1))
    possible_t = list(range(n))
    possible_t.remove(s)
    t = draw(st.sampled_from(possible_t))

    return G, s, t

#------------------------
# Invariant-1
#------------------------
@given(generate_flow_input())
def test_capacity_constraint(data):
    """
    Property:
    For every edge (u, v), the flow f(u, v) satisfies:
    0 ≤ f(u, v) ≤ capacity(u, v).

    Mathematical basis:
    This is a fundamental constraint for a feasible flow. 
    Flow cannot be negative,
    and it cannot exceed the capacity assigned to an edge.

    Test strategy:
    Random directed graphs with capacities are generated using Hypothesis.
    For each generated graph, maximum flow is computed using NetworkX.
    The flow assigned to each edge is checked against its capacity.

    Why this matters:
    If this property fails, the algorithm is producing an invalid flow that
    violates basic feasibility invariants, indicating a serious correctness issue.
    """
    G, s, t = data
    flow_value, flow_dict = nx.maximum_flow(G, s, t)

    for u, v in G.edges:
        flow = flow_dict[u][v]
        capacity = G[u][v]['capacity']

        assert flow >= 0
        assert flow <= capacity
    
    print("PASS: All edges satisfy capacity constraints")

#------------------------
# Invariant-2
#------------------------
@given(generate_flow_input())
def test_flow_conservation(data):
    """
    Property:
    For every node u other than source (s) and sink (t),
    the total incoming flow equals the total outgoing flow.

    Mathematical basis:
    This follows from the conservation of flow principle.
    Intermediate nodes cannot create or destroy flow; they only pass it along.

    Test strategy:
    Random directed graphs with capacities are generated.
    After computing maximum flow, for each intermediate node,
    the sum of incoming flows is compared with the sum of outgoing flows.

    Why this matters:
    If this property fails, it indicates that flow is either being lost
    or artificially created at some node, violating fundamental flow rules.
    """
    G, s, t = data
    flow_value, flow_dict = nx.maximum_flow(G, s, t)

    for u in G.nodes:
        if u == s or u == t:
            continue

        incoming = sum(flow_dict[v].get(u, 0) for v in G.nodes)
        outgoing = sum(flow_dict[u].get(v, 0) for v in G.nodes)

        assert incoming == outgoing
    
    print("PASS: Flow is conserved at all intermediate nodes")

#------------------------
# Equivalence checking
#------------------------

@given(generate_flow_input())
def test_maxflow_mincut(data):
    """
    Property:
    The value of the maximum flow from source (s) to sink (t) is equal to
    the capacity of the minimum cut separating s and t.

    Mathematical basis:
    This is the Max-Flow Min-Cut Theorem, a fundamental theorem in graph theory.
    It states that the maximum achievable flow in a network is equal to
    the capacity of the smallest cut that separates the source from the sink.

    Test strategy:
    Random directed graphs with capacities are generated using Hypothesis.
    For each graph, we compute:
      - Maximum flow value using nx.maximum_flow
      - Minimum cut capacity using nx.minimum_cut
    The test verifies that both values are equal.

    Why this matters:
    If this property fails, it indicates a serious violation of flow theory
    or a bug in the max-flow or min-cut implementation, since these two
    quantities are theoretically guaranteed to be equal.
    """
    G, s, t = data

    flow_value, flow_dict = nx.maximum_flow(G, s, t)
    cut_value, (reachable, non_reachable) = nx.minimum_cut(G, s, t)
    # where reachable is set of nodes that can still be reached from source s in the residual graph
    # while non_reachable is the set of nodes that are not reachable from s after cut
    
    assert flow_value == cut_value

    print("PASS: Max-flow equals min-cut")
