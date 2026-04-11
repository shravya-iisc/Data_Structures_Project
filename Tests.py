import networkx as nx
from hypothesis import given, strategies as st

# =========================
# Manju's Section
# =========================

# Property-based tests on SCC (Strongly Connected Components) Detection in NetworkX.
# SCCs are sets of nodes where every pair is mutually reachable.
# Below mentioned tests are to verify key properties of SCC algorithms using random directed graphs.

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

    Test Strategy:
    For each SCC, check every pair of nodes for bidirectional paths.

    Why this is important:
    This is the definition of SCC. Any violation indicates a core bug in
    SCC detection or in reachability routines.
    """
    sccs = list(nx.strongly_connected_components(G))

    for comp in sccs:
        comp = list(comp)
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                u, v = comp[i], comp[j]
                # Check path from u to v
                assert nx.has_path(G, u, v), f"FAIL: Path not found from {u} to {v}"
                # Check path from v to u
                assert nx.has_path(G, v, u), f"FAIL: Path not found from {v} to {u}"
    
    print(f"PASS: All {len(sccs)} SCC(s) satisfy mutual reachability property")


@given(directed_graphs())
def test_scc_partition(G):
    """
    Property: SCCs form a partition of the nodes - they are disjoint and cover all nodes.

    Mathematical Basis:
    SCCs should partition the vertex set: every node belongs to exactly one SCC,
    and SCCs are pairwise disjoint.

    Test Strategy:
    Collect all nodes in SCCs and check they match graph nodes exactly.
    Also ensure no node appears in multiple SCCs.

    Why this is important:
    This ensures no node is missed or duplicated in SCC computation.
    """
    sccs = list(nx.strongly_connected_components(G))
    
    # Check coverage: all nodes are in some SCC
    all_nodes = set(G.nodes)
    covered = set()
    for comp in sccs:
        covered.update(comp)
    assert covered == all_nodes, "FAIL: Not all nodes are covered by SCCs"
    
    # Check disjointness: no node in more than one SCC
    seen = set()
    for comp in sccs:
        for node in comp:
            assert node not in seen, f"FAIL: Node {node} in multiple SCCs"
            seen.add(node)
    
    print(f"PASS: SCCs form a valid partition of {len(all_nodes)} nodes")


@given(directed_graphs())
def test_condensation_is_dag(G):
    """
    Property: The condensation graph (where each SCC is a supernode) is a Directed Acyclic Graph (DAG).

    Mathematical Basis:
    A cycle in the condensation would imply that multiple SCCs form a bigger cycle.
    But SCCs are maximal sets of nodes such that every node can reach every other node

    If two SCCs had a cycle between them, they would not be separate SCCs—they would form one larger SCC.
    Thus cycles between SCCs are impossible.
    
    In the condensation, there are no cycles between SCCs, making it acyclic.
    This follows from the definition of SCCs.

    Test Strategy:
    Compute the condensation and check if it's a DAG using NetworkX.

    Why this is important:
    SCCs condense cycles into single nodes, so the resulting graph should be acyclic.
    """
    condensation = nx.condensation(G)
    # Verify it's acyclic
    assert nx.is_directed_acyclic_graph(condensation), "FAIL: Condensation graph has a cycle"
    
    print("PASS: Condensation graph is a DAG")


@given(directed_graphs())
def test_scc_reachability_implies_condensation_path(G):
    """
    Property: If there is a path from u to v in the original graph, then there is a path
    from SCC(u) to SCC(v) in the condensation graph.

    Mathematical Basis:
    Since paths between SCCs are preserved in the condensation,
    reachability in the original graph implies reachability in the condensation.

    Test Strategy:
    Map nodes to their SCCs, then for every reachable pair (u,v) where SCC(u) != SCC(v),
    check if SCC(u) reaches SCC(v) in condensation.

    Why this is important:
    This ensures the condensation correctly captures inter-SCC connectivity.
    """
    sccs = list(nx.strongly_connected_components(G))
    node_to_scc = {}
    for idx, comp in enumerate(sccs):
        for node in comp:
            node_to_scc[node] = idx
    
    condensation = nx.condensation(G)
    
    # Check for all pairs u,v: if path in G, then path in condensation
    for u in G.nodes:
        for v in G.nodes:
            if u != v and nx.has_path(G, u, v):
                scc_u = node_to_scc[u]
                scc_v = node_to_scc[v]
                if scc_u != scc_v:
                    assert nx.has_path(condensation, scc_u, scc_v), \
                        f"FAIL: Path from SCC {scc_u} to SCC {scc_v} missing in condensation"
    
    print("PASS: Original reachability implies condensation reachability")


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

#------------------------
# Postcondition-1
#------------------------
@given(generate_flow_input())
def test_source_outgoing_equals_flow_value(data):
    """
    Property:
    The total outgoing flow from the source node equals the computed maximum flow value.

    Mathematical basis:
    The value of the flow is defined as the net amount leaving the source.
    So the sum of flows on all edges outgoing from the source must equal the
    maximum flow computed by the algorithm.

    Test strategy:
    Generate random directed graphs with capacities using Hypothesis.
    Compute the maximum flow and compare it with the total outgoing flow from the source.

    Why this matters:
    If this property fails, it indicates inconsistency in how flow value is computed,
    suggesting a bug in the f.
    """
    G, s, t = data

    flow_value, flow_dict = nx.maximum_flow(G, s, t)
    outgoing = sum(flow_dict[s].get(v, 0) for v in G.nodes)

    assert flow_value == outgoing

    print("PASS: Source outgoing flow matches max flow value")

#------------------------
# Postcondition-2
#------------------------
@given(generate_flow_input())
def test_sink_incoming_equals_flow_value(data):
    """
    Property:
    The total incoming flow into the sink node equals the computed maximum flow value.

    Mathematical basis:
    By flow conservation, all flow leaving the source must eventually reach the sink.
    So the total incoming flow at the sink equals the total flow value.

    Test strategy:
    Generate random graphs and compare the sum of incoming flows at the sink with
    the maximum flow value.

    Why this matters:
    If this fails, it indicates incorrect flow propagation at the sink,
    violating conservation principles.
    """
    G, s, t = data

    flow_value, flow_dict = nx.maximum_flow(G, s, t)
    incoming = sum(flow_dict[v].get(t, 0) for v in G.nodes)

    assert flow_value == incoming

    print("PASS: Sink incoming flow matches max flow value")

#------------------------
# Metamorphic Property
#------------------------
@given(generate_flow_input())
def test_adding_edge_does_not_decrease_flow(data):
    """
    Property:
    Adding a direct edge from source (s) to sink (t) 
    cannot decrease the maximum flow.

    Mathematical basis:
    Adding an edge increases or preserves the number of possible augmenting paths.
    The original flow remains valid, and a new direct path from s to t may allow
    additional flow. So, the maximum flow value cannot decrease.

    Test strategy:
    Generate a random directed graph with capacities.
    Compute the maximum flow.
    Then add a direct edge from source to sink with positive capacity.
    Recompute the flow and verify that it does not decrease.

    Why this matters:
    If violated, it indicates incorrect handling of graph updates or flow computation.
    """
    G, s, t = data

    flow_value_1, _ = nx.maximum_flow(G, s, t)
    # Copy graph
    G2 = G.copy()
    # Add direct edge from source to sink (strongest possible path (edge-case))
    if not G2.has_edge(s, t):
        G2.add_edge(s, t, capacity=10)
        flow_value_2, _ = nx.maximum_flow(G2, s, t)

        assert flow_value_2 >= flow_value_1

        print("PASS: Adding source→sink edge does not decrease max flow")

        # =========================
        # Hafsa's Section
        # =========================

        # Property-based tests for Dijkstra's Algorithm in NetworkX.
        # Dijkstra's finds the shortest paths from a source node to all others in graphs with non-negative weights.

        # =========================
        # Hafsa's Section: Dijkstra
        # =========================

    @st.composite
    def weighted_graphs(draw):
        """Custom strategy for weighted graphs with non-negative weights[cite: 57]."""
        n = draw(st.integers(2, 15))
        edges = draw(st.lists(st.tuples(st.integers(0, n - 1), st.integers(0, n - 1), st.integers(0, 100))))
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, w in edges:
            if u != v: G.add_edge(u, v, weight=w)
        return G, draw(st.integers(0, n - 1))

    @given(weighted_graphs())
    def test_dijkstra_triangle_inequality(data):
        """
        Property: Shortest paths must satisfy Triangle Inequality[cite: 18, 30].
        Mathematical Basis: For nodes u, v, w, dist(u, w) <= dist(u, v) + weight(v, w)[cite: 31].
        Test Strategy: Verify the inequality for all reachable edges[cite: 31].
        """
        G, s = data
        dist = nx.single_source_dijkstra_path_length(G, s)
        for v, w in G.edges:
            if v in dist and w in dist:
                weight = G[v][w]['weight']
                assert dist[w] <= dist[v] + weight
                assert dist[v] <= dist[w] + weight

    @given(weighted_graphs())
    def test_dijkstra_path_optimality(data):
        """
        Property: Sub-paths of shortest paths are also shortest paths[cite: 18, 30].
        Mathematical Basis: If P is a shortest path, every sub-segment must be optimal[cite: 31].
        Test Strategy: Verify dist[target] = dist[predecessor] + edge_weight[cite: 31].
        """
        G, s = data
        paths = nx.single_source_dijkstra_path(G, s)
        dist = nx.single_source_dijkstra_path_length(G, s)
        for target, path in paths.items():
            if len(path) > 1:
                prev_node = path[-2]
                assert dist[target] == dist[prev_node] + G[prev_node][target]['weight']

    @given(weighted_graphs())
    def test_dijkstra_non_negativity(data):
        """Property: All computed distances must be non-negative[cite: 23]."""
        G, s = data
        dist = nx.single_source_dijkstra_path_length(G, s)
        assert dist[s] == 0
        for d in dist.values():
            assert d >= 0

    @given(weighted_graphs())
    def test_dijkstra_weight_scaling(data):
        """
        Property: Scaling all weights by k scales all distances by k[cite: 24, 30].
        Metamorphic Property: Linear transformation of input preserves path structure[cite: 31].
        """
        G, s = data
        dist1 = nx.single_source_dijkstra_path_length(G, s)
        G2 = G.copy()
        for u, v in G2.edges: G2[u][v]['weight'] *= 2
        dist2 = nx.single_source_dijkstra_path_length(G2, s)
        for node in dist1:
            assert dist2[node] == dist1[node] * 2

    @given(weighted_graphs())
    def test_dijkstra_monotone_increase(data):
        """
        Property: Increasing an edge weight cannot decrease path length[cite: 24, 30].
        Test Strategy: Increase one edge weight and ensure no distance decreases[cite: 31].
        """
        G, s = data
        dist1 = nx.single_source_dijkstra_path_length(G, s)
        if not G.edges: return
        u, v = list(G.edges)[0]
        G[u][v]['weight'] += 10
        dist2 = nx.single_source_dijkstra_path_length(G, s)
        for node in dist1:
            assert dist2[node] >= dist1[node]

