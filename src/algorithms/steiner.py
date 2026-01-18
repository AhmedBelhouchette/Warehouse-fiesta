"""
Steiner Tree - MST Approximation
"""
import numpy as np
import networkx as nx

def steiner_tree_mst(G, terminals):
    """
    Compute Steiner tree using MST approximation.
    
    Args:
        G: NetworkX graph
        terminals: List of terminal nodes to connect
    
    Returns:
        steiner_tree: NetworkX graph of the Steiner tree
        steiner_nodes: Set of all nodes in tree (terminals + intermediate)
        total_weight: Total edge weight of tree
        path_dict: Dictionary of paths between terminal pairs
    """
    n = len(terminals)
    
    # Compute distance matrix
    dist_matrix = np.full((n, n), np.inf)
    path_dict = {}
    
    for i in range(n):
        for j in range(i+1, n):
            try:
                dist_matrix[i,j] = nx.dijkstra_path_length(G, terminals[i], terminals[j], weight='weight')
                dist_matrix[j,i] = dist_matrix[i,j]
                path_dict[(i,j)] = nx.dijkstra_path(G, terminals[i], terminals[j], weight='weight')
            except:
                pass
    
    # Build complete graph on terminals
    complete_G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i,j] < float('inf'):
                complete_G.add_edge(i, j, weight=dist_matrix[i,j])
    
    if complete_G.number_of_edges() == 0:
        return nx.Graph(), set(), 0, path_dict
    
    # MST of complete graph
    mst = nx.minimum_spanning_tree(complete_G, weight='weight')
    
    # Build Steiner tree from MST
    steiner_tree = nx.Graph()
    steiner_nodes = set()
    
    for u, v in mst.edges():
        path = path_dict.get((u, v)) or path_dict.get((v, u))
        if path:
            for node in path:
                steiner_nodes.add(node)
            for k in range(len(path) - 1):
                n1, n2 = path[k], path[k+1]
                if not steiner_tree.has_edge(n1, n2):
                    steiner_tree.add_edge(n1, n2, weight=G[n1][n2]['weight'])
    
    total_weight = sum(steiner_tree[u][v]['weight'] for u, v in steiner_tree.edges())
    
    return steiner_tree, steiner_nodes, total_weight, path_dict
