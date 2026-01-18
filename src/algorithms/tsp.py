"""
Traveling Salesman Problem - Nearest Neighbor Heuristic
"""
import numpy as np
import networkx as nx

def tsp_nearest_neighbor(G, terminals):
    """
    Solve TSP using nearest neighbor heuristic.
    
    Args:
        G: NetworkX graph
        terminals: List of nodes to visit
    
    Returns:
        route: Order of terminal indices
        total_dist: Total route distance
        dist_matrix: Distance matrix between terminals
        path_dict: Dictionary of paths between terminal pairs
    """
    n = len(terminals)
    
    # Compute distance matrix
    dist_matrix = np.full((n, n), np.inf)
    path_dict = {}
    
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i,j] = 0
            else:
                try:
                    dist_matrix[i,j] = nx.dijkstra_path_length(G, terminals[i], terminals[j], weight='weight')
                    path_dict[(i,j)] = nx.dijkstra_path(G, terminals[i], terminals[j], weight='weight')
                except:
                    pass
    
    # Nearest neighbor heuristic
    visited = [False] * n
    route = [0]
    visited[0] = True
    
    for _ in range(n - 1):
        current = route[-1]
        best_next = -1
        best_dist = float('inf')
        for j in range(n):
            if not visited[j] and dist_matrix[current, j] < best_dist:
                best_dist = dist_matrix[current, j]
                best_next = j
        if best_next == -1:
            break
        route.append(best_next)
        visited[best_next] = True
    
    # Calculate total (including return to start)
    total = sum(dist_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    total += dist_matrix[route[-1], route[0]]
    
    return route, total, dist_matrix, path_dict
