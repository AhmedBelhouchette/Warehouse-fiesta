"""
Shortest Path Algorithms: Dijkstra and A*
"""
import numpy as np
import networkx as nx

def dijkstra_shortest_path(G, source, target):
    """
    Compute shortest path using Dijkstra's algorithm.
    
    Args:
        G: NetworkX graph
        source: Start node
        target: End node
    
    Returns:
        path: List of nodes in path
        dist: Total distance
    """
    try:
        path = nx.dijkstra_path(G, source, target, weight='weight')
        dist = nx.dijkstra_path_length(G, source, target, weight='weight')
        return path, dist
    except nx.NetworkXNoPath:
        return [], float('inf')


def astar_shortest_path(G, source, target):
    """
    Compute shortest path using A* algorithm.
    Uses Euclidean distance as heuristic.
    
    Args:
        G: NetworkX graph (nodes must have 'pos' attribute)
        source: Start node
        target: End node
    
    Returns:
        path: List of nodes in path
        dist: Total distance
    """
    def heuristic(u, v):
        p1 = G.nodes[u]['pos']
        p2 = G.nodes[v]['pos']
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    try:
        path = nx.astar_path(G, source, target, heuristic=heuristic, weight='weight')
        dist = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        return path, dist
    except nx.NetworkXNoPath:
        return [], float('inf')
