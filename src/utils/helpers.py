"""
Helper functions for graph building
"""
import numpy as np
import networkx as nx

def path_is_clear(p1, p2, floor_mask, threshold=0.85):
    """Check if straight line path between two points is on floor."""
    num_samples = max(int(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)), 5)
    x_coords = np.linspace(p1[0], p2[0], num_samples).astype(int)
    y_coords = np.linspace(p1[1], p2[1], num_samples).astype(int)
    h, w = floor_mask.shape
    x_coords = np.clip(x_coords, 0, w-1)
    y_coords = np.clip(y_coords, 0, h-1)
    path_values = floor_mask[y_coords, x_coords]
    return np.mean(path_values) > threshold * 255


def build_graph_from_nodes(nodes, floor_mask, max_dist=150):
    """Build a graph connecting nearby nodes if path is clear."""
    G = nx.Graph()
    for i, pos in enumerate(nodes):
        G.add_node(i, pos=pos)
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            p1, p2 = nodes[i], nodes[j]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if dist <= max_dist and path_is_clear(p1, p2, floor_mask):
                G.add_edge(i, j, weight=dist)
    return G


def cluster_points(points, min_dist=30):
    """Cluster nearby points into centroids."""
    if len(points) == 0:
        return []
    clustered = []
    used = [False] * len(points)
    for i, p1 in enumerate(points):
        if used[i]:
            continue
        cluster = [p1]
        used[i] = True
        for j, p2 in enumerate(points):
            if used[j]:
                continue
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if dist < min_dist:
                cluster.append(p2)
                used[j] = True
        centroid = (int(np.mean([p[0] for p in cluster])),
                   int(np.mean([p[1] for p in cluster])))
        clustered.append(centroid)
    return clustered
