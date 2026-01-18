"""
Corner + Aisle Graph Extraction
Uses Harris corner detection at aisle intersections.
"""
import cv2
import numpy as np
import networkx as nx

def cluster_points(points, min_dist=30):
    if len(points) == 0:
        return []
    clustered, used = [], [False] * len(points)
    for i, p1 in enumerate(points):
        if used[i]:
            continue
        cluster = [p1]
        used[i] = True
        for j, p2 in enumerate(points):
            if not used[j] and np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < min_dist:
                cluster.append(p2)
                used[j] = True
        clustered.append((int(np.mean([p[0] for p in cluster])), int(np.mean([p[1] for p in cluster]))))
    return clustered

def path_is_clear(p1, p2, floor_mask, threshold=0.85):
    samples = max(int(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)), 5)
    xs = np.linspace(p1[0], p2[0], samples).astype(int)
    ys = np.linspace(p1[1], p2[1], samples).astype(int)
    h, w = floor_mask.shape
    xs, ys = np.clip(xs, 0, w-1), np.clip(ys, 0, h-1)
    return np.mean(floor_mask[ys, xs]) > threshold * 255

def build_corner_aisle_graph(floor_mask):
    """Build graph from corner detection at aisle intersections."""
    h, w = floor_mask.shape
    floor_edges = cv2.Canny(floor_mask, 50, 150)
    corners = cv2.cornerHarris(floor_edges.astype(np.float32), 9, 3, 0.04)
    corners = cv2.dilate(corners, None)
    
    coords = np.argwhere(corners > 0.01 * corners.max())
    nodes = [(int(p[1]), int(p[0])) for p in coords 
             if 10 <= p[1] < w-10 and 10 <= p[0] < h-10 and floor_mask[p[0], p[1]] > 200]
    nodes = cluster_points(nodes, 30)
    
    G = nx.Graph()
    for i, pos in enumerate(nodes):
        G.add_node(i, pos=pos)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            dist = np.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2)
            if dist <= 150 and path_is_clear(nodes[i], nodes[j], floor_mask):
                G.add_edge(i, j, weight=dist)
    return G
