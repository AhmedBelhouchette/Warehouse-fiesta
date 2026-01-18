"""
Medial Axis Graph Extraction
Uses distance transform peaks as nodes.
"""
import cv2
import numpy as np
import networkx as nx
from scipy.ndimage import maximum_filter

def path_is_clear(p1, p2, floor_mask, threshold=0.85):
    samples = max(int(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)), 5)
    xs = np.linspace(p1[0], p2[0], samples).astype(int)
    ys = np.linspace(p1[1], p2[1], samples).astype(int)
    h, w = floor_mask.shape
    xs, ys = np.clip(xs, 0, w-1), np.clip(ys, 0, h-1)
    return np.mean(floor_mask[ys, xs]) > threshold * 255

def build_medial_axis_graph(floor_mask):
    """Build graph from medial axis (distance transform peaks)."""
    floor_binary = (floor_mask > 127).astype(np.uint8)
    dist_transform = cv2.distanceTransform(floor_binary, cv2.DIST_L2, 5)
    
    local_max = maximum_filter(dist_transform, size=20)
    medial_points = (dist_transform == local_max) & (dist_transform > 15)
    
    nodes = [(int(p[1]), int(p[0])) for p in np.argwhere(medial_points)]
    if len(nodes) > 200:
        indices = np.random.choice(len(nodes), 200, replace=False)
        nodes = [nodes[i] for i in indices]
    
    G = nx.Graph()
    for i, pos in enumerate(nodes):
        G.add_node(i, pos=pos)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            dist = np.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2)
            if dist <= 80 and path_is_clear(nodes[i], nodes[j], floor_mask):
                G.add_edge(i, j, weight=dist)
    return G
