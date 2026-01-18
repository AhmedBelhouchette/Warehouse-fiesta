"""
Voronoi-based Graph Extraction
Creates graph from Voronoi diagram of obstacle boundaries.
"""
import cv2
import numpy as np
import networkx as nx
from scipy.spatial import Voronoi

def path_is_clear(p1, p2, floor_mask, threshold=0.85):
    samples = max(int(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)), 5)
    xs = np.linspace(p1[0], p2[0], samples).astype(int)
    ys = np.linspace(p1[1], p2[1], samples).astype(int)
    h, w = floor_mask.shape
    xs, ys = np.clip(xs, 0, w-1), np.clip(ys, 0, h-1)
    return np.mean(floor_mask[ys, xs]) > threshold * 255

def build_voronoi_graph(floor_mask, obstacle_mask):
    """Build graph from Voronoi diagram of obstacles."""
    h, w = floor_mask.shape
    
    obstacle_points = []
    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        for i in range(0, len(cnt), 10):
            obstacle_points.append(cnt[i][0].tolist())
    
    # Add boundary points
    for x in range(0, w, 20):
        obstacle_points.extend([[x, 0], [x, h-1]])
    for y in range(0, h, 20):
        obstacle_points.extend([[0, y], [w-1, y]])
    
    if len(obstacle_points) < 4:
        return nx.Graph()
    
    vor = Voronoi(np.array(obstacle_points))
    nodes = [(int(v[0]), int(v[1])) for v in vor.vertices 
             if 10 <= v[0] < w-10 and 10 <= v[1] < h-10 and floor_mask[int(v[1]), int(v[0])] > 200]
    
    G = nx.Graph()
    for i, pos in enumerate(nodes):
        G.add_node(i, pos=pos)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            dist = np.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2)
            if dist <= 100 and path_is_clear(nodes[i], nodes[j], floor_mask):
                G.add_edge(i, j, weight=dist)
    return G
