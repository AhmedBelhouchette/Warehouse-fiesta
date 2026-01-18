"""
Visibility Graph Extraction
Connects obstacle corners with line-of-sight edges.
"""
import cv2
import numpy as np
import networkx as nx

def path_is_clear(p1, p2, floor_mask, threshold=0.85):
    samples = max(int(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)), 5)
    xs = np.linspace(p1[0], p2[0], samples).astype(int)
    ys = np.linspace(p1[1], p2[1], samples).astype(int)
    h, w = floor_mask.shape
    xs, ys = np.clip(xs, 0, w-1), np.clip(ys, 0, h-1)
    return np.mean(floor_mask[ys, xs]) > threshold * 255

def build_visibility_graph(floor_mask, obstacle_mask):
    """Build graph connecting obstacle corners with visibility check."""
    h, w = floor_mask.shape
    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corners = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        for point in approx:
            x, y = point[0]
            for dx, dy in [(-10,-10), (-10,10), (10,-10), (10,10)]:
                nx_, ny_ = x+dx, y+dy
                if 0 <= nx_ < w and 0 <= ny_ < h and floor_mask[ny_, nx_] > 200:
                    corners.append((nx_, ny_))
                    break
    
    corners = list(set(corners))
    G = nx.Graph()
    for i, pos in enumerate(corners):
        G.add_node(i, pos=pos)
    for i in range(len(corners)):
        for j in range(i+1, len(corners)):
            dist = np.sqrt((corners[i][0]-corners[j][0])**2 + (corners[i][1]-corners[j][1])**2)
            if dist <= 300 and path_is_clear(corners[i], corners[j], floor_mask):
                G.add_edge(i, j, weight=dist)
    return G
