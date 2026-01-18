"""
Grid-based Graph Extraction (BEST METHOD)
Nodes placed uniformly on floor areas, connected to neighbors.
"""
import cv2
import numpy as np
import networkx as nx

def path_is_clear(p1, p2, floor_mask, threshold=0.85):
    num_samples = max(int(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)), 5)
    x_coords = np.linspace(p1[0], p2[0], num_samples).astype(int)
    y_coords = np.linspace(p1[1], p2[1], num_samples).astype(int)
    h, w = floor_mask.shape
    x_coords = np.clip(x_coords, 0, w-1)
    y_coords = np.clip(y_coords, 0, h-1)
    return np.mean(floor_mask[y_coords, x_coords]) > threshold * 255

def build_grid_graph(floor_mask, grid_spacing=25):
    """
    Build graph using uniform grid on floor areas.
    
    Args:
        floor_mask: Binary mask (WHITE=floor, BLACK=obstacle)
        grid_spacing: Distance between grid points (default 25px)
    
    Returns:
        G: NetworkX graph with nodes at valid grid positions
    """
    h, w = floor_mask.shape
    nodes = []
    
    for y in range(grid_spacing, h - grid_spacing, grid_spacing):
        for x in range(grid_spacing, w - grid_spacing, grid_spacing):
            region = floor_mask[max(0,y-8):min(h,y+8), max(0,x-8):min(w,x+8)]
            if np.mean(region) > 200:
                nodes.append((x, y))
    
    G = nx.Graph()
    for i, pos in enumerate(nodes):
        G.add_node(i, pos=pos)
    
    max_dist = grid_spacing * 1.5
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            p1, p2 = nodes[i], nodes[j]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if dist <= max_dist and path_is_clear(p1, p2, floor_mask):
                G.add_edge(i, j, weight=dist)
    
    return G
