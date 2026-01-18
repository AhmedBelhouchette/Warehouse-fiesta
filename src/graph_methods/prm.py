"""
PRM (Probabilistic Roadmap) Graph Extraction
Random sampling on floor areas.
"""
import numpy as np
import networkx as nx

def path_is_clear(p1, p2, floor_mask, threshold=0.85):
    samples = max(int(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)), 5)
    xs = np.linspace(p1[0], p2[0], samples).astype(int)
    ys = np.linspace(p1[1], p2[1], samples).astype(int)
    h, w = floor_mask.shape
    xs, ys = np.clip(xs, 0, w-1), np.clip(ys, 0, h-1)
    return np.mean(floor_mask[ys, xs]) > threshold * 255

def build_prm_graph(floor_mask, num_samples=100, seed=42):
    """Build graph using random sampling on floor."""
    h, w = floor_mask.shape
    np.random.seed(seed)
    
    nodes = []
    attempts = 0
    while len(nodes) < num_samples and attempts < 5000:
        x, y = np.random.randint(20, w-20), np.random.randint(20, h-20)
        if floor_mask[y, x] > 200:
            region = floor_mask[max(0,y-10):min(h,y+10), max(0,x-10):min(w,x+10)]
            if np.mean(region) > 200:
                nodes.append((x, y))
        attempts += 1
    
    G = nx.Graph()
    for i, pos in enumerate(nodes):
        G.add_node(i, pos=pos)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            dist = np.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2)
            if dist <= 80 and path_is_clear(nodes[i], nodes[j], floor_mask):
                G.add_edge(i, j, weight=dist)
    return G
