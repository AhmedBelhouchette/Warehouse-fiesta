"""
Skeleton-based Graph Extraction
Extracts graph from morphological skeleton of floor area.
"""
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

def cluster_points(points, min_dist=20):
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
            if np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < min_dist:
                cluster.append(p2)
                used[j] = True
        clustered.append((int(np.mean([p[0] for p in cluster])),
                         int(np.mean([p[1] for p in cluster]))))
    return clustered

def build_skeleton_graph(floor_mask):
    """Build graph from skeleton of floor area."""
    h, w = floor_mask.shape
    skeleton = skeletonize(floor_mask > 127)
    skeleton_img = img_as_ubyte(skeleton)
    
    skel_binary = (skeleton_img > 127).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_binary, -1, kernel) * skel_binary - skel_binary
    
    endpoints = [(int(p[1]), int(p[0])) for p in np.argwhere(neighbor_count == 1)]
    intersections = [(int(p[1]), int(p[0])) for p in np.argwhere(neighbor_count >= 3)]
    
    nodes = cluster_points(endpoints, 20) + cluster_points(intersections, 20)
    skel_dilated = cv2.dilate(skel_binary, np.ones((5,5), np.uint8), iterations=1)
    
    G = nx.Graph()
    for i, pos in enumerate(nodes):
        G.add_node(i, pos=pos)
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            p1, p2 = nodes[i], nodes[j]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if dist < 200:
                samples = max(int(dist), 10)
                xs = np.linspace(p1[0], p2[0], samples).astype(int)
                ys = np.linspace(p1[1], p2[1], samples).astype(int)
                xs, ys = np.clip(xs, 0, w-1), np.clip(ys, 0, h-1)
                if np.mean(skel_dilated[ys, xs]) > 0.5:
                    G.add_edge(i, j, weight=dist)
    return G
