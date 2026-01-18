"""
Compare all 7 graph extraction methods
Generates visual comparison for the report
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.detection import detect_floor_and_obstacles
from src.graph_methods.grid import build_grid_graph
from src.graph_methods.skeleton import build_skeleton_graph
from src.graph_methods.voronoi import build_voronoi_graph
from src.graph_methods.prm import build_prm_graph
from src.graph_methods.medial_axis import build_medial_axis_graph
from src.graph_methods.corner_aisle import build_corner_aisle_graph
from src.graph_methods.visibility import build_visibility_graph


def visualize_graph(img, G):
    vis = img.copy()
    for u, v in G.edges():
        p1 = G.nodes[u]['pos']
        p2 = G.nodes[v]['pos']
        cv2.line(vis, p1, p2, (0, 255, 0), 2)
    for node in G.nodes():
        pos = G.nodes[node]['pos']
        cv2.circle(vis, pos, 5, (255, 0, 0), -1)
    return vis


def main(image_path):
    print("Loading image...")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("Detecting floor/obstacles...")
    floor_mask = detect_floor_and_obstacles(img)
    obstacle_mask = cv2.bitwise_not(floor_mask)
    
    print("Building graphs...")
    methods = {
        '1. Grid (BEST)': build_grid_graph(floor_mask, 25),
        '2. Skeleton': build_skeleton_graph(floor_mask),
        '3. Voronoi': build_voronoi_graph(floor_mask, obstacle_mask),
        '4. PRM': build_prm_graph(floor_mask, 100),
        '5. Medial Axis': build_medial_axis_graph(floor_mask),
        '6. Corner+Aisle': build_corner_aisle_graph(floor_mask),
        '7. Visibility': build_visibility_graph(floor_mask, obstacle_mask),
    }
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    for idx, (name, G) in enumerate(methods.items()):
        row, col = (idx + 1) // 4, (idx + 1) % 4
        vis = visualize_graph(img_rgb.copy(), G)
        axes[row, col].imshow(vis)
        axes[row, col].set_title(f'{name}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
        axes[row, col].axis('off')
        print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    plt.tight_layout()
    plt.savefig('outputs/graph_methods_comparison.png', dpi=150)
    plt.show()
    print("\nSaved to outputs/graph_methods_comparison.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_methods.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
