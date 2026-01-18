import cv2
import time
import argparse
import networkx as nx

# Import modules
from utils.detection import detect_floor_and_obstacles
from graph_methods import (
    build_grid_graph,
    build_skeleton_graph,
    build_voronoi_graph,
    build_prm_graph,
    build_medial_axis_graph,
    build_corner_aisle_graph,
    build_visibility_graph
)
from algorithms import (
    dijkstra_shortest_path,
    astar_shortest_path,
    tsp_nearest_neighbor,
    steiner_tree_mst
)
from point_selector import PointSelector
from visualization import visualize_results


def build_graph(method, floor_mask, obstacle_mask=None, grid_spacing=25):
    """Build navigation graph using specified method."""
    if method == 'grid':
        return build_grid_graph(floor_mask, grid_spacing=grid_spacing)
    elif method == 'skeleton':
        return build_skeleton_graph(floor_mask)
    elif method == 'voronoi':
        return build_voronoi_graph(floor_mask, obstacle_mask)
    elif method == 'prm':
        return build_prm_graph(floor_mask, n_samples=100)
    elif method == 'medial_axis':
        return build_medial_axis_graph(floor_mask)
    elif method == 'corner_aisle':
        return build_corner_aisle_graph(floor_mask)
    elif method == 'visibility':
        return build_visibility_graph(floor_mask, obstacle_mask)
    else:
        print(f"Unknown method '{method}', using grid")
        return build_grid_graph(floor_mask, grid_spacing=grid_spacing)


def run_algorithms(G, selected_nodes):
    """Run all pathfinding algorithms and return results."""
    results = {}
    
    # Dijkstra
    start_time = time.time()
    path, dist = dijkstra_shortest_path(G, selected_nodes['start'], selected_nodes['end'])
    results['dijkstra'] = {'path': path, 'dist': dist, 'time': time.time() - start_time}
    
    # A*
    start_time = time.time()
    path, dist = astar_shortest_path(G, selected_nodes['start'], selected_nodes['end'])
    results['astar'] = {'path': path, 'dist': dist, 'time': time.time() - start_time}
    
    # TSP and Steiner need terminals
    terminals = [selected_nodes['start']] + selected_nodes['pickups'] + [selected_nodes['end']]
    
    if len(terminals) >= 2:
        # TSP
        start_time = time.time()
        route, dist, dist_matrix, path_dict = tsp_nearest_neighbor(G, terminals)
        results['tsp'] = {
            'route': route, 'dist': dist, 'time': time.time() - start_time,
            'path_dict': path_dict, 'terminals': terminals
        }
        
        # Steiner Tree
        start_time = time.time()
        steiner_tree, steiner_nodes, weight, path_dict = steiner_tree_mst(G, terminals)
        results['steiner'] = {
            'tree': steiner_tree, 'nodes': steiner_nodes, 'weight': weight,
            'time': time.time() - start_time, 'terminals': terminals
        }
    else:
        results['tsp'] = {'route': [], 'dist': 0, 'time': 0, 'path_dict': {}, 'terminals': terminals}
        results['steiner'] = {'tree': nx.Graph(), 'nodes': set(), 'weight': 0, 'time': 0, 'terminals': terminals}
    
    return results


def main(image_path, method='grid', grid_spacing=25):
    """Main pipeline."""
    print("\n" + "="*60)
    print("     AUTOMATIC WAREHOUSE PATHFINDING SYSTEM")
    print("="*60)
    
    # 1. Load image
    print(f"\n[1] Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    print(f"    Size: {img.shape[1]} x {img.shape[0]}")
    
    # 2. Detect floor/obstacles
    print("\n[2] Detecting floor and obstacles...")
    floor_mask = detect_floor_and_obstacles(img)
    obstacle_mask = cv2.bitwise_not(floor_mask)
    print("    Done!")
    
    # 3. Build graph
    print(f"\n[3] Building graph ({method.upper()} method)...")
    G = build_graph(method, floor_mask, obstacle_mask, grid_spacing)
    print(f"    Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # 4. Interactive point selection
    print("\n[4] Opening point selector...")
    selector = PointSelector(img, G)
    selected_nodes = selector.run()
    
    print("\n[5] Selected points:")
    print(f"    Start: {selected_nodes['start']}")
    print(f"    End: {selected_nodes['end']}")
    print(f"    Pickups: {selected_nodes['pickups']}")
    
    if selected_nodes['start'] is None or selected_nodes['end'] is None:
        print("\nError: Must select START and END points!")
        return
    
    # 5. Run algorithms
    print("\n[6] Running algorithms...")
    results = run_algorithms(G, selected_nodes)
    
    print(f"    Dijkstra: dist={results['dijkstra']['dist']:.1f}, time={results['dijkstra']['time']*1000:.2f}ms")
    print(f"    A*: dist={results['astar']['dist']:.1f}, time={results['astar']['time']*1000:.2f}ms")
    print(f"    TSP: dist={results['tsp']['dist']:.1f}, time={results['tsp']['time']*1000:.2f}ms")
    print(f"    Steiner: weight={results['steiner']['weight']:.1f}, time={results['steiner']['time']*1000:.2f}ms")
    
    # 6. Visualize
    print("\n[7] Generating visualization...")
    visualize_results(img, G, selected_nodes, results)
    
    print("\n" + "="*60)
    print("                    COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automatic Warehouse Pathfinding System')
    parser.add_argument('image_path', help='Path to warehouse floor plan image')
    parser.add_argument('--method', '-m', default='grid',
                        choices=['grid', 'skeleton', 'voronoi', 'prm', 'medial_axis', 'corner_aisle', 'visibility'],
                        help='Graph extraction method (default: grid)')
    parser.add_argument('--spacing', '-s', type=int, default=25,
                        help='Grid spacing for grid method (default: 25)')
    
    args = parser.parse_args()
    main(args.image_path, method=args.method, grid_spacing=args.spacing)
