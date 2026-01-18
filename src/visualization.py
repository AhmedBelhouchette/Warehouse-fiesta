"""
Visualization module for pathfinding results.
"""
import cv2
import matplotlib.pyplot as plt


def visualize_results(img, G, selected_nodes, results, output_path='pathfinding_results.png'):
    """Visualize all algorithm results in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    def draw_base(ax, title):
        """Draw base image with graph overlay."""
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for u, v in G.edges():
            p1 = G.nodes[u]['pos']
            p2 = G.nodes[v]['pos']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=0.5, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # 1. Dijkstra
    dijkstra = results['dijkstra']
    draw_base(axes[0,0], f"Dijkstra Shortest Path\nDist: {dijkstra['dist']:.1f}, Time: {dijkstra['time']*1000:.2f}ms")
    path = dijkstra['path']
    if path:
        for i in range(len(path)-1):
            p1, p2 = G.nodes[path[i]]['pos'], G.nodes[path[i+1]]['pos']
            axes[0,0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=3)
        axes[0,0].plot(*G.nodes[path[0]]['pos'], 'go', markersize=15)
        axes[0,0].plot(*G.nodes[path[-1]]['pos'], 'ro', markersize=15)
    
    # 2. A*
    astar = results['astar']
    draw_base(axes[0,1], f"A* Shortest Path\nDist: {astar['dist']:.1f}, Time: {astar['time']*1000:.2f}ms")
    path = astar['path']
    if path:
        for i in range(len(path)-1):
            p1, p2 = G.nodes[path[i]]['pos'], G.nodes[path[i+1]]['pos']
            axes[0,1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'm-', linewidth=3)
        axes[0,1].plot(*G.nodes[path[0]]['pos'], 'go', markersize=15)
        axes[0,1].plot(*G.nodes[path[-1]]['pos'], 'ro', markersize=15)
    
    # 3. TSP
    tsp = results['tsp']
    draw_base(axes[1,0], f"TSP (Nearest Neighbor)\nDist: {tsp['dist']:.1f}, Time: {tsp['time']*1000:.2f}ms")
    tsp_route = tsp['route']
    path_dict = tsp['path_dict']
    terminals = tsp['terminals']
    for i in range(len(tsp_route)):
        next_i = (i + 1) % len(tsp_route)
        path = path_dict.get((tsp_route[i], tsp_route[next_i])) or path_dict.get((tsp_route[next_i], tsp_route[i]))
        if path:
            for k in range(len(path)-1):
                p1, p2 = G.nodes[path[k]]['pos'], G.nodes[path[k+1]]['pos']
                axes[1,0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'orange', linewidth=3)
    for i, t in enumerate(terminals):
        pos = G.nodes[t]['pos']
        axes[1,0].plot(*pos, 'ro', markersize=12)
        axes[1,0].text(pos[0]+8, pos[1]+8, str(i+1), fontsize=10, color='white', fontweight='bold')
    
    # 4. Steiner Tree
    steiner = results['steiner']
    draw_base(axes[1,1], f"Steiner Tree (MST)\nWeight: {steiner['weight']:.1f}, Time: {steiner['time']*1000:.2f}ms")
    steiner_tree = steiner['tree']
    terminals = steiner['terminals']
    for u, v in steiner_tree.edges():
        p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
        axes[1,1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'c-', linewidth=3)
    for t in terminals:
        axes[1,1].plot(*G.nodes[t]['pos'], 'ro', markersize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Results saved to {output_path}")


def visualize_graph(img, G, title="Navigation Graph", output_path=None):
    """Visualize just the graph on the image."""
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Draw edges
    for u, v in G.edges():
        p1 = G.nodes[u]['pos']
        p2 = G.nodes[v]['pos']
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=1, alpha=0.7)
    
    # Draw nodes
    for node in G.nodes():
        pos = G.nodes[node]['pos']
        plt.plot(*pos, 'ro', markersize=4)
    
    plt.title(f"{title}\nNodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()
