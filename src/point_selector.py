import cv2
import numpy as np


class PointSelector:
    def __init__(self, img, graph):
        """
        Initialize point selector.
        
        Args:
            img: OpenCV image (BGR)
            graph: NetworkX graph with 'pos' attribute on nodes
        """
        if isinstance(img, str):
            self.img_original = cv2.imread(img)
        else:
            self.img_original = img.copy()
        
        self.img = self.img_original.copy()
        self.graph = graph
        
        # Build position lookup
        self.node_positions = {}
        if self.graph:
            for n in self.graph.nodes():
                self.node_positions[n] = self.graph.nodes[n]['pos']
        
        # Selected points
        self.selected_nodes = {'start': None, 'end': None, 'pickups': []}
        self.positions = {'start': None, 'end': None, 'pickups': []}
        self.current_mode = 'start'
        
        self.window_name = 'Point Selector - S:Start E:End P:Pickup R:Reset Q:Quit'
    
    def find_nearest_node(self, x, y):
        """Find nearest graph node to click position."""
        if not self.node_positions:
            return None, (x, y)
        
        min_dist = float('inf')
        nearest_node = None
        nearest_pos = (x, y)
        
        for node, pos in self.node_positions.items():
            dist = np.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                nearest_pos = pos
        
        return nearest_node, nearest_pos
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            node, pos = self.find_nearest_node(x, y)
            
            if node is None:
                print(f"No graph - using raw position ({x}, {y})")
                pos = (x, y)
            else:
                print(f"Clicked ({x},{y}) -> Snapped to node {node} at {pos}")
            
            if self.current_mode == 'start':
                self.positions['start'] = pos
                self.selected_nodes['start'] = node
                print(f">>> START set: node {node}")
            elif self.current_mode == 'end':
                self.positions['end'] = pos
                self.selected_nodes['end'] = node
                print(f">>> END set: node {node}")
            elif self.current_mode == 'pickup':
                if node not in self.selected_nodes['pickups']:
                    self.positions['pickups'].append(pos)
                    self.selected_nodes['pickups'].append(node)
                    print(f">>> PICKUP {len(self.selected_nodes['pickups'])} added: node {node}")
                else:
                    print(f"Node {node} already added!")
            
            self.redraw()
    
    def redraw(self):
        """Redraw the image with points and graph."""
        self.img = self.img_original.copy()
        
        # Draw graph edges (light gray)
        if self.graph:
            for u, v in self.graph.edges():
                p1 = self.graph.nodes[u]['pos']
                p2 = self.graph.nodes[v]['pos']
                cv2.line(self.img, p1, p2, (200, 200, 200), 1)
            
            # Draw graph nodes (small gray dots)
            for node in self.graph.nodes():
                pos = self.graph.nodes[node]['pos']
                cv2.circle(self.img, pos, 3, (150, 150, 150), -1)
        
        # Draw pickups (blue)
        for i, p in enumerate(self.positions['pickups']):
            cv2.circle(self.img, p, 12, (255, 0, 0), -1)
            cv2.circle(self.img, p, 12, (255, 255, 255), 2)
            cv2.putText(self.img, f'P{i+1}', (p[0]+15, p[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw start (green)
        if self.positions['start']:
            p = self.positions['start']
            cv2.circle(self.img, p, 14, (0, 255, 0), -1)
            cv2.circle(self.img, p, 14, (255, 255, 255), 2)
            cv2.putText(self.img, 'START', (p[0]+18, p[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw end (red)
        if self.positions['end']:
            p = self.positions['end']
            cv2.circle(self.img, p, 14, (0, 0, 255), -1)
            cv2.circle(self.img, p, 14, (255, 255, 255), 2)
            cv2.putText(self.img, 'END', (p[0]+18, p[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show mode
        mode_colors = {'start': (0, 255, 0), 'end': (0, 0, 255), 'pickup': (255, 0, 0)}
        cv2.putText(self.img, f"Mode: {self.current_mode.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_colors[self.current_mode], 2)
        cv2.putText(self.img, "S:Start E:End P:Pickup R:Reset Q:Quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(self.window_name, self.img)
    
    def run(self):
        """Run the interactive selector. Returns selected node IDs."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*50)
        print("       POINT SELECTOR")
        print("="*50)
        print("Click to select points (snaps to graph)")
        print("  S = Start mode    E = End mode")
        print("  P = Pickup mode   R = Reset")
        print("  Q = Quit and continue")
        print("="*50 + "\n")
        
        self.redraw()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.current_mode = 'start'
                print("Mode: START")
                self.redraw()
            elif key == ord('e'):
                self.current_mode = 'end'
                print("Mode: END")
                self.redraw()
            elif key == ord('p'):
                self.current_mode = 'pickup'
                print("Mode: PICKUP")
                self.redraw()
            elif key == ord('r'):
                self.selected_nodes = {'start': None, 'end': None, 'pickups': []}
                self.positions = {'start': None, 'end': None, 'pickups': []}
                self.current_mode = 'start'
                print(">>> Reset all points")
                self.redraw()
            elif key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()
        return self.selected_nodes
