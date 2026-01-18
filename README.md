# Automatic Warehouse Pathfinding System

**Project 4 - Computer Vision & Graph Optimization**
## Overview

This project implements an automatic warehouse navigation system that:
1. Loads warehouse floor plan images
2. Detects obstacles (racks, walls) using computer vision
3. Extracts navigation graphs using multiple methods
4. Computes optimal paths using various algorithms
5. Provides interactive point selection via OpenCV GUI

## Project Structure

```
warehouse_final/
├── src/
│   ├── main.py                 # Main integrated application
│   ├── point_selector.py       # Interactive OpenCV GUI
│   ├── graph_methods/          # 7 graph extraction methods
│   │   ├── grid.py            # Grid-based (BEST)
│   │   ├── skeleton.py        # Skeleton-based
│   │   ├── voronoi.py         # Voronoi diagram
│   │   ├── prm.py             # Probabilistic Roadmap
│   │   ├── medial_axis.py     # Distance transform
│   │   ├── corner_aisle.py    # Corner detection
│   │   └── visibility.py      # Visibility graph
│   ├── algorithms/             # Pathfinding algorithms
│   │   ├── shortest_path.py   # Dijkstra & A*
│   │   ├── tsp.py             # Traveling Salesman
│   │   └── steiner.py         # Steiner Tree
│   └── utils/                  # Helper functions
│       ├── helpers.py         # Graph utilities
│       └── detection.py       # Floor/obstacle detection
├── examples/                   # Example warehouse images
├── outputs/                    # Generated results
├── docs/                       # Documentation
└── README.md
```

## Requirements

```bash
pip install opencv-python numpy networkx matplotlib scipy scikit-image
```

## Usage

### Quick Start

```bash
python src/main.py examples/warehouse.jpg
```

### Interactive Controls

When the point selector opens:
- **S** - Set START mode (green)
- **E** - Set END mode (red)
- **P** - Set PICKUP mode (blue)
- **R** - Reset all points
- **Q** - Quit and run algorithms

### Graph Methods Comparison

| Method | Nodes | Edges | Best For |
|--------|-------|-------|----------|
| Grid | ~278 | ~746 | General use (BEST) |
| Skeleton | ~56 | ~76 | Narrow aisles |
| Voronoi | ~186 | ~2016 | Maximum clearance |
| PRM | ~100 | ~296 | Random exploration |
| Medial Axis | ~200 | ~1556 | Aisle centers |
| Corner+Aisle | ~79 | ~275 | Intersection focus |
| Visibility | ~4 | ~0 | Open spaces |

### Algorithms

| Algorithm | Purpose | Complexity |
|-----------|---------|------------|
| Dijkstra | Shortest path A→B | O(V²) |
| A* | Shortest path with heuristic | O(E) |
| TSP | Visit all points | NP-hard |
| Steiner Tree | Connect all points | NP-hard |

## Output

The system generates:
- `pathfinding_results.png` - Visualization of all algorithm results
- `selected_points.json` - Saved point selections

## Convention

- **WHITE = Floor** (walkable areas)
- **BLACK = Obstacles** (racks, walls)

```bash
cd src
python main.py warehouse.jpg                    # Grid (default)
python main.py warehouse.jpg --method skeleton  # Skeleton
python main.py warehouse.jpg -m voronoi         # Voronoi
python main.py warehouse.jpg -m prm             # PRM
python main.py warehouse.jpg -m grid -s 30      # Grid with 30px spacing
```