#!/usr/bin/env python3
"""
Quick run script for the warehouse pathfinding system
Usage: python run.py <image_path>
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 50)
        print("  Automatic Warehouse Pathfinding System")
        print("=" * 50)
        print("\nUsage: python run.py <image_path>")
        print("\nExample:")
        print("  python run.py examples/warehouse.jpg")
        print("\nControls:")
        print("  S - Set START point")
        print("  E - Set END point")
        print("  P - Add PICKUP point")
        print("  R - Reset all")
        print("  Q - Quit and compute paths")
        sys.exit(1)
    
    main(sys.argv[1])
