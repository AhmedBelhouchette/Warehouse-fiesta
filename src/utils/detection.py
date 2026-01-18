"""
Floor and obstacle detection using computer vision
"""
import cv2
import numpy as np

def detect_floor_and_obstacles(img):
    """
    Detect floor (walkable) and obstacles from warehouse image.
    
    Args:
        img: BGR image (OpenCV format)
    
    Returns:
        floor_mask: Binary mask where WHITE=floor, BLACK=obstacles
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Edge detection methods
    edges_canny = cv2.Canny(gray, 30, 100)
    _, edges_dark = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    edges_color = (hsv[:, :, 1] > 15).astype(np.uint8) * 255
    dark_brown = (gray < 100).astype(np.uint8) * 255
    
    # Combine all edges
    edges = cv2.bitwise_or(edges_canny, edges_dark)
    edges = cv2.bitwise_or(edges, edges_color)
    edges = cv2.bitwise_or(edges, dark_brown)
    
    # Close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Floor = NOT obstacles (WHITE = floor)
    floor_mask = cv2.bitwise_not(edges_closed)
    
    return floor_mask
