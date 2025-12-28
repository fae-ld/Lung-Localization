"""
Image utility functions.
"""
import numpy as np
import cv2


def create_complex_mask(size=256):
    """
    Membuat mask contoh dengan Danau, Pulau, dan Sungai.
    
    Parameters
    ----------
    size : int
        Size of the square mask (default=256).
    
    Returns
    -------
    mask : ndarray
        Binary mask with complex structure.
    """
    mask = np.ones((size, size), dtype=np.uint8)
    
    # Danau (lake)
    cv2.circle(mask, (int(size*0.5), int(size*0.5)), 40, 0, -1)
    
    # Pulau (island)
    cv2.circle(mask, (int(size*0.5) + 15, int(size*0.5) - 15), 5, 1, -1)
    
    # Sungai (river)
    cv2.rectangle(mask, (80, 5), (95, 80), 0, -1)
    
    return mask