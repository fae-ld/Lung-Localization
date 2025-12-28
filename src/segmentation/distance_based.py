"""
Distance-based segmentation methods for chest X-ray image analysis.
"""
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from typing import Tuple, Dict, List


def calculate_8_way_distances(mask: np.ndarray) -> np.ndarray:
    """
    Menghitung jarak rata-rata 8-arah (heatmap) dari setiap piksel air (0) 
    ke daratan (1) atau batas.
    
    Parameters
    ----------
    mask : ndarray
        Binary mask where 0=water, 1=land.
    
    Returns
    -------
    distance_map : ndarray
        2D array containing minimum distances for each water pixel.
    """
    H, W = mask.shape
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    distance_map = np.zeros_like(mask, dtype=np.float32)
    
    for y in range(H):
        for x in range(W):
            if mask[y, x] == 0:
                distances = []
                for dy, dx in directions:
                    dist = 0
                    cy, cx = y, x
                    
                    while True:
                        cy += dy
                        cx += dx
                        dist += 1
                        
                        is_out_of_bounds = not (0 <= cy < H and 0 <= cx < W)
                        is_land = False
                        if not is_out_of_bounds and mask[cy, cx] == 1:
                            is_land = True
                        
                        if is_out_of_bounds or is_land:
                            # Jarak Euclidean: 1 untuk H/V, sqrt(2) untuk Diagonal
                            if abs(dx) == 1 and abs(dy) == 1:
                                final_dist = dist * np.sqrt(2)
                            else:
                                final_dist = dist
                            distances.append(final_dist)
                            break
                
                distance_map[y, x] = np.min(distances)
    
    return distance_map


def watershed_core(avg):
    """
    Apply watershed segmentation based on distance peaks.
    
    Parameters
    ----------
    avg : ndarray
        Average distance map.
    
    Returns
    -------
    mask : ndarray
        Binary segmented mask.
    """
    coords = peak_local_max(avg, min_distance=3)
    markers = np.zeros_like(avg, dtype=np.int32)
    
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1
    
    ws = watershed(-avg, markers, mask=(avg > 0))
    return (ws > 0).astype(np.uint8)


def auto_distance_segmentation(avg_distance_map):
    """
    Automatic segmentation using Otsu thresholding.
    
    Parameters
    ----------
    avg_distance_map : ndarray
        Distance map to threshold.
    
    Returns
    -------
    mask : ndarray
        Binary segmented mask.
    """
    t = threshold_otsu(avg_distance_map)
    return (avg_distance_map >= t).astype(np.uint8)


def sigma_core_mask(avg_dist):
    """
    Sigma-based core mask extraction.
    
    Parameters
    ----------
    avg_dist : ndarray
        Average distance map.
    
    Returns
    -------
    mask : ndarray
        Binary core mask.
    """
    m = np.mean(avg_dist)
    s = np.std(avg_dist)
    return (avg_dist >= m + 0.5 * s).astype(np.uint8)


def apply_distance_thresholding(avg_distance_map: np.ndarray, method="otsu") -> np.ndarray:
    """
    Membuat mask biner baru berdasarkan peta jarak rata-rata,
    tanpa parameter threshold dari user.
    
    Parameters
    ----------
    avg_distance_map : ndarray
        Average distance map.
    method : str
        Method to use: 'otsu', 'sigma', or 'watershed'.
    
    Returns
    -------
    mask : ndarray
        Binary segmented mask.
    
    Raises
    ------
    ValueError
        If method is not one of the supported methods.
    """
    if method == "otsu":
        return auto_distance_segmentation(avg_distance_map)
    
    elif method == "sigma":
        return sigma_core_mask(avg_distance_map)
    
    elif method == "watershed":
        return watershed_core(avg_distance_map)
    
    else:
        raise ValueError("Method must be 'otsu', 'sigma', or 'watershed'.")