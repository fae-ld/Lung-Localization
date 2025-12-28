"""
Morphological operations for image segmentation.
"""
import numpy as np
from scipy import ndimage as ndi


def keep_only_outer_land(mask):
    """
    Keep only land components that touch the image border.
    
    Parameters
    ----------
    mask : ndarray
        Binary mask where 0=water, 1=land.
    
    Returns
    -------
    outer_mask : ndarray
        Binary mask containing only outer land components.
    """
    land = (mask == 1)
    labels, n = ndi.label(land)  # label connected components
    
    H, W = mask.shape
    boundary_coords = [
        (0, slice(None)),
        (H-1, slice(None)),
        (slice(None), 0),
        (slice(None), W-1)
    ]
    
    # Cari label mana yang muncul di boundary
    boundary_labels = set()
    for r, c in boundary_coords:
        boundary_labels.update(np.unique(labels[r, c]))
    
    # Label 0 = background → di-skip
    boundary_labels.discard(0)
    
    # Bikin mask baru berisi hanya outer land
    keep = np.isin(labels, list(boundary_labels))
    return keep.astype(np.uint8)