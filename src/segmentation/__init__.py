# ========================================
# src/segmentation/__init__.py
# ========================================
"""
Segmentation module for distance-based and morphological operations.
"""
from .distance_based import (
    calculate_8_way_distances,
    apply_distance_thresholding,
    watershed_core,
    auto_distance_segmentation,
    sigma_core_mask
)
from .morphology import keep_only_outer_land

__all__ = [
    'calculate_8_way_distances',
    'apply_distance_thresholding',
    'watershed_core',
    'auto_distance_segmentation',
    'sigma_core_mask',
    'keep_only_outer_land'
]