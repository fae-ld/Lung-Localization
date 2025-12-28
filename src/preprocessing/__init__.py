# ========================================
# src/preprocessing/__init__.py
# ========================================
"""
Preprocessing module for image enhancement and body extraction.
"""
from .body_extraction import fill_holes_continuous, extract_body_mask

__all__ = ['fill_holes_continuous', 'extract_body_mask']