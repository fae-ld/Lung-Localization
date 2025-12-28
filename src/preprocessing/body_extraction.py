"""
Body extraction and mask filling operations for chest X-ray images.
"""
import numpy as np
from skimage.morphology import reconstruction
from skimage import filters, exposure


def fill_holes_continuous(mask):
    """
    Fill holes in a binary mask using morphological reconstruction.
    
    Parameters
    ----------
    mask : ndarray
        Binary mask to fill holes in.
    
    Returns
    -------
    filled : ndarray
        Mask with holes filled.
    """
    # Invert mask supaya hole muncul sebagai basins
    seed = np.copy(mask)
    seed[1:-1, 1:-1] = mask.max()
    
    filled = reconstruction(seed, mask, method='erosion')
    return filled


def extract_body_mask(img, sigma=1.0, clip_limit=0.03, percentile=70):
    """
    Ekstraksi body mask dari citra X-ray menggunakan threshold, flood-fill,
    dan morphological reconstruction.
    
    Parameters
    ----------
    img : ndarray
        Input image (grayscale, float atau uint8).
    sigma : float, optional
        Gaussian blur sigma untuk smoothing (default=1.0).
    clip_limit : float, optional
        Clip limit untuk CLAHE (default=0.03).
    percentile : int, optional
        Persentil brightness untuk threshold (default=70).
    
    Returns
    -------
    body : ndarray (bool)
        Mask biner area tubuh (foreground).
    recon : ndarray (float)
        Hasil morphological reconstruction (background flood).
    mask : ndarray (float)
        Mask threshold awal.
    """
    # 1. Gaussian smoothing
    img_smooth = filters.gaussian(img, sigma=sigma)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    img_clahe = exposure.equalize_adapthist(img_smooth, clip_limit=clip_limit)
    
    # 3. Threshold kasar
    th = np.percentile(img_clahe, percentile)
    mask = (img_clahe > th).astype(float)
    
    # 4. Marker dari border (background flood-fill)
    marker = np.zeros_like(mask)
    marker[0, :] = mask[0, :]
    marker[-1, :] = mask[-1, :]
    marker[:, 0] = mask[:, 0]
    marker[:, -1] = mask[:, -1]
    marker = marker.astype(float)
    
    # 5. Morphological reconstruction
    recon = reconstruction(marker, mask, method='dilation')
    
    # 6. Body mask = mask minus reconstructed flood
    body = mask.astype(bool) & (~(recon > 0.5))
    
    return body, recon, mask