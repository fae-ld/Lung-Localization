"""
Rib pattern detection and scoring algorithms for chest X-ray analysis.
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import label
from typing import Tuple, Dict, Any


def rib_peak_score(s_y, min_prom=0.05):
    """
    Calculate rib score based on peak detection and regularity.
    
    Parameters
    ----------
    s_y : ndarray
        1D signal array (e.g., vertical projection).
    min_prom : float
        Minimum prominence for peak detection.
    
    Returns
    -------
    score : float
        Regularity score (higher = more regular peaks).
    peaks : ndarray
        Indices of detected peaks.
    delta : ndarray or None
        Differences between consecutive peaks.
    """
    peaks, _ = find_peaks(s_y, prominence=min_prom)
    if len(peaks) < 3:
        return 0.0, peaks, None
    
    delta = np.diff(peaks)
    sigma = np.std(delta)
    score = len(peaks) / (sigma + 1e-6)
    return score, peaks, delta


def rib_autocorr_score(s_y, tau_min=5, tau_max=80):
    """
    Calculate rib score using autocorrelation.
    
    Parameters
    ----------
    s_y : ndarray
        1D signal array.
    tau_min : int
        Minimum lag to consider.
    tau_max : int
        Maximum lag to consider.
    
    Returns
    -------
    score : float
        Autocorrelation score.
    r : ndarray
        Normalized autocorrelation function.
    """
    s = s_y - s_y.mean()
    r = np.correlate(s, s, mode="full")
    r = r[r.size // 2:]        # positive lags
    r /= (r[0] + 1e-6)         # normalize
    
    tau_max = min(tau_max, len(r) - 1)
    score = r[tau_min:tau_max].sum()
    return score, r


def extract_patch(img, comp_mask):
    """
    Extract the bounding box region of a connected component from img.
    
    Parameters
    ----------
    img : ndarray
        Source image.
    comp_mask : ndarray
        Binary mask of the component.
    
    Returns
    -------
    patch : ndarray or None
        Extracted rectangular patch, or None if empty.
    """
    ys, xs = np.where(comp_mask > 0)
    if len(xs) == 0:
        return None
    
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    
    patch = img[y1:y2+1, x1:x2+1]
    return patch


def rib_pattern_detection(img, mask, core_mask, method="autocorrelation"):
    """
    Detect rib patterns in chest X-ray using 2D periodicity analysis.
    
    Parameters
    ----------
    img : ndarray
        Raw CXR image (2D numpy array).
    mask : ndarray
        Body mask (kept for future logic).
    core_mask : ndarray
        Mask expected to contain 2 largest lung components.
    method : str
        Detection method: 'autocorrelation', 'fft', or 'autocorrelation_entropy'.
    
    Returns
    -------
    results : dict
        Dictionary with scores for each connected lung component.
    """
    from .periodicity import autocorrelation_2d, fft_periodicity_2d, autocorrelation_entropy
    
    # Label connected components in core_mask
    labeled, n_comp = label(core_mask > 0)
    
    results = {}
    
    for comp_id in range(1, n_comp+1):
        comp_mask = (labeled == comp_id)
        patch = extract_patch(img, comp_mask)
        
        if patch is None or patch.size == 0:
            continue
        
        # Apply mask within patch
        masked_patch = patch.copy()
        pm = extract_patch(comp_mask.astype(np.uint8), comp_mask)
        masked_patch[pm == 0] = 0
        
        # Choose method
        if method == "autocorrelation":
            score, aux = autocorrelation_2d(masked_patch)
        elif method == "fft":
            score, aux = fft_periodicity_2d(masked_patch)
        elif method == "autocorrelation_entropy":
            score, aux = autocorrelation_entropy(masked_patch)
        else:
            raise ValueError("Method must be 'autocorrelation', 'fft', or 'autocorrelation_entropy'")
        
        results[f"component_{comp_id}"] = {
            "score": float(score),
            "patch": masked_patch,
            "aux_map": aux,
        }
    
    return results