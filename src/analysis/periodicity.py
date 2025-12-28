"""
Periodicity analysis using autocorrelation and FFT methods.
"""
import numpy as np
from scipy.signal import fftconvolve
from numpy.fft import fft2, fftshift
from typing import Tuple


def autocorrelation_2d(img):
    """
    Parameter-free 2D autocorrelation score.
    
    Score = max(acorr(lag != 0)) / acorr(0)
    
    Parameters
    ----------
    img : ndarray
        2D image array.
    
    Returns
    -------
    score : float
        Normalized autocorrelation score (0..1).
    ac : ndarray
        Full 2D autocorrelation map.
    """
    img = img.astype(np.float32)
    img = img - np.mean(img)
    
    # Full autocorrelation via FFT
    ac = fftconvolve(img, img[::-1, ::-1], mode='full')
    
    H, W = ac.shape
    center = (H // 2, W // 2)
    ac0 = ac[center]
    
    if ac0 == 0:
        return 0.0, ac  # no signal at all
    
    # Remove center peak
    ac_wo_center = ac.copy()
    ac_wo_center[center] = -np.inf
    
    peak = np.max(ac_wo_center)
    
    # Normalized score (0..1)
    score = float(np.clip(peak / ac0, 0, 1))
    
    return score, ac


def fft_periodicity_2d(img):
    """
    Compute periodicity via dominance of a non-zero frequency peak.
    
    Parameters
    ----------
    img : ndarray
        2D image array.
    
    Returns
    -------
    score : float
        Periodicity score (peak/total energy).
    F : ndarray
        2D FFT magnitude spectrum.
    """
    img = img.astype(np.float32)
    img = img - img.mean()
    
    # 2D FFT magnitude
    F = fftshift(np.abs(fft2(img)))
    
    # Zero-out DC component (center)
    cx, cy = F.shape[0]//2, F.shape[1]//2
    F[cx, cy] = 0
    
    # Score = strongest peak / total energy
    peak = np.max(F)
    total = np.sum(F) + 1e-8
    score = peak / total
    
    return score, F


def autocorrelation_entropy(img):
    """
    Parameter-free entropy-based periodicity score.
    
    Parameters
    ----------
    img : ndarray
        2D image array.
    
    Returns
    -------
    score : float
        Normalized entropy score (0..1).
    ac : ndarray
        2D autocorrelation map.
    """
    score_ac, ac = autocorrelation_2d(img)
    
    # Ambil magnitudo
    ac_abs = np.abs(ac)
    total = np.sum(ac_abs)
    
    if total == 0:
        return 0.0, ac
    
    # Distribusi probabilitas
    P = ac_abs / total
    
    # Hindari log(0)
    P_safe = P[P > 0]
    
    H = -np.sum(P_safe * np.log(P_safe))
    
    # Normalisasi entropy ke 0..1
    # Nilai max ≈ log(N) tapi kita normalisasi pakai ukuran peta AC
    H_max = np.log(P.size)
    score = float(H / H_max)
    
    return score, ac