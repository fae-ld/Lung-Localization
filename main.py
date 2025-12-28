import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.morphology import reconstruction
from skimage import filters, exposure
from skimage.transform import resize
from skimage.morphology import closing, square, opening, dilation, thin
from scipy.ndimage import median_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
import cv2
from typing import List, Tuple, Dict
from scipy.signal import find_peaks
from scipy.ndimage import label
from numpy.fft import fft2, fftshift
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d

def fill_holes_continuous(mask):
    # invert mask supaya hole muncul sebagai basins
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

def calculate_8_way_distances(mask: np.ndarray) -> Tuple[Dict[Tuple[int, int], List[float]], np.ndarray]:
    """
    Menghitung jarak rata-rata 8-arah (heatmap) dari setiap piksel air (0) ke daratan (1) atau batas.
    Hanya mengembalikan peta jarak rata-rata untuk efisiensi.
    """
    H, W = mask.shape
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    # directions = [(0, -1), (0, 1)]

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

def create_complex_mask(size=256):
    """Membuat mask contoh dengan Danau, Pulau, dan Sungai."""
    mask = np.ones((size, size), dtype=np.uint8)
    cv2.circle(mask, (int(size*0.5), int(size*0.5)), 40, 0, -1)
    cv2.circle(mask, (int(size*0.5) + 15, int(size*0.5) - 15), 5, 1, -1)
    cv2.rectangle(mask, (80, 5), (95, 80), 0, -1)
    return mask

# --- FUNGSI UTAMA: THRESHOLDING JARAK ---
# -------------------------
# 1. Watershed Core
# -------------------------
def watershed_core(avg):
    coords = peak_local_max(avg, min_distance=3)
    markers = np.zeros_like(avg, dtype=np.int32)

    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1

    ws = watershed(-avg, markers, mask=(avg > 0))
    return (ws > 0).astype(np.uint8)


# -------------------------
# 2. Otsu Automatic Segmentation
# -------------------------
def auto_distance_segmentation(avg_distance_map):
    t = threshold_otsu(avg_distance_map)
    return (avg_distance_map >= t).astype(np.uint8)


# -------------------------
# 3. Sigma-Based Core Mask
# -------------------------
def sigma_core_mask(avg_dist):
    m = np.mean(avg_dist)
    s = np.std(avg_dist)
    return (avg_dist >= m + 0.5 * s).astype(np.uint8)


# -------------------------
# 4. Main Thresholding Function (Parameter-free)
# -------------------------
def apply_distance_thresholding(avg_distance_map: np.ndarray, method="otsu") -> np.ndarray:
    """
    Membuat mask biner baru berdasarkan peta jarak rata-rata,
    tanpa parameter threshold dari user.

    method:
        - "otsu"   : Otsu thresholding otomatis
        - "sigma"  : mean + 0.5*std
        - "watershed" : region expansion berdasarkan puncak jarak
    """

    if method == "otsu":
        return auto_distance_segmentation(avg_distance_map)

    elif method == "sigma":
        return sigma_core_mask(avg_distance_map)

    elif method == "watershed":
        return watershed_core(avg_distance_map)

    else:
        raise ValueError("Method must be 'otsu', 'sigma', or 'watershed'.")


# -------------------------
# 5. Visualization
# -------------------------
def visualize_thresholding(mask: np.ndarray, avg_distance_map: np.ndarray, new_mask: np.ndarray, title="Auto Core Mask"):
    """Memvisualisasikan mask asli, heatmap, dan hasil segmentasi otomatis."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Mask Asli
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title("1. Original Mask (1=Land, 0=Water)")
    axes[0].axis('off')

    # 2. Heatmap Distance
    heatmap_masked = np.ma.masked_where(mask == 1, avg_distance_map)
    im = axes[1].imshow(heatmap_masked, cmap='plasma')
    axes[1].set_title("2. Average 8-Way Distance Heatmap")
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04).set_label('Distance')

    # 3. Mask Core Baru
    axes[2].imshow(new_mask, cmap='Blues')
    axes[2].set_title(f"3. {title}")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def keep_only_outer_land(mask):
    """
    mask: 0 = water, 1 = land
    return: mask baru yang hanya menyisakan daratan yang nyentuh border
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

    # cari label mana yang muncul di boundary
    boundary_labels = set()
    for r, c in boundary_coords:
        boundary_labels.update(np.unique(labels[r, c]))

    # label 0 = background → di-skip
    boundary_labels.discard(0)

    # bikin mask baru berisi hanya outer land
    keep = np.isin(labels, list(boundary_labels))
    return keep.astype(np.uint8)

def rib_peak_score(s_y, min_prom=0.05):
    peaks, _ = find_peaks(s_y, prominence=min_prom)
    if len(peaks) < 3:
        return 0.0, peaks, None

    delta = np.diff(peaks)
    sigma = np.std(delta)
    score = len(peaks) / (sigma + 1e-6)
    return score, peaks, delta

def rib_autocorr_score(s_y, tau_min=5, tau_max=80):
    s = s_y - s_y.mean()
    r = np.correlate(s, s, mode="full")
    r = r[r.size // 2:]        # positive lags
    r /= (r[0] + 1e-6)         # normalize

    tau_max = min(tau_max, len(r) - 1)
    score = r[tau_min:tau_max].sum()
    return score, r

# =========================
# PARAMETERS
# =========================
MIN_AREA = 500
CANNY_T1 = 50
CANNY_T2 = 150

# =========================
# HELPER FUNCTIONS
# =========================
def rib_peak_score(s_y, min_prom=0.05):
    peaks, _ = find_peaks(s_y, prominence=min_prom)
    if len(peaks) < 3:
        return 0.0, peaks, None
    delta = np.diff(peaks)
    sigma = np.std(delta)
    score = len(peaks) / (sigma + 1e-6)
    return score, peaks, delta

def rib_autocorr_score(s_y, tau_min=5, tau_max=80):
    s = s_y - s_y.mean()
    r = np.correlate(s, s, mode="full")
    r = r[r.size // 2:]
    r /= (r[0] + 1e-6)
    tau_max = min(tau_max, len(r) - 1)
    score = r[tau_min:tau_max].sum()
    return score, r

def autocorrelation_2d(img):
    """
    Parameter-free 2D autocorrelation score:
    S = max(acorr(lag != 0)) / acorr(0)
    Returns: (score, ac_map)
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

def extract_patch(img, comp_mask):
    """
    Extract the bounding box region of a connected component from img.
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
    img        : raw CXR image (2D numpy)
    mask       : body mask (not fully used yet, but kept for future logic)
    core_mask  : mask expected to contain 2 largest lung components
    method     : 'autocorrelation' or 'fft'
    ----------------------------------------------------------
    Returns dict with scores for each connected lung component.
    """

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
        # Mask irrelevant pixels
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
            raise ValueError("Method must be 'autocorrelation' or 'fft'")

        results[f"component_{comp_id}"] = {
            "score": float(score),
            "patch": masked_patch,
            "aux_map": aux,
        }

    return results

def autocorrelation_entropy(img):
    """
    Parameter-free entropy-based periodicity score.
    Returns: (score, ac_map)
    """
    ac = autocorrelation_2d(img)

    # ambil magnitudo
    ac_abs = np.abs(ac)
    total = np.sum(ac_abs)

    if total == 0:
        return 0.0, ac

    # distribusi probabilitas
    P = ac_abs / total

    # hindari log(0)
    P_safe = P[P > 0]

    H = -np.sum(P_safe * np.log(P_safe))

    # Normalisasi entropy ke 0..1
    # Nilai max ≈ log(N) tapi kita normalisasi pakai ukuran peta AC
    H_max = np.log(P.size)
    score = float(H / H_max)
    
    return score, ac



## Main pipeline

# image_paths = ["/content/00000013_005.png", "/content/00000042_002.png", "/content/test.jpg"]
# image_paths = ["/content/test.jpg"] # Perfect
image_paths = ["/content/00000013_005.png"] # Hard
# image_paths = ["/content/00000042_002.png"] # Average
n_images = len(image_paths)
fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
OG_SHAPE = None

# Kalau hanya 1 gambar, axes jadi 1D → ubah ke 2D biar konsisten
if n_images == 1:
    axes = [axes]

for i, img_path in enumerate(image_paths):
    # Load & resize
    og_img = img_as_float(io.imread(img_path, as_gray=True))
    OG_SHAPE = og_img.shape
    img = resize(og_img, (256, 256), anti_aliasing=True)
    kernel = np.ones((9,9), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Ekstraksi body mask
    body, recon, mask = extract_body_mask(img)

    # Post-processing mask
    # mask = opening(mask, square(3))
    # mask = median_filter(mask, size=9)
    mask = keep_only_outer_land(mask)
    mask = closing(mask, square(5))

    # Plot ke subplot baris i
    axes[i][0].imshow(img, cmap='gray')
    axes[i][0].set_title("Original")
    axes[i][0].axis("off")

    axes[i][1].imshow(mask, cmap='gray')
    axes[i][1].set_title("Threshold Mask")
    axes[i][1].axis("off")

    axes[i][2].imshow(body, cmap='gray')
    axes[i][2].set_title("Body Mask")
    axes[i][2].axis("off")

plt.suptitle("Hasil Ekstraksi Semua Gambar", fontsize=16)
plt.tight_layout()
plt.show()

avg_distance_map = calculate_8_way_distances(mask)

core_mask = apply_distance_thresholding(avg_distance_map, method='sigma')

visualize_thresholding(mask, avg_distance_map, core_mask)

result = rib_pattern_detection(img, mask, core_mask, method='autocorrelation')

# visualize_boxes_with_scores(img, core_mask, result)

H0, W0 = OG_SHAPE  # (1000, 1024)

mask_hh = cv2.resize(
    core_mask.astype(np.uint8),
    (H0, H0),                      # (width, height)
    interpolation=cv2.INTER_NEAREST
)

core_mask_upscaled = cv2.resize(
    mask_hh,
    (W0, H0),
    interpolation=cv2.INTER_NEAREST
)

img_norm = og_img.astype(float)
img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    core_mask_upscaled,
    connectivity=8
)

MIN_AREA = 500

bboxes = []

for i in range(1, num_labels):  # skip background
    x, y, w, h, area = stats[i]

    if area < MIN_AREA:
        continue

    bboxes.append((x, y, w, h))

# =========================
# PREPROCESSING (GLOBAL)
# =========================
img = og_img.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min())
img = (img * 255).astype(np.uint8)

img_blur = cv2.GaussianBlur(img, (5, 5), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_blur)

edges = cv2.Canny(img_clahe, CANNY_T1, CANNY_T2)

grad_y = cv2.Sobel(img_clahe, cv2.CV_32F, dx=0, dy=1, ksize=3)
grad_y_pos = np.maximum(grad_y, 0)

# =========================
# CONNECTED COMPONENT LOOP
# =========================
results = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area < MIN_AREA:
        continue

    comp_mask = (labels == i)

    grad_crop = grad_y_pos[y:y+h, x:x+w]
    edge_crop = edges[y:y+h, x:x+w]
    mask_crop = comp_mask[y:y+h, x:x+w]

    gated_grad = grad_crop * (edge_crop > 0) * mask_crop
    if gated_grad.max() <= 0:
        continue

    grad_log = np.log1p(gated_grad)
    grad_log /= grad_log.max()

    s_y = grad_log.sum(axis=1)
    if s_y.max() > 0:
        s_y /= s_y.max()

    s_y_smooth = gaussian_filter1d(s_y, sigma=2)

    peak_score, peaks, _ = rib_peak_score(s_y_smooth)
    auto_score, r = rib_autocorr_score(s_y_smooth)
    hybrid_score = peak_score * auto_score

    corners = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]

    results.append({
        "label": i,
        "hybrid_score": hybrid_score,
        "peak_score": peak_score,
        "auto_score": auto_score,
        "bbox": (x, y, w, h),
        "corners": corners
    })

# =========================
# TAKE TOP-2 HYBRID SCORES
# =========================
results_sorted = sorted(results, key=lambda r: r["hybrid_score"], reverse=True)
top2 = results_sorted[:2]