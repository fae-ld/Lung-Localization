"""
Visualization utilities for image segmentation and analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label


def visualize_thresholding(mask: np.ndarray, avg_distance_map: np.ndarray, 
                          new_mask: np.ndarray, title="Auto Core Mask"):
    """
    Memvisualisasikan mask asli, heatmap, dan hasil segmentasi otomatis.
    
    Parameters
    ----------
    mask : ndarray
        Original binary mask (1=Land, 0=Water).
    avg_distance_map : ndarray
        Average distance heatmap.
    new_mask : ndarray
        Segmented core mask result.
    title : str
        Title for the third subplot.
    """
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
    fig.colorbar(im, ax=axes[1], orientation='vertical', 
                 fraction=0.046, pad=0.04).set_label('Distance')
    
    # 3. Mask Core Baru
    axes[2].imshow(new_mask, cmap='Blues')
    axes[2].set_title(f"3. {title}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_boxes_with_scores(img, core_mask, result):
    """
    Visualize bounding boxes with rib detection scores on CXR image.
    
    Parameters
    ----------
    img : ndarray
        Raw CXR image.
    core_mask : ndarray
        Connected component mask (lung candidates).
    result : dict
        Output dict from rib_pattern_detection().
    """
    labeled, n_comp = label(core_mask > 0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray")
    ax = plt.gca()
    
    # Loop setiap connected component
    for comp_id in range(1, n_comp + 1):
        comp_mask = (labeled == comp_id)
        ys, xs = np.where(comp_mask)
        
        if len(xs) == 0:
            continue
        
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        
        comp_name = f"component_{comp_id}"
        if comp_name not in result:
            continue
        
        score = result[comp_name]["score"]
        
        # Bikin bounding box
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor='lime',
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Taruh teks skor di atas box
        ax.text(
            x1,
            y1 - 5,
            f"{score:.4f}",
            color="yellow",
            fontsize=12,
            weight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=2)
        )
    
    plt.title("CXR with Component Bounding Boxes + Scores")
    plt.axis("off")
    plt.show()