import os
import sys

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm
from skimage import io
from scipy.ndimage import label
from src.utils.metrics import calculate_metrics
from skimage.util import img_as_float
from skimage.transform import resize
from skimage.morphology import closing, square, opening

from src.preprocessing import extract_body_mask
from src.segmentation import calculate_8_way_distances, apply_distance_thresholding, keep_only_outer_land
from src.analysis import rib_pattern_detection
from src.utils import visualize_thresholding, visualize_boxes_with_scores

def batch_process_images(image_paths, resize_shape=(256, 256), 
                         morph_kernel_size=9, closing_size=5,
                         distance_method='sigma', rib_method='autocorrelation'):
    """
    Process multiple CXR images and display results.
    
    Parameters
    ----------
    image_paths : list of str
        List of image file paths.
    resize_shape : tuple
        Target size for resizing (height, width).
    morph_kernel_size : int
        Kernel size for morphological closing on input image.
    closing_size : int
        Size of closing operation on mask.
    distance_method : str
        Method for distance thresholding ('otsu', 'sigma', 'watershed').
    rib_method : str
        Method for rib detection ('autocorrelation', 'fft', 'autocorrelation_entropy').
    
    Returns
    -------
    results : list of dict
        Processing results for each image.
    """
    n_images = len(image_paths)
    
    results = []
    
    for i, img_path in enumerate(image_paths):
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{n_images}: {img_path}")
        print(f"{'='*60}")
        
        # Load & resize
        og_img = img_as_float(io.imread(img_path, as_gray=True))
        og_shape = og_img.shape
        print(f"Original shape: {og_shape}")
        
        img = resize(og_img, resize_shape, anti_aliasing=True)
        print(f"Resized to: {img.shape}")
        
        # Morphological closing
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # Ekstraksi body mask
        print("Extracting body mask...")
        body, recon, mask = extract_body_mask(img)
        
        # Post-processing mask
        print("Post-processing mask...")
        mask = keep_only_outer_land(mask)
        mask = closing(mask, square(closing_size))
        
        # Calculate distance map
        print("Calculating distance map...")
        min_distance_map = calculate_8_way_distances(mask)
        
        # Apply thresholding
        print(f"Applying distance thresholding (method: {distance_method})...")
        core_mask = apply_distance_thresholding(min_distance_map, method=distance_method)
        
        # Rib pattern detection
        print(f"Detecting rib patterns (method: {rib_method})...")
        result = rib_pattern_detection(img, mask, core_mask, method=rib_method)
        
        # Print scores
        print("\nRib detection scores:")
        for comp_name, data in result.items():
            print(f"  {comp_name}: score = {data['score']:.4f}")
        
        # Store results
        results.append({
            'image_path': img_path,
            'original_shape': og_shape,
            'processed_img': img,
            'mask': mask,
            'body_mask': body,
            'distance_map': min_distance_map,
            'core_mask': core_mask,
            'rib_results': result
        })
    
    return results

def get_dual_boxes_from_mask(mask, min_area=500):
    """
    Memisahkan mask paru menjadi dua bounding box (kiri dan kanan).
    
    Parameters:
    -----------
    mask : ndarray
        Mask biner (0 untuk background, >0 untuk paru).
    min_area : int
        Jumlah minimum pixel agar dianggap sebagai paru (menghindari noise).
        
    Returns:
    --------
    list of lists or None
        [[x1, y1, x2, y2]_left, [x1, y1, x2, y2]_right] 
        Mengembalikan None jika tidak ditemukan 2 komponen yang valid.
    """
    # 1. Labeling connected components
    # structure=np.ones((3,3)) memastikan pixel diagonal juga terhitung menyambung
    labeled, n_components = label(mask > 0)
    
    comp_list = []
    
    for i in range(1, n_components + 1):
        # Ambil koordinat pixel untuk komponen ke-i
        ys, xs = np.where(labeled == i)
        
        # Hitung luas (jumlah pixel)
        area = len(xs)
        
        # Filter jika objek terlalu kecil (noise)
        if area < min_area:
            continue
            
        # Tentukan Bounding Box [x1, y1, x2, y2]
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        
        # Simpan centroid X untuk sorting kiri-kanan
        centroid_x = xs.mean()
        
        comp_list.append({
            "box": box,
            "centroid_x": centroid_x
        })
    
    # 2. Validasi jumlah paru yang ditemukan
    # Idealnya harus 2. Jika lebih, ambil 2 yang paling besar (opsional).
    # Di sini kita sort berdasarkan centroid_x agar index 0 = Kiri, index 1 = Kanan.
    if len(comp_list) >= 2:
        # Urutkan berdasarkan posisi X (dari kiri ke kanan)
        sorted_comps = sorted(comp_list, key=lambda x: x['centroid_x'])
        
        # Kita ambil dua yang pertama (setelah di-sort X, biasanya paru kiri dan kanan)
        # Catatan: Di CXR, "Left Lung" pasien ada di sisi kanan gambar (RHS), 
        # tapi secara koordinat image, kita sebut saja Left-most dan Right-most.
        return [sorted_comps[0]['box'], sorted_comps[1]['box']]
    
    else:
        # Jika hanya ketemu 1 atau tidak ada sama sekali
        print(f"Warning: Hanya menemukan {len(comp_list)} komponen paru.")
        return None

def save_comparison_plot(img, pred_boxes, gt_boxes, save_path, filename, metrics_summary):
    """Menyimpan gambar dengan BBox dan ringkasan metrik di judul."""
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    ax = plt.gca()

    # Plot Prediksi (Lime)
    for i, box in enumerate(pred_boxes):
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             fill=False, edgecolor='lime', linewidth=3, label='Pred' if i==0 else "")
        ax.add_patch(rect)
    
    # Plot Ground Truth (Red Dashed)
    for i, box in enumerate(gt_boxes):
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             fill=False, edgecolor='red', linewidth=3, linestyle='--', label='GT' if i==0 else "")
        ax.add_patch(rect)

    title_str = f"{filename}\nAvg IoU: {metrics_summary['avg_iou']:.3f} | Avg Scale: {metrics_summary['avg_scale']:.3f}"
    plt.title(title_str, color='white', backgroundcolor='black')
    plt.legend()
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def run_comparison():
    # Setup Path
    img_dir = "data/cxr"
    mask_dir = "data/masks"
    loc_dir = "data/localizations"
    csv_path = "results_eval.csv"
    os.makedirs(loc_dir, exist_ok=True)

    # Cek progress existing
    processed_files = []
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        processed_files = df_old['filename'].tolist()

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in tqdm(img_files, desc="Batch Evaluating"):
        if filename in processed_files:
            continue # Skip jika sudah diproses
            
        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        if not os.path.exists(mask_path):
            print(f"Mask not found for {filename}, skipping...")
            continue

        # 1. Jalankan algoritma kamu
        with SuppressOutput():
            res_list = batch_process_images([img_path], resize_shape=(256, 256))
        
        if not res_list: continue
        res = res_list[0]
        
        # Ekstrak 2 box prediksi terbaik (kiri-kanan)
        labeled_pred, _ = label(res['core_mask'] > 0)
        sorted_comps = sorted(res['rib_results'].items(), key=lambda x: x[1]['score'], reverse=True)
        pred_info = []
        for name, _ in sorted_comps[:2]:
            cid = int(name.split('_')[1])
            ys, xs = np.where(labeled_pred == cid)
            if len(xs) > 0:
                pred_info.append({"box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "cx": xs.mean()})
        pred_boxes = [b['box'] for b in sorted(pred_info, key=lambda x: x['cx'])]

        # 2. Ekstrak Ground Truth dari mask
        gt_mask = io.imread(mask_path, as_gray=True)
        gt_mask_res = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        gt_boxes = get_dual_boxes_from_mask(gt_mask_res)

        # 3. Hitung & Simpan jika lengkap
        if gt_boxes and len(pred_boxes) == 2:
            L = calculate_metrics(pred_boxes[0], gt_boxes[0])
            R = calculate_metrics(pred_boxes[1], gt_boxes[1])
            
            summary = {
                "filename": filename,
                "L_iou": L['iou'], "L_diou": L['diou'], "L_scale": L['scale'], "L_dist": L['distance'],
                "R_iou": R['iou'], "R_diou": R['diou'], "R_scale": R['scale'], "R_dist": R['distance'],
                "avg_iou": (L['iou'] + R['iou']) / 2,
                "avg_scale": (L['scale'] + R['scale']) / 2
            }

            # Simpan Gambar
            save_comparison_plot(res['processed_img'], pred_boxes, gt_boxes, 
                os.path.join(loc_dir, f"eval_{filename}"), filename, summary)
            
            # Append ke CSV (metode ini aman dari crash)
            df_new = pd.DataFrame([summary])
            df_new.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
            
if __name__ == "__main__":
    run_comparison()