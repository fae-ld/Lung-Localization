import numpy as np

def calculate_metrics(pred_box, gt_box):
    """
    Menghitung IoU, Euclidean Distance, DIoU, dan Area Ratio (Scale Factor).
    box format: [x1, y1, x2, y2]
    """
    x1_p, y1_p, x2_p, y2_p = pred_box
    x1_g, y1_g, x2_g, y2_g = gt_box

    # Hitung Luas masing-masing
    area_p = (x2_p - x1_p) * (y2_p - y1_p)
    area_g = (x2_g - x1_g) * (y2_g - y1_g)

    # 1. Intersection over Union (IoU)
    ix1, iy1 = max(x1_p, x1_g), max(y1_p, y1_g)
    ix2, iy2 = min(x2_p, x2_g), min(y2_p, y2_g)
    
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union_area = area_p + area_g - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    # 2. Centroid Distance (Euclidean)
    cp_x, cp_y = (x1_p + x2_p) / 2, (y1_p + y2_p) / 2
    cg_x, cg_y = (x1_g + x2_g) / 2, (y1_g + y2_g) / 2
    dist = np.sqrt((cp_x - cg_x)**2 + (cp_y - cg_y)**2)

    # 3. Distance-IoU (DIoU)
    rho2 = (cp_x - cg_x)**2 + (cp_y - cg_y)**2
    cx1, cy1 = min(x1_p, x1_g), min(y1_p, y1_g)
    cx2, cy2 = max(x2_p, x2_g), max(y2_p, y2_g)
    c2 = (cx2 - cx1)**2 + (cy2 - cy1)**2
    diou = iou - (rho2 / c2) if c2 > 0 else iou

    # 4. Area Ratio / Scale Factor (Ide Kamu)
    # > 1 berarti Prediksi kebesaran, < 1 berarti Prediksi kekecilan
    area_ratio = area_p / area_g if area_g > 0 else 0

    return {
        "iou": float(iou),
        "distance": float(dist),
        "diou": float(diou),
        "scale": float(area_ratio)
    }