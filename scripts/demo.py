"""
Batch processing demo for multiple CXR images.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.util import img_as_float
from skimage.transform import resize
from skimage.morphology import closing, square, opening

from src.preprocessing import extract_body_mask
from src.segmentation import calculate_8_way_distances, apply_distance_thresholding, keep_only_outer_land
from src.analysis import rib_pattern_detection
from src.utils import visualize_thresholding, visualize_boxes_with_scores

import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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
        
        import json
        
        with open("result.json", "w") as file:
            json.dump(result, file, cls=NumpyEncoder, indent=4)
        
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


def visualize_detailed_results(results, show_distance_map=True, show_boxes=True):
    """
    Visualize detailed results for each processed image.
    
    Parameters
    ----------
    results : list of dict
        Results from batch_process_images().
    show_distance_map : bool
        Whether to show distance map visualization.
    show_boxes : bool
        Whether to show bounding boxes with scores.
    """
    for i, res in enumerate(results):
        print(f"\n{'='*60}")
        print(f"Detailed visualization for image {i+1}")
        print(f"{'='*60}")
        
        if show_distance_map:
            print("Showing distance map...")
            visualize_thresholding(
                res['mask'], 
                res['distance_map'], 
                res['core_mask'],
                title=f"Core Mask - Image {i+1}"
            )
        
        if show_boxes:
            print("Showing bounding boxes with scores...")
            visualize_boxes_with_scores(
                res['processed_img'], 
                res['core_mask'], 
                res['rib_results']
            )


if __name__ == "__main__":
    print("=" * 60)
    print("Batch CXR Processing Demo")
    print("=" * 60)
    
    # Define image paths
    image_paths = [
        # "test.jpg",  
        # 'data/cxr/JPCLN001.png',
        'data\cxr\JPCLN002.png'
        # "path/to/your/image2.png",  # Average
    ]
    
    # Check if paths exist
    import os
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    
    if not valid_paths:
        print("\n⚠️  No valid image paths found!")
        print("Please update the image_paths list with your actual file paths.")
        print("\nExample:")
        print('  image_paths = ["C:/Users/yourname/images/cxr1.png"]')
        
        # Use demo with complex mask instead
        print("\n" + "="*60)
        print("Running demo with synthetic mask instead...")
        print("="*60)
        
        from src.utils import create_complex_mask
        
        mask = create_complex_mask(size=256)
        distance_map = calculate_8_way_distances(mask)
        core_mask = apply_distance_thresholding(distance_map, method='sigma')
        visualize_thresholding(mask, distance_map, core_mask)
        
    else:
        print(f"\nFound {len(valid_paths)} valid image(s)")
        
        # Process images
        results = batch_process_images(
            valid_paths,
            resize_shape=(256, 256),
            morph_kernel_size=9,
            closing_size=5,
            distance_method='sigma',
            rib_method='autocorrelation'
        )
        
        # Show detailed visualizations
        print("\n" + "="*60)
        print("Generating detailed visualizations...")
        print("="*60)
        
        visualize_detailed_results(
            results,
            show_distance_map=False,
            show_boxes=True
        )
        
        print("\n" + "="*60)
        print("Processing completed!")
        print("="*60)