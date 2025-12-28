Yet another repository for Digital Image Processing course project.

```
project/
├── src/
│ ├── **init**.py
│ ├── preprocessing/
│ │ ├── **init**.py
│ │ ├── body_extraction.py # extract_body_mask, fill_holes_continuous
│ │ └── enhancement.py # CLAHE, smoothing functions
│ ├── segmentation/
│ │ ├── **init**.py
│ │ ├── distance_based.py # calculate_8_way_distances, apply_distance_thresholding
│ │ ├── watershed.py # watershed_core
│ │ └── morphology.py # keep_only_outer_land, closing, opening helpers
│ ├── analysis/
│ │ ├── **init**.py
│ │ ├── rib_detection.py # rib_peak_score, rib_autocorr_score, rib_pattern_detection
│ │ └── periodicity.py # autocorrelation_2d, fft_periodicity_2d, autocorrelation_entropy
│ └── utils/
│ ├── **init**.py
│ ├── image_utils.py # extract_patch, create_complex_mask
│ └── visualization.py # visualize_thresholding
├── tests/
│ └── **init**.py
├── examples/
│ └── demo.py
└── requirements.txt
```
