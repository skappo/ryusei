import argparse
import sys
import numpy as np
import cv2
import os

def load_image(image_path):
    """Load image using OpenCV, return both BGR and grayscale versions."""
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' could not be found.")
        sys.exit(1)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Failed to open '{image_path}'.")
        sys.exit(1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_gray

def get_roi_coordinates(h, w):
    """Calculates centered ROI coordinates based on a single percentage."""
    roi_percent = 0.7
    margin_h = (1.0 - roi_percent) / 2
    margin_w = (1.0 - roi_percent) / 2

    y_start, y_end = int(h * margin_h), int(h * (1.0 - margin_h))
    x_start, x_end = int(w * margin_w), int(w * (1.0 - margin_w))
    
    return y_start, y_end, x_start, x_end, roi_percent

def count_features_in_roi(features, roi_coords):
    """Counts how many detected points fall within the ROI boundaries."""
    if features is None:
        return 0
    y_start, y_end, x_start, x_end, _ = roi_coords
    count = 0
    for f in features:
        x, y = f.ravel()
        if x_start <= x < x_end and y_start <= y < y_end:
            count += 1
    return count

def analyze_stddev(img_gray, roi_coords):
    y_start, y_end, x_start, x_end, p = roi_coords
    
    std_global = np.std(img_gray)
    roi_array = img_gray[y_start:y_end, x_start:x_end]
    std_roi = np.std(roi_array)

    print("\n--- PART A: Noise Analysis ---")
    print(f"  ROI Setting:     {p*100:.1f}% of dimensions ({ (p*p)*100:.1f}% area)")
    print(f"  Global StdDev:   {std_global:.4f}")
    print(f"  ROI StdDev:      {std_roi:.4f}")
    print(f"  Difference:      {std_global - std_roi:.4f}")

def detect_features(img_bgr, img_gray, output_path, roi_coords):
    y_start, y_end, x_start, x_end, p = roi_coords
    print("\n--- PART B: Feature Detection ---")

    # 1. Current Settings
    current_features = cv2.goodFeaturesToTrack(img_gray, 150, 0.01, 10)
    count_current = 0 if current_features is None else len(current_features)
    roi_count_current = count_features_in_roi(current_features, roi_coords)
    
    print(f"  1. Current Settings: {count_current} total / {roi_count_current} inside ROI")
    print("     (qualityLevel=0.01, maxCorners=150, No normalization)")
    
    # 2. New Settings
    contrast_img = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    new_features = cv2.goodFeaturesToTrack(contrast_img, 200, 0.001, 10)
    count_new = 0 if new_features is None else len(new_features)
    roi_count_new = count_features_in_roi(new_features, roi_coords)
    
    print(f"  2. New Settings:     {count_new} total / {roi_count_new} inside ROI")
    print("     (qualityLevel=0.001, maxCorners=200, With contrast normalization)")

    # Visual Output
    img_vis = img_bgr.copy()
    cv2.rectangle(img_vis, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)

    if current_features is not None:
        for i in current_features:
            cv2.circle(img_vis, tuple(i.ravel().astype(int)), 5, (0, 0, 255), -1)
    if new_features is not None:
        for i in new_features:
            cv2.circle(img_vis, tuple(i.ravel().astype(int)), 3, (0, 255, 0), 1)

    cv2.imwrite(output_path, img_vis)
    print(f"\n  Saved visual debug to: {output_path}")
    print("  Legend: RED DOTS = Current Settings | GREEN RINGS = New Settings")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--output", default="debug_features.jpg")
    args = parser.parse_args()

    img_bgr, img_gray = load_image(args.image_path)
    h, w = img_gray.shape

    roi_coords = get_roi_coordinates(h, w)

    print(f"ANALYZING: {args.image_path} ({w}x{h})")
    analyze_stddev(img_gray, roi_coords)
    detect_features(img_bgr, img_gray, args.output, roi_coords)

if __name__ == "__main__":
    main()