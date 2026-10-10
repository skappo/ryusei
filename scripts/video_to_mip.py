import cv2
import numpy as np
import argparse
import os

def create_mip(input_path, output_path):
    # 1. Open the video file
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    mip_frame = None
    count = 0
    
    # Get total frames for progress reporting (optional)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames if total_frames > 0 else 'unknown'} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 2. Initialize or Update the Maximum Projection
        if mip_frame is None:
            # Start with the first frame
            mip_frame = frame.astype(np.uint8)
        else:
            # np.maximum compares each pixel/channel and keeps the highest value
            mip_frame = np.maximum(mip_frame, frame)
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} frames...")

    # 3. Save the result
    if mip_frame is not None:
        cv2.imwrite(output_path, mip_frame)
        print(f"Successfully saved MIP to: {output_path}")
    else:
        print("Error: No frames were found in the video.")

    cap.release()

if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Create a Maximum Intensity Projection (MIP) PNG from a video.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path where the output PNG will be saved")

    args = parser.parse_args()

    # Run the function
    create_mip(args.input, args.output)