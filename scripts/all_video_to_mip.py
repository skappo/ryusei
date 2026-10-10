import cv2
import numpy as np
import argparse
import os
import glob

def create_mip(input_path, output_path):
    # 1. Open the video file
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return False

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
    success = False
    if mip_frame is not None:
        cv2.imwrite(output_path, mip_frame)
        print(f"Successfully saved MIP to: {output_path}")
        success = True
    else:
        print("Error: No frames were found in the video.")

    cap.release()
    return success


def process_directory(input_dir, output_dir):
    # Find all mp4 files in the input directory (case-insensitive extension)
    mp4_files = sorted(
        f for f in glob.glob(os.path.join(input_dir, "*"))
        if f.lower().endswith(".mp4")
    )

    if not mp4_files:
        print(f"No .mp4 files found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(mp4_files)} mp4 file(s) in {input_dir}")

    index = 1
    for video_path in mp4_files:
        output_path = os.path.join(output_dir, f"{index}.png")
        print(f"\n=== [{index}/{len(mp4_files)}] {os.path.basename(video_path)} -> {output_path} ===")
        ok = create_mip(video_path, output_path)
        if ok:
            index += 1
        else:
            print(f"Skipped {video_path} (no output generated, numbering not incremented)")


if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(
        description="Create Maximum Intensity Projection (MIP) PNGs from all mp4 videos in a directory."
    )
    parser.add_argument("input_dir", help="Path to the directory containing input .mp4 files")
    parser.add_argument("output_dir", help="Directory where output PNGs (1.png, 2.png, ...) will be saved")

    args = parser.parse_args()

    # Run the function
    process_directory(args.input_dir, args.output_dir)
