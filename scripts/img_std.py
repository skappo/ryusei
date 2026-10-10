import argparse
import sys
import numpy as np
from PIL import Image

def main():
    # 1. Setup argument parser
    parser = argparse.ArgumentParser(description="Calculate the Standard Deviation of an image.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--grayscale", action="store_true", help="Convert image to grayscale before calculating")
    
    args = parser.parse_args()

    try:
        # 2. Open the image
        with Image.open(args.image_path) as img:
            
            # Optional: Convert to grayscale if the flag is set
            # This is useful if you want to measure brightness contrast ignoring color info
            if args.grayscale:
                img = img.convert('L')
                mode_display = "Grayscale"
            else:
                mode_display = img.mode

            # 3. Convert image data to a NumPy array
            img_array = np.array(img)

            # 4. Calculate Standard Deviation
            # np.std() calculates the standard deviation of all values in the array
            std_dev = np.std(img_array)

            # 5. Output results
            print(f"Image: {args.image_path}")
            print(f"Mode:  {mode_display}")
            print(f"Standard Deviation: {std_dev:.4f}")

    except FileNotFoundError:
        print(f"Error: The file '{args.image_path}' could not be found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to process image. Reason: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()