"""
Test script to verify image path and OpenCV installation
"""

import os
import cv2

def test_opencv():
    """Test if OpenCV is working."""
    print("Testing OpenCV installation...")
    try:
        print(f"OpenCV version: {cv2.__version__}")
        print("✓ OpenCV is working!")
        return True
    except Exception as e:
        print(f"✗ OpenCV error: {e}")
        return False

def test_image_path(image_path):
    """Test if image path is valid."""
    print(f"\nTesting image path: {image_path}")
    
    # Remove quotes if present
    if image_path.startswith('"') and image_path.endswith('"'):
        image_path = image_path[1:-1]
    elif image_path.startswith("'") and image_path.endswith("'"):
        image_path = image_path[1:-1]
    
    print(f"Cleaned path: {image_path}")
    print(f"File exists: {os.path.exists(image_path)}")
    
    if os.path.exists(image_path):
        try:
            # Try to load the image
            image = cv2.imread(image_path)
            if image is not None:
                print(f"✓ Image loaded successfully!")
                print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
                print(f"  Image channels: {image.shape[2] if len(image.shape) > 2 else 1}")
                return True
            else:
                print("✗ Could not load image (file may be corrupted)")
                return False
        except Exception as e:
            print(f"✗ Error loading image: {e}")
            return False
    else:
        print("✗ File does not exist")
        return False

def main():
    """Main test function."""
    print("Image Path and OpenCV Test")
    print("=" * 40)
    
    # Test OpenCV
    if not test_opencv():
        print("\nPlease install OpenCV first:")
        print("pip install opencv-python")
        return
    
    # Test image path
    image_path = input("\nEnter image path to test: ").strip()
    
    if test_image_path(image_path):
        print("\n✓ Everything is working! You can now run:")
        print("python face_detection_simple.py")
    else:
        print("\n✗ Please check your image path and try again.")
        print("\nTips:")
        print("- Don't use quotes around the path")
        print("- Use forward slashes (/) instead of backslashes (\\)")
        print("- Make sure the file exists and is a valid image")

if __name__ == "__main__":
    main()
