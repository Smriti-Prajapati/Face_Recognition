"""
Basic Face Detection Example
This example demonstrates basic face detection functionality.
"""

import cv2
import os
from face_detection_basic import FaceDetector

def main():
    """Run basic face detection example."""
    print("Basic Face Detection Example")
    print("=" * 40)
    
    # Initialize face detector
    detector = FaceDetector()
    
    # Load an image
    image_path = "sample_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        print("Please provide a valid image path or create a sample image.")
        return
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}")
        return
    
    print(f"Loaded image: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Detect faces
    print("\nDetecting faces...")
    faces = detector.detect_faces(image)
    
    print(f"Found {len(faces)} face(s)")
    
    # Print face coordinates
    for i, (x, y, w, h) in enumerate(faces):
        print(f"  Face {i+1}: x={x}, y={y}, width={w}, height={h}")
    
    # Draw rectangles around faces
    result_image = detector.draw_faces(image, faces)
    
    # Save result
    output_path = "basic_detection_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"\nResult saved as: {output_path}")
    
    # Display result
    cv2.imshow("Basic Face Detection", result_image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
