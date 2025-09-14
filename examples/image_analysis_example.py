"""
Advanced Image Analysis Example
This example demonstrates comprehensive image analysis capabilities.
"""

import cv2
import os
from image_face_detection import ImageFaceDetector

def main():
    """Run advanced image analysis example."""
    print("Advanced Image Analysis Example")
    print("=" * 40)
    
    # Initialize detector
    detector = ImageFaceDetector()
    
    # Load an image
    image_path = "sample_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        print("Please provide a valid image path or create a sample image.")
        return
    
    # Analyze image
    print(f"Analyzing image: {image_path}")
    results = detector.analyze_image(image_path, save_result=True)
    
    # Print results
    print("\nAnalysis Results:")
    print(f"  Total faces: {results['total_faces']}")
    print(f"  Frontal faces: {results['frontal_faces']}")
    print(f"  Profile faces: {results['profile_faces']}")
    print(f"  Eyes detected: {results['eyes_detected']}")
    print(f"  Smiles detected: {results['smiles_detected']}")
    
    # Print face coordinates
    print("\nFace Coordinates:")
    for i, (x, y, w, h) in enumerate(results['face_coordinates']):
        print(f"  Face {i+1}: x={x}, y={y}, width={w}, height={h}")
    
    # Print eyes per face
    print("\nEyes per Face:")
    for i, eyes in enumerate(results['eyes_per_face']):
        print(f"  Face {i+1}: {len(eyes)} eyes")
        for j, (ex, ey, ew, eh) in enumerate(eyes):
            print(f"    Eye {j+1}: x={ex}, y={ey}, width={ew}, height={eh}")
    
    # Print smiles per face
    print("\nSmiles per Face:")
    for i, smiles in enumerate(results['smiles_per_face']):
        print(f"  Face {i+1}: {len(smiles)} smiles")
        for j, (sx, sy, sw, sh) in enumerate(smiles):
            print(f"    Smile {j+1}: x={sx}, y={sy}, width={sw}, height={sh}")
    
    print(f"\nResult saved as: analyzed_{os.path.basename(image_path)}")
    
    # Display result
    result_image = cv2.imread(f"analyzed_{os.path.basename(image_path)}")
    if result_image is not None:
        cv2.imshow("Advanced Image Analysis", result_image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def batch_processing_example():
    """Example of batch processing multiple images."""
    print("\nBatch Processing Example")
    print("=" * 40)
    
    # Create input directory if it doesn't exist
    input_dir = "input_images"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created directory: {input_dir}")
        print("Please add some images to this directory and run again.")
        return
    
    # Initialize detector
    detector = ImageFaceDetector()
    
    # Process all images in directory
    print(f"Processing images in: {input_dir}")
    results = detector.batch_process_images(input_dir, "output_images")
    
    print(f"Processed {len(results)} images")
    
    # Print summary
    total_faces = sum(r['total_faces'] for r in results)
    total_eyes = sum(r['eyes_detected'] for r in results)
    total_smiles = sum(r['smiles_detected'] for r in results)
    
    print(f"Total faces found: {total_faces}")
    print(f"Total eyes detected: {total_eyes}")
    print(f"Total smiles detected: {total_smiles}")

if __name__ == "__main__":
    main()
    batch_processing_example()
