"""
Basic Face Detection using OpenCV Haar Cascade Classifier
This module provides fundamental face detection capabilities.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

class FaceDetector:
    """A class for detecting faces in images and video streams."""
    
    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize the face detector.
        
        Args:
            cascade_path: Path to Haar cascade XML file. If None, uses default OpenCV cascade.
        """
        if cascade_path is None:
            # Use OpenCV's built-in Haar cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Check if cascade loaded successfully
        if self.face_cascade.empty():
            raise ValueError("Could not load face cascade classifier")
    
    def detect_faces(self, image: np.ndarray, scale_factor: float = 1.1, 
                    min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size. Objects smaller than this are ignored
            
        Returns:
            List of rectangles (x, y, w, h) where faces are detected
        """
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return faces.tolist()
    
    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                  color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw rectangles around detected faces.
        
        Args:
            image: Input image
            faces: List of face rectangles (x, y, w, h)
            color: Color of the rectangle (B, G, R)
            thickness: Thickness of the rectangle lines
            
        Returns:
            Image with rectangles drawn around faces
        """
        result_image = image.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
        
        return result_image
    
    def detect_and_draw(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Detect faces and draw rectangles around them in one step.
        
        Args:
            image: Input image
            **kwargs: Additional arguments for detect_faces method
            
        Returns:
            Tuple of (image with drawn rectangles, list of face coordinates)
        """
        faces = self.detect_faces(image, **kwargs)
        result_image = self.draw_faces(image, faces)
        return result_image, faces

def main():
    """Example usage of the FaceDetector class."""
    # Initialize face detector
    detector = FaceDetector()
    
    # Load an image (replace with your image path)
    image_path = "sample_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Please provide a valid image path.")
        return
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}")
        return
    
    # Detect faces
    result_image, faces = detector.detect_and_draw(image)
    
    # Print results
    print(f"Found {len(faces)} face(s)")
    for i, (x, y, w, h) in enumerate(faces):
        print(f"Face {i+1}: x={x}, y={y}, width={w}, height={h}")
    
    # Display result
    cv2.imshow("Face Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    cv2.imwrite("face_detection_result.jpg", result_image)
    print("Result saved as 'face_detection_result.jpg'")

if __name__ == "__main__":
    main()
