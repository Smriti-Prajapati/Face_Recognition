"""
Advanced Image Face Detection Module
This module provides comprehensive face detection capabilities for static images.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from face_detection_basic import FaceDetector

class ImageFaceDetector(FaceDetector):
    """Extended face detector with advanced image processing capabilities."""
    
    def __init__(self, cascade_path: Optional[str] = None):
        """Initialize the image face detector."""
        super().__init__(cascade_path)
        
        # Additional cascade classifiers for different face angles
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    def detect_faces_multiple_angles(self, image: np.ndarray) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Detect faces from multiple angles (frontal and profile).
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with 'frontal' and 'profile' face detections
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect frontal faces
        frontal_faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Detect profile faces
        profile_faces = self.profile_cascade.detectMultiScale(gray, 1.1, 5)
        
        return {
            'frontal': frontal_faces.tolist(),
            'profile': profile_faces.tolist()
        }
    
    def detect_eyes(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int, int, int]]]:
        """
        Detect eyes within detected face regions.
        
        Args:
            image: Input image
            faces: List of face rectangles
            
        Returns:
            List of eye detections for each face
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes_per_face = []
        
        for (x, y, w, h) in faces:
            # Define region of interest (face area)
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            
            # Adjust coordinates to image coordinates
            adjusted_eyes = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
            eyes_per_face.append(adjusted_eyes)
        
        return eyes_per_face
    
    def detect_smiles(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int, int, int]]]:
        """
        Detect smiles within detected face regions.
        
        Args:
            image: Input image
            faces: List of face rectangles
            
        Returns:
            List of smile detections for each face
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smiles_per_face = []
        
        for (x, y, w, h) in faces:
            # Define region of interest (lower half of face for smile detection)
            roi_gray = gray[y + h//2:y+h, x:x+w]
            
            # Detect smiles in the face region
            smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.1, 5)
            
            # Adjust coordinates to image coordinates
            adjusted_smiles = [(x + sx, y + h//2 + sy, sw, sh) for (sx, sy, sw, sh) in smiles]
            smiles_per_face.append(adjusted_smiles)
        
        return smiles_per_face
    
    def draw_detections(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                       eyes_per_face: List[List[Tuple[int, int, int, int]]] = None,
                       smiles_per_face: List[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
        """
        Draw all detections on the image.
        
        Args:
            image: Input image
            faces: List of face rectangles
            eyes_per_face: List of eye detections per face
            smiles_per_face: List of smile detections per face
            
        Returns:
            Image with all detections drawn
        """
        result_image = image.copy()
        
        # Draw faces
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_image, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw eyes
        if eyes_per_face:
            for i, eyes in enumerate(eyes_per_face):
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(result_image, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                    cv2.putText(result_image, 'Eye', (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw smiles
        if smiles_per_face:
            for i, smiles in enumerate(smiles_per_face):
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(result_image, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                    cv2.putText(result_image, 'Smile', (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return result_image
    
    def analyze_image(self, image_path: str, save_result: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive face analysis on an image.
        
        Args:
            image_path: Path to the input image
            save_result: Whether to save the result image
            
        Returns:
            Dictionary containing analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect faces from multiple angles
        face_detections = self.detect_faces_multiple_angles(image)
        all_faces = face_detections['frontal'] + face_detections['profile']
        
        # Detect eyes and smiles
        eyes_per_face = self.detect_eyes(image, all_faces)
        smiles_per_face = self.detect_smiles(image, all_faces)
        
        # Draw all detections
        result_image = self.draw_detections(image, all_faces, eyes_per_face, smiles_per_face)
        
        # Save result if requested
        if save_result:
            output_path = f"analyzed_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, result_image)
            print(f"Analysis result saved as: {output_path}")
        
        # Compile results
        results = {
            'image_path': image_path,
            'total_faces': len(all_faces),
            'frontal_faces': len(face_detections['frontal']),
            'profile_faces': len(face_detections['profile']),
            'face_coordinates': all_faces,
            'eyes_detected': sum(len(eyes) for eyes in eyes_per_face),
            'smiles_detected': sum(len(smiles) for smiles in smiles_per_face),
            'eyes_per_face': eyes_per_face,
            'smiles_per_face': smiles_per_face
        }
        
        return results
    
    def batch_process_images(self, input_dir: str, output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Process multiple images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results (optional)
            
        Returns:
            List of analysis results for each image
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(input_dir, filename)
                
                try:
                    result = self.analyze_image(image_path, save_result=False)
                    
                    if output_dir:
                        # Save result image
                        result_image = self.draw_detections(
                            cv2.imread(image_path), 
                            result['face_coordinates'],
                            result['eyes_per_face'],
                            result['smiles_per_face']
                        )
                        output_path = os.path.join(output_dir, f"analyzed_{filename}")
                        cv2.imwrite(output_path, result_image)
                    
                    results.append(result)
                    print(f"Processed: {filename} - Found {result['total_faces']} faces")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        return results

def main():
    """Example usage of the ImageFaceDetector class."""
    # Initialize detector
    detector = ImageFaceDetector()
    
    # Example 1: Analyze a single image
    image_path = "sample_image.jpg"
    
    if os.path.exists(image_path):
        try:
            results = detector.analyze_image(image_path)
            
            print("\n=== Face Analysis Results ===")
            print(f"Image: {results['image_path']}")
            print(f"Total faces: {results['total_faces']}")
            print(f"Frontal faces: {results['frontal_faces']}")
            print(f"Profile faces: {results['profile_faces']}")
            print(f"Eyes detected: {results['eyes_detected']}")
            print(f"Smiles detected: {results['smiles_detected']}")
            
            # Print face coordinates
            for i, (x, y, w, h) in enumerate(results['face_coordinates']):
                print(f"Face {i+1}: x={x}, y={y}, width={w}, height={h}")
                
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
    else:
        print(f"Image {image_path} not found. Please provide a valid image path.")
    
    # Example 2: Batch process images in a directory
    input_directory = "input_images"
    if os.path.exists(input_directory):
        print(f"\n=== Batch Processing Images in {input_directory} ===")
        results = detector.batch_process_images(input_directory, "output_images")
        print(f"Processed {len(results)} images successfully")

if __name__ == "__main__":
    main()
