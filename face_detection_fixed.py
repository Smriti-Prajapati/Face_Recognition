"""
Fixed Face Detection System
This version properly handles numpy arrays from OpenCV
"""

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Optional

class SimpleFaceDetector:
    """A simplified face detector using only OpenCV."""
    
    def __init__(self):
        """Initialize the face detector."""
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load additional cascades
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Check if cascades loaded successfully
        if self.face_cascade.empty():
            raise ValueError("Could not load face cascade classifier")
    
    def detect_faces(self, image: np.ndarray, scale_factor: float = 1.1, 
                    min_neighbors: int = 5) -> np.ndarray:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            
        Returns:
            Numpy array of rectangles (x, y, w, h) where faces are detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        return faces
    
    def detect_eyes(self, image: np.ndarray, faces: np.ndarray) -> List[List[Tuple[int, int, int, int]]]:
        """
        Detect eyes within detected face regions.
        
        Args:
            image: Input image
            faces: Numpy array of face rectangles
            
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
    
    def detect_smiles(self, image: np.ndarray, faces: np.ndarray) -> List[List[Tuple[int, int, int, int]]]:
        """
        Detect smiles within detected face regions.
        
        Args:
            image: Input image
            faces: Numpy array of face rectangles
            
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
    
    def draw_detections(self, image: np.ndarray, faces: np.ndarray, 
                       eyes_per_face: List[List[Tuple[int, int, int, int]]] = None,
                       smiles_per_face: List[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
        """
        Draw all detections on the image.
        
        Args:
            image: Input image
            faces: Numpy array of face rectangles
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
    
    def analyze_image(self, image_path: str, save_result: bool = True) -> dict:
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
        
        # Detect faces
        faces = self.detect_faces(image)
        
        # Detect eyes and smiles
        eyes_per_face = self.detect_eyes(image, faces)
        smiles_per_face = self.detect_smiles(image, faces)
        
        # Draw all detections
        result_image = self.draw_detections(image, faces, eyes_per_face, smiles_per_face)
        
        # Save result if requested
        if save_result:
            output_path = f"analyzed_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, result_image)
            print(f"Analysis result saved as: {output_path}")
        
        # Compile results
        results = {
            'image_path': image_path,
            'total_faces': len(faces),
            'face_coordinates': faces.tolist() if len(faces) > 0 else [],
            'eyes_detected': sum(len(eyes) for eyes in eyes_per_face),
            'smiles_detected': sum(len(smiles) for smiles in smiles_per_face),
            'eyes_per_face': eyes_per_face,
            'smiles_per_face': smiles_per_face
        }
        
        return results
    
    def run_webcam_detection(self, camera_index: int = 0):
        """
        Run real-time face detection on webcam.
        
        Args:
            camera_index: Camera index (usually 0 for default camera)
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        print("Face detection started. Press 'q' to quit, 's' to save screenshot.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Detect eyes and smiles
                eyes_per_face = self.detect_eyes(frame, faces)
                smiles_per_face = self.detect_smiles(frame, faces)
                
                # Draw detections
                result_frame = self.draw_detections(frame, faces, eyes_per_face, smiles_per_face)
                
                # Add info overlay
                cv2.putText(result_frame, f"Faces: {len(faces)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Eyes: {sum(len(eyes) for eyes in eyes_per_face)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Smiles: {sum(len(smiles) for smiles in smiles_per_face)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Simple Face Detection", result_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, result_frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("Face detection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Face detection stopped")

def main():
    """Example usage of the SimpleFaceDetector class."""
    # Initialize detector
    detector = SimpleFaceDetector()
    
    print("Simple Face Detection System")
    print("=" * 40)
    print("1. Analyze image")
    print("2. Webcam detection")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Image analysis
        image_path = input("Enter image path: ").strip()
        
        # Remove quotes if present
        if image_path.startswith('"') and image_path.endswith('"'):
            image_path = image_path[1:-1]
        elif image_path.startswith("'") and image_path.endswith("'"):
            image_path = image_path[1:-1]
        
        print(f"Looking for image at: {image_path}")
        print(f"File exists: {os.path.exists(image_path)}")
        
        if not image_path or not os.path.exists(image_path):
            print("Invalid image path. Please check the path and try again.")
            print("Tip: Don't use quotes around the path, or use forward slashes")
            return
        
        try:
            results = detector.analyze_image(image_path)
            
            print(f"\nAnalysis Results:")
            print(f"  Total faces: {results['total_faces']}")
            print(f"  Eyes detected: {results['eyes_detected']}")
            print(f"  Smiles detected: {results['smiles_detected']}")
            
            # Print face coordinates
            for i, (x, y, w, h) in enumerate(results['face_coordinates']):
                print(f"  Face {i+1}: x={x}, y={y}, width={w}, height={h}")
            
            # Display result
            result_image = cv2.imread(f"analyzed_{os.path.basename(image_path)}")
            if result_image is not None:
                cv2.imshow("Face Detection Result", result_image)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
    
    elif choice == "2":
        # Webcam detection
        try:
            detector.run_webcam_detection()
        except Exception as e:
            print(f"Error with webcam detection: {str(e)}")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
