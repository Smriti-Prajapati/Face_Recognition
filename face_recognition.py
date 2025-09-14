"""
Face Recognition Module
This module provides face recognition and identification capabilities.
"""

import cv2
import numpy as np
import face_recognition
import os
import pickle
import json
from typing import List, Tuple, Optional, Dict, Any
from face_detection_basic import FaceDetector

class FaceRecognizer(FaceDetector):
    """Face detector with recognition and identification capabilities."""
    
    def __init__(self, cascade_path: Optional[str] = None, 
                 encodings_file: str = "face_encodings.pkl"):
        """
        Initialize the face recognizer.
        
        Args:
            cascade_path: Path to Haar cascade XML file
            encodings_file: Path to save/load face encodings
        """
        super().__init__(cascade_path)
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_encodings_data = {}
        
        # Load existing encodings if available
        self.load_encodings()
    
    def load_encodings(self):
        """Load face encodings from file."""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    self.face_encodings_data = pickle.load(f)
                    self.known_face_encodings = self.face_encodings_data.get('encodings', [])
                    self.known_face_names = self.face_encodings_data.get('names', [])
                print(f"Loaded {len(self.known_face_names)} known faces")
            except Exception as e:
                print(f"Error loading encodings: {str(e)}")
                self.face_encodings_data = {}
                self.known_face_encodings = []
                self.known_face_names = []
        else:
            print("No existing encodings found. Starting fresh.")
    
    def save_encodings(self):
        """Save face encodings to file."""
        self.face_encodings_data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(self.face_encodings_data, f)
            print(f"Saved {len(self.known_face_names)} face encodings")
        except Exception as e:
            print(f"Error saving encodings: {str(e)}")
    
    def encode_face(self, image: np.ndarray, 
                   face_location: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Encode a face from an image.
        
        Args:
            image: Input image
            face_location: Face location (top, right, bottom, left) or None for auto-detection
            
        Returns:
            Face encoding or None if no face found
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if face_location is None:
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image)
            if not face_locations:
                return None
            face_location = face_locations[0]  # Use first face found
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, [face_location])
        
        if face_encodings:
            return face_encodings[0]
        
        return None
    
    def add_face(self, name: str, image: np.ndarray, 
                face_location: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Add a new face to the recognition database.
        
        Args:
            name: Name of the person
            image: Image containing the face
            face_location: Face location or None for auto-detection
            
        Returns:
            True if face was added successfully, False otherwise
        """
        # Encode the face
        face_encoding = self.encode_face(image, face_location)
        
        if face_encoding is not None:
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.save_encodings()
            print(f"Added face for {name}")
            return True
        else:
            print(f"Could not detect face in image for {name}")
            return False
    
    def add_faces_from_directory(self, directory_path: str) -> Dict[str, int]:
        """
        Add faces from a directory of images.
        
        Args:
            directory_path: Path to directory containing face images
            
        Returns:
            Dictionary with success/failure counts
        """
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} not found")
            return {'success': 0, 'failed': 0}
        
        results = {'success': 0, 'failed': 0}
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(image_extensions):
                # Extract name from filename (remove extension)
                name = os.path.splitext(filename)[0]
                
                # Load image
                image_path = os.path.join(directory_path, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    if self.add_face(name, image):
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                else:
                    print(f"Could not load image: {filename}")
                    results['failed'] += 1
        
        print(f"Added {results['success']} faces, {results['failed']} failed")
        return results
    
    def recognize_faces(self, image: np.ndarray, 
                       tolerance: float = 0.6) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Recognize faces in an image.
        
        Args:
            image: Input image
            tolerance: Face recognition tolerance (lower = more strict)
            
        Returns:
            List of (name, confidence, face_location) tuples
        """
        if not self.known_face_encodings:
            return []
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        recognized_faces = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=tolerance
            )
            
            # Calculate face distances
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            
            # Find best match
            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = self.known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
            else:
                name = "Unknown"
                confidence = 0.0
            
            recognized_faces.append((name, confidence, face_location))
        
        return recognized_faces
    
    def draw_recognition_results(self, image: np.ndarray, 
                               recognized_faces: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Draw recognition results on the image.
        
        Args:
            image: Input image
            recognized_faces: List of recognition results
            
        Returns:
            Image with recognition results drawn
        """
        result_image = image.copy()
        
        for name, confidence, (top, right, bottom, left) in recognized_faces:
            # Draw rectangle around face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(result_image, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_image, (left, top - label_size[1] - 10), 
                         (left + label_size[0], top), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (left, top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def recognize_and_draw(self, image: np.ndarray, 
                          tolerance: float = 0.6) -> Tuple[np.ndarray, List[Tuple[str, float, Tuple[int, int, int, int]]]]:
        """
        Recognize faces and draw results in one step.
        
        Args:
            image: Input image
            tolerance: Face recognition tolerance
            
        Returns:
            Tuple of (image with results drawn, recognition results)
        """
        recognized_faces = self.recognize_faces(image, tolerance)
        result_image = self.draw_recognition_results(image, recognized_faces)
        return result_image, recognized_faces
    
    def train_from_video(self, video_path: str, person_name: str, 
                        max_frames: int = 50) -> int:
        """
        Train face recognition from a video file.
        
        Args:
            video_path: Path to video file
            person_name: Name of the person
            max_frames: Maximum number of frames to process
            
        Returns:
            Number of faces successfully added
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return 0
        
        frames_processed = 0
        faces_added = 0
        
        print(f"Training face recognition for {person_name}...")
        
        try:
            while frames_processed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames (process every 10th frame)
                if frames_processed % 10 == 0:
                    if self.add_face(person_name, frame):
                        faces_added += 1
                
                frames_processed += 1
        
        finally:
            cap.release()
        
        print(f"Added {faces_added} faces for {person_name} from {frames_processed} frames")
        return faces_added
    
    def get_face_database_info(self) -> Dict[str, Any]:
        """
        Get information about the face database.
        
        Returns:
            Dictionary with database information
        """
        unique_names = list(set(self.known_face_names))
        name_counts = {name: self.known_face_names.count(name) for name in unique_names}
        
        return {
            'total_faces': len(self.known_face_names),
            'unique_people': len(unique_names),
            'people': unique_names,
            'face_counts': name_counts
        }
    
    def remove_person(self, name: str) -> int:
        """
        Remove all faces of a specific person from the database.
        
        Args:
            name: Name of the person to remove
            
        Returns:
            Number of faces removed
        """
        indices_to_remove = [i for i, n in enumerate(self.known_face_names) if n == name]
        
        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del self.known_face_encodings[i]
            del self.known_face_names[i]
        
        if indices_to_remove:
            self.save_encodings()
            print(f"Removed {len(indices_to_remove)} faces for {name}")
        
        return len(indices_to_remove)
    
    def clear_database(self):
        """Clear the entire face database."""
        self.known_face_encodings = []
        self.known_face_names = []
        self.save_encodings()
        print("Face database cleared")

def main():
    """Example usage of the FaceRecognizer class."""
    # Initialize recognizer
    recognizer = FaceRecognizer()
    
    print("Face Recognition System")
    print("1. Add faces from directory")
    print("2. Add face from image")
    print("3. Recognize faces in image")
    print("4. Train from video")
    print("5. View database info")
    print("6. Remove person")
    print("7. Clear database")
    
    choice = input("Enter your choice (1-7): ").strip()
    
    if choice == "1":
        # Add faces from directory
        directory = input("Enter directory path: ").strip()
        if directory:
            results = recognizer.add_faces_from_directory(directory)
            print(f"Results: {results}")
    
    elif choice == "2":
        # Add face from image
        name = input("Enter person's name: ").strip()
        image_path = input("Enter image path: ").strip()
        
        if name and image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                recognizer.add_face(name, image)
            else:
                print("Could not load image")
        else:
            print("Invalid input")
    
    elif choice == "3":
        # Recognize faces
        image_path = input("Enter image path: ").strip()
        
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                result_image, recognized_faces = recognizer.recognize_and_draw(image)
                
                print(f"Recognized {len(recognized_faces)} faces:")
                for name, confidence, location in recognized_faces:
                    print(f"  {name}: {confidence:.2f}")
                
                # Save result
                cv2.imwrite("recognition_result.jpg", result_image)
                print("Result saved as 'recognition_result.jpg'")
                
                # Display result
                cv2.imshow("Face Recognition", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Could not load image")
        else:
            print("Image not found")
    
    elif choice == "4":
        # Train from video
        video_path = input("Enter video path: ").strip()
        person_name = input("Enter person's name: ").strip()
        
        if video_path and person_name and os.path.exists(video_path):
            faces_added = recognizer.train_from_video(video_path, person_name)
            print(f"Added {faces_added} faces")
        else:
            print("Invalid input")
    
    elif choice == "5":
        # View database info
        info = recognizer.get_face_database_info()
        print(f"Database Info: {json.dumps(info, indent=2)}")
    
    elif choice == "6":
        # Remove person
        name = input("Enter person's name to remove: ").strip()
        if name:
            removed = recognizer.remove_person(name)
            print(f"Removed {removed} faces")
    
    elif choice == "7":
        # Clear database
        confirm = input("Are you sure? (y/n): ").strip().lower()
        if confirm == 'y':
            recognizer.clear_database()
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
