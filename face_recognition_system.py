"""
Face Recognition System
This system can learn to recognize specific people and identify them in new images/videos.
"""

import cv2
import numpy as np
import os
import pickle
import json
import time
from typing import List, Tuple, Optional, Dict, Any

class FaceRecognitionSystem:
    """A complete face recognition system using OpenCV."""
    
    def __init__(self, database_file: str = "face_database.pkl"):
        """
        Initialize the face recognition system.
        
        Args:
            database_file: Path to save/load the face database
        """
        self.database_file = database_file
        self.face_database = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load existing database
        self.load_database()
    
    def load_database(self):
        """Load the face database from file."""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'rb') as f:
                    self.face_database = pickle.load(f)
                print(f"Loaded database with {len(self.face_database)} people")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.face_database = {}
        else:
            print("No existing database found. Starting fresh.")
    
    def save_database(self):
        """Save the face database to file."""
        try:
            with open(self.database_file, 'wb') as f:
                pickle.dump(self.face_database, f)
            print(f"Database saved with {len(self.face_database)} people")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def extract_face_features(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract features from a face region.
        This is a simplified feature extraction - in practice, you'd use more sophisticated methods.
        
        Args:
            image: Input image
            face_rect: Face rectangle (x, y, w, h)
            
        Returns:
            Feature vector representing the face
        """
        x, y, w, h = face_rect
        
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        
        # Resize to standard size
        face_resized = cv2.resize(face_roi, (100, 100))
        
        # Convert to grayscale
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Flatten to create feature vector
        features = face_gray.flatten()
        
        # Normalize features
        features = features.astype(np.float32) / 255.0
        
        return features
    
    def add_person(self, name: str, image_path: str) -> bool:
        """
        Add a person to the recognition database.
        
        Args:
            name: Name of the person
            image_path: Path to the person's image
            
        Returns:
            True if person was added successfully
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return False
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return False
        
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            print(f"No faces detected in {image_path}")
            return False
        
        if len(faces) > 1:
            print(f"Multiple faces detected in {image_path}. Using the largest face.")
            # Use the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            faces = [largest_face]
        
        # Extract features from the face
        face_features = self.extract_face_features(image, faces[0])
        
        # Add to database
        if name not in self.face_database:
            self.face_database[name] = []
        
        self.face_database[name].append({
            'features': face_features,
            'image_path': image_path,
            'face_rect': faces[0].tolist(),
            'added_date': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Save database
        self.save_database()
        
        print(f"Added {name} to database. Total samples: {len(self.face_database[name])}")
        return True
    
    def add_person_from_webcam(self, name: str, num_samples: int = 5) -> int:
        """
        Add a person to the database by taking photos with webcam.
        
        Args:
            name: Name of the person
            num_samples: Number of photos to take
            
        Returns:
            Number of successful samples added
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Could not open webcam")
            return 0
        
        print(f"Taking {num_samples} photos of {name}")
        print("Press SPACE to capture, 'q' to quit")
        
        samples_added = 0
        sample_count = 0
        
        try:
            while sample_count < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                
                # Draw rectangle around face
                display_frame = frame.copy()
                if len(faces) > 0:
                    # Use the largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Sample {sample_count + 1}/{num_samples}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add instructions
                cv2.putText(display_frame, f"Capturing {name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "SPACE: Capture, Q: Quit", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Add Person to Database", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Space key
                    if len(faces) > 0:
                        # Extract features and add to database
                        face_features = self.extract_face_features(frame, largest_face)
                        
                        if name not in self.face_database:
                            self.face_database[name] = []
                        
                        self.face_database[name].append({
                            'features': face_features,
                            'image_path': f"webcam_sample_{sample_count + 1}.jpg",
                            'face_rect': largest_face.tolist(),
                            'added_date': time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Save the sample image
                        cv2.imwrite(f"webcam_sample_{sample_count + 1}.jpg", frame)
                        
                        samples_added += 1
                        sample_count += 1
                        print(f"Captured sample {sample_count}/{num_samples}")
                    else:
                        print("No face detected. Please position your face in the camera.")
                
                elif key == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if samples_added > 0:
            self.save_database()
            print(f"Added {samples_added} samples for {name}")
        
        return samples_added
    
    def recognize_faces(self, image: np.ndarray, threshold: float = 0.6) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Recognize faces in an image.
        
        Args:
            image: Input image
            threshold: Recognition threshold (lower = more strict)
            
        Returns:
            List of (name, confidence, face_location) tuples
        """
        if not self.face_database:
            return []
        
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        recognized_faces = []
        
        for face_rect in faces:
            # Extract features from face
            face_features = self.extract_face_features(image, face_rect)
            
            best_match = None
            best_distance = float('inf')
            
            # Compare with all people in database
            for person_name, samples in self.face_database.items():
                for sample in samples:
                    # Calculate distance between features
                    distance = np.linalg.norm(face_features - sample['features'])
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = person_name
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = max(0, 1 - best_distance)
            
            if confidence >= threshold:
                recognized_faces.append((best_match, confidence, tuple(face_rect)))
            else:
                recognized_faces.append(("Unknown", confidence, tuple(face_rect)))
        
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
        
        for name, confidence, (x, y, w, h) in recognized_faces:
            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw rectangle around face
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the face database.
        
        Returns:
            Dictionary with database information
        """
        total_samples = sum(len(samples) for samples in self.face_database.values())
        
        info = {
            'total_people': len(self.face_database),
            'total_samples': total_samples,
            'people': list(self.face_database.keys()),
            'samples_per_person': {name: len(samples) for name, samples in self.face_database.items()}
        }
        
        return info
    
    def remove_person(self, name: str) -> bool:
        """
        Remove a person from the database.
        
        Args:
            name: Name of the person to remove
            
        Returns:
            True if person was removed
        """
        if name in self.face_database:
            del self.face_database[name]
            self.save_database()
            print(f"Removed {name} from database")
            return True
        else:
            print(f"{name} not found in database")
            return False
    
    def clear_database(self):
        """Clear the entire face database."""
        self.face_database = {}
        self.save_database()
        print("Database cleared")

def main():
    """Main function to run the face recognition system."""
    system = FaceRecognitionSystem()
    
    while True:
        print("\n" + "="*50)
        print("Face Recognition System")
        print("="*50)
        print("1. Add person from image")
        print("2. Add person from webcam")
        print("3. Recognize faces in image")
        print("4. Recognize faces in webcam")
        print("5. View database info")
        print("6. Remove person")
        print("7. Clear database")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == "1":
            # Add person from image
            name = input("Enter person's name: ").strip()
            image_path = input("Enter image path: ").strip()
            
            # Remove quotes if present
            if image_path.startswith('"') and image_path.endswith('"'):
                image_path = image_path[1:-1]
            elif image_path.startswith("'") and image_path.endswith("'"):
                image_path = image_path[1:-1]
            
            if name and image_path:
                system.add_person(name, image_path)
            else:
                print("Invalid input")
        
        elif choice == "2":
            # Add person from webcam
            name = input("Enter person's name: ").strip()
            if name:
                num_samples = input("Number of samples to capture (default 5): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 5
                system.add_person_from_webcam(name, num_samples)
            else:
                print("Please enter a name")
        
        elif choice == "3":
            # Recognize faces in image
            image_path = input("Enter image path: ").strip()
            
            # Remove quotes if present
            if image_path.startswith('"') and image_path.endswith('"'):
                image_path = image_path[1:-1]
            elif image_path.startswith("'") and image_path.endswith("'"):
                image_path = image_path[1:-1]
            
            if image_path and os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    threshold = input("Recognition threshold (0.1-1.0, default 0.6): ").strip()
                    threshold = float(threshold) if threshold else 0.6
                    
                    recognized_faces = system.recognize_faces(image, threshold)
                    result_image = system.draw_recognition_results(image, recognized_faces)
                    
                    print(f"\nRecognition Results:")
                    for name, confidence, location in recognized_faces:
                        print(f"  {name}: {confidence:.2f}")
                    
                    # Save result
                    output_path = f"recognition_result_{int(time.time())}.jpg"
                    cv2.imwrite(output_path, result_image)
                    print(f"Result saved as: {output_path}")
                    
                    # Display result
                    cv2.imshow("Face Recognition", result_image)
                    print("Press any key to close...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Could not load image")
            else:
                print("Image not found")
        
        elif choice == "4":
            # Recognize faces in webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Could not open webcam")
                continue
            
            print("Real-time face recognition started. Press 'q' to quit")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Recognize faces
                    recognized_faces = system.recognize_faces(frame)
                    result_frame = system.draw_recognition_results(frame, recognized_faces)
                    
                    # Add info overlay
                    cv2.putText(result_frame, f"Faces: {len(recognized_faces)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Real-time Face Recognition", result_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            finally:
                cap.release()
                cv2.destroyAllWindows()
        
        elif choice == "5":
            # View database info
            info = system.get_database_info()
            print(f"\nDatabase Information:")
            print(f"  Total people: {info['total_people']}")
            print(f"  Total samples: {info['total_samples']}")
            print(f"  People in database: {', '.join(info['people'])}")
            
            for person, count in info['samples_per_person'].items():
                print(f"    {person}: {count} samples")
        
        elif choice == "6":
            # Remove person
            name = input("Enter person's name to remove: ").strip()
            if name:
                system.remove_person(name)
        
        elif choice == "7":
            # Clear database
            confirm = input("Are you sure? (y/n): ").strip().lower()
            if confirm == 'y':
                system.clear_database()
        
        elif choice == "8":
            # Exit
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
