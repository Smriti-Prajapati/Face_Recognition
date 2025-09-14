"""
Fixed Face Recognition System
This version properly saves webcam photos and recognizes them correctly.
"""

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Dict

class FixedFaceRecognition:
    """Fixed face recognition that properly saves and loads faces."""
    
    def __init__(self):
        """Initialize the face recognition system."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}  # Will store face features for each person
        self.known_faces_folder = "known_people"
        
        # Create known faces folder if it doesn't exist
        if not os.path.exists(self.known_faces_folder):
            os.makedirs(self.known_faces_folder)
            print(f"Created folder: {self.known_faces_folder}")
    
    def extract_face_features(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract features from a face region."""
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
    
    def add_person_from_webcam(self, name: str, num_samples: int = 5) -> int:
        """
        Add a person to the database by taking photos with webcam.
        This properly saves photos to the folder structure.
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Could not open webcam")
            return 0
        
        # Create person folder
        person_folder = os.path.join(self.known_faces_folder, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
            print(f"Created folder: {person_folder}")
        
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
                        # Save the photo
                        photo_path = os.path.join(person_folder, f"{name}_sample_{sample_count + 1}.jpg")
                        cv2.imwrite(photo_path, frame)
                        
                        samples_added += 1
                        sample_count += 1
                        print(f"✓ Captured sample {sample_count}/{num_samples} - {photo_path}")
                    else:
                        print("No face detected. Please position your face in the camera.")
                
                elif key == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if samples_added > 0:
            print(f"✓ Added {samples_added} samples for {name}")
            print(f"Photos saved in: {person_folder}")
            # Reload the database
            self.load_known_faces_from_folder(self.known_faces_folder)
        
        return samples_added
    
    def load_known_faces_from_folder(self, folder_path: str = None):
        """
        Load known faces from a folder.
        Each person should have their own subfolder with their images.
        """
        if folder_path is None:
            folder_path = self.known_faces_folder
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return
        
        print(f"Loading faces from: {folder_path}")
        
        # Clear existing known faces
        self.known_faces = {}
        
        # Go through each subfolder (each person)
        for person_name in os.listdir(folder_path):
            person_folder = os.path.join(folder_path, person_name)
            
            if not os.path.isdir(person_folder):
                continue
            
            print(f"Loading images for: {person_name}")
            person_features = []
            
            # Load all images for this person
            for filename in os.listdir(person_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(person_folder, filename)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        # Detect faces
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                        
                        if len(faces) > 0:
                            # Use the largest face
                            largest_face = max(faces, key=lambda f: f[2] * f[3])
                            features = self.extract_face_features(image, largest_face)
                            person_features.append(features)
                            print(f"  ✓ Loaded {filename}")
                        else:
                            print(f"  ✗ No face found in {filename}")
            
            if person_features:
                # Store average features for this person
                self.known_faces[person_name] = np.mean(person_features, axis=0)
                print(f"  ✓ Added {person_name} with {len(person_features)} face samples")
            else:
                print(f"  ✗ No valid faces found for {person_name}")
        
        print(f"\n✓ Loaded {len(self.known_faces)} people:")
        for name in self.known_faces.keys():
            print(f"  - {name}")
    
    def recognize_faces(self, image: np.ndarray, threshold: float = 0.6) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Recognize faces in an image."""
        if not self.known_faces:
            return []
        
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        recognized_faces = []
        
        for face_rect in faces:
            # Extract features from face
            face_features = self.extract_face_features(image, face_rect)
            
            best_match = "Unknown"
            best_distance = float('inf')
            
            # Compare with all known people
            for person_name, person_features in self.known_faces.items():
                # Calculate distance between features
                distance = np.linalg.norm(face_features - person_features)
                
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
        """Draw recognition results on the image."""
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
    
    def recognize_image(self, image_path: str, threshold: float = 0.6):
        """Recognize faces in a single image."""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Recognize faces
        recognized_faces = self.recognize_faces(image, threshold)
        
        # Draw results
        result_image = self.draw_recognition_results(image, recognized_faces)
        
        # Print results
        print(f"\nRecognition Results for {os.path.basename(image_path)}:")
        for name, confidence, location in recognized_faces:
            print(f"  {name}: {confidence:.2f}")
        
        # Save result
        output_path = f"recognized_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result_image)
        print(f"Result saved as: {output_path}")
        
        # Display result
        cv2.imshow("Face Recognition", result_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run_webcam_recognition(self, threshold: float = 0.6):
        """Run real-time face recognition from webcam."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Could not open webcam")
            return
        
        print("Real-time face recognition started. Press 'q' to quit")
        print(f"Recognizing {len(self.known_faces)} known people")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Recognize faces
                recognized_faces = self.recognize_faces(frame, threshold)
                result_frame = self.draw_recognition_results(frame, recognized_faces)
                
                # Add info overlay
                cv2.putText(result_frame, f"Faces: {len(recognized_faces)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Known: {len(self.known_faces)} people", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show recognition results
                for i, (name, confidence, _) in enumerate(recognized_faces):
                    cv2.putText(result_frame, f"{name}: {confidence:.2f}", (10, 90 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Real-time Face Recognition", result_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function."""
    print("Fixed Face Recognition System")
    print("=" * 40)
    print("This version properly saves webcam photos and recognizes them!")
    print()
    
    # Initialize recognition system
    recognizer = FixedFaceRecognition()
    
    while True:
        print("\n" + "="*40)
        print("Menu:")
        print("1. Add person from webcam (takes photos)")
        print("2. Load known faces from folder")
        print("3. Recognize faces in image")
        print("4. Real-time recognition (webcam)")
        print("5. View loaded people")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            # Add person from webcam
            name = input("Enter person's name: ").strip()
            if name:
                num_samples = input("Number of samples to capture (default 5): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 5
                recognizer.add_person_from_webcam(name, num_samples)
            else:
                print("Please enter a name")
        
        elif choice == "2":
            # Load known faces
            folder_path = input("Enter folder path (default: known_people): ").strip()
            if not folder_path:
                folder_path = "known_people"
            
            recognizer.load_known_faces_from_folder(folder_path)
        
        elif choice == "3":
            # Recognize in image
            if not recognizer.known_faces:
                print("Please add people first (option 1) or load known faces (option 2)")
                continue
            
            image_path = input("Enter image path: ").strip()
            
            # Remove quotes if present
            if image_path.startswith('"') and image_path.endswith('"'):
                image_path = image_path[1:-1]
            elif image_path.startswith("'") and image_path.endswith("'"):
                image_path = image_path[1:-1]
            
            threshold = input("Recognition threshold (0.1-1.0, default 0.6): ").strip()
            threshold = float(threshold) if threshold else 0.6
            
            recognizer.recognize_image(image_path, threshold)
        
        elif choice == "4":
            # Real-time recognition
            if not recognizer.known_faces:
                print("Please add people first (option 1) or load known faces (option 2)")
                continue
            
            threshold = input("Recognition threshold (0.1-1.0, default 0.6): ").strip()
            threshold = float(threshold) if threshold else 0.6
            
            recognizer.run_webcam_recognition(threshold)
        
        elif choice == "5":
            # View loaded people
            if recognizer.known_faces:
                print(f"\nLoaded {len(recognizer.known_faces)} people:")
                for name in recognizer.known_faces.keys():
                    print(f"  - {name}")
            else:
                print("No people loaded yet")
        
        elif choice == "6":
            # Exit
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
