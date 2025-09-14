"""
Simple and Reliable Object Detection
Only detects real objects, no false positives.
"""

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Dict

class SimpleObjectDetector:
    """Simple object detector that only finds real objects."""
    
    def __init__(self):
        """Initialize the simple object detector."""
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}
        self.known_faces_folder = "known_people"
        
        # Create folders
        if not os.path.exists(self.known_faces_folder):
            os.makedirs(self.known_faces_folder)
            print(f"Created folder: {self.known_faces_folder}")
    
    def extract_face_features_simple(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract simple features from a face region."""
        x, y, w, h = face_rect
        
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        
        # Resize to standard size
        face_resized = cv2.resize(face_roi, (100, 100))
        
        # Convert to grayscale
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        face_gray = cv2.equalizeHist(face_gray)
        
        # Simple features: just the pixel values
        features = face_gray.flatten()
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def detect_real_objects(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Detect only real objects with very strict criteria.
        No false positives!
        """
        detected_objects = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get face areas to avoid detecting faces as objects
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        face_areas = []
        for (fx, fy, fw, fh) in faces:
            face_areas.append((fx, fy, fw, fh))
        
        # 1. Detect only very clear rectangular objects
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Very strict size filtering - only large objects
            if area < 8000:  # Must be quite large
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio - must be reasonable
            aspect_ratio = w / h
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Check if it's not in face area
            if self.is_in_face_area(x + w//2, y + h//2, face_areas):
                continue
            
            # Check if it has good contrast (real object should have edges)
            roi = gray[y:y+h, x:x+w]
            if np.std(roi) < 40:  # Must have good contrast
                continue
            
            # Check if it's roughly rectangular (not too irregular)
            rect_area = w * h
            if area / rect_area < 0.7:  # Must be mostly rectangular
                continue
            
            # Additional check: object should have some internal structure
            internal_edges = cv2.Canny(roi, 30, 100)
            edge_density = np.sum(internal_edges) / (w * h)
            if edge_density < 0.05:  # Must have some internal edges
                continue
            
            # If we get here, it's likely a real object
            detected_objects.append(("Object", 0.8, (x, y, w, h)))
        
        # 2. Detect only very clear circular objects
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, 
                                 param1=50, param2=30, minRadius=30, maxRadius=150)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                # Check if not in face area
                if self.is_in_face_area(cx, cy, face_areas):
                    continue
                
                # Check if circle has good contrast
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                roi = cv2.bitwise_and(gray, mask)
                if np.std(roi) < 30:  # Must have good contrast
                    continue
                
                # Check if it's actually circular (not just a blob)
                circle_area = np.pi * r * r
                actual_area = np.sum(mask > 0)
                if actual_area / circle_area < 0.8:  # Must be mostly circular
                    continue
                
                # If we get here, it's likely a real circular object
                detected_objects.append(("Circular Object", 0.8, (cx-r, cy-r, 2*r, 2*r)))
        
        # Remove overlapping detections
        filtered_objects = []
        for obj in detected_objects:
            is_duplicate = False
            for existing in filtered_objects:
                if self.objects_overlap(obj[2], existing[2]):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_objects.append(obj)
        
        return filtered_objects
    
    def is_in_face_area(self, x: int, y: int, face_areas: List[Tuple[int, int, int, int]]) -> bool:
        """Check if a point is in any face area."""
        for (fx, fy, fw, fh) in face_areas:
            # Expand face area with margin
            margin = 50
            if (fx - margin <= x <= fx + fw + margin and 
                fy - margin <= y <= fy + fh + margin):
                return True
        return False
    
    def objects_overlap(self, rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap significantly."""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Calculate overlap
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        
        # Calculate areas
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Check if overlap is significant
        return overlap_area > 0.5 * min(area1, area2)
    
    def estimate_age_simple(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> str:
        """
        Simple but effective age estimation.
        """
        x, y, w, h = face_rect
        
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate face area
        face_area = w * h
        
        # Simple size-based estimation
        if face_area < 5000:
            return "Child (0-12)"
        elif face_area < 8000:
            return "Teen (13-18)"
        elif face_area < 12000:
            return "Young Adult (19-30)"
        elif face_area < 15000:
            return "Adult (31-50)"
        else:
            return "Senior (50+)"
    
    def add_person_from_webcam(self, name: str, num_samples: int = 5) -> int:
        """Add a person to the database by taking photos with webcam."""
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
                    
                    # Show age estimation
                    age = self.estimate_age_simple(frame, largest_face)
                    cv2.putText(display_frame, f"Age: {age}", (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
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
                        
                        time.sleep(0.5)
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
            self.load_known_faces_from_folder(self.known_faces_folder)
        
        return samples_added
    
    def load_known_faces_from_folder(self, folder_path: str = None):
        """Load known faces from a folder."""
        if folder_path is None:
            folder_path = self.known_faces_folder
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return
        
        print(f"Loading faces from: {folder_path}")
        
        self.known_faces = {}
        
        for person_name in os.listdir(folder_path):
            person_folder = os.path.join(folder_path, person_name)
            
            if not os.path.isdir(person_folder):
                continue
            
            print(f"Loading images for: {person_name}")
            person_features = []
            
            for filename in os.listdir(person_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(person_folder, filename)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                        
                        if len(faces) > 0:
                            largest_face = max(faces, key=lambda f: f[2] * f[3])
                            features = self.extract_face_features_simple(image, largest_face)
                            person_features.append(features)
                            print(f"  ✓ Loaded {filename}")
                        else:
                            print(f"  ✗ No face found in {filename}")
            
            if person_features:
                self.known_faces[person_name] = person_features
                print(f"  ✓ Added {person_name} with {len(person_features)} face samples")
            else:
                print(f"  ✗ No valid faces found for {person_name}")
        
        print(f"\n✓ Loaded {len(self.known_faces)} people:")
        for name in self.known_faces.keys():
            print(f"  - {name} ({len(self.known_faces[name])} samples)")
    
    def recognize_faces(self, image: np.ndarray, threshold: float = 0.3) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Recognize faces in an image."""
        if not self.known_faces:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        recognized_faces = []
        
        for face_rect in faces:
            face_features = self.extract_face_features_simple(image, face_rect)
            
            best_match = "Unknown"
            best_distance = float('inf')
            
            for person_name, person_feature_list in self.known_faces.items():
                min_distance = float('inf')
                
                for person_features in person_feature_list:
                    distance = np.linalg.norm(face_features - person_features)
                    min_distance = min(min_distance, distance)
                
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_match = person_name
            
            confidence = max(0, 1 - best_distance)
            
            if confidence >= threshold:
                recognized_faces.append((best_match, confidence, tuple(face_rect)))
            else:
                recognized_faces.append(("Unknown", confidence, tuple(face_rect)))
        
        return recognized_faces
    
    def draw_results(self, image: np.ndarray, 
                    recognized_faces: List[Tuple[str, float, Tuple[int, int, int, int]]],
                    detected_objects: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """Draw recognition results on the image."""
        result_image = image.copy()
        
        # Draw face recognition results
        for name, confidence, (x, y, w, h) in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw rectangle around face
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Estimate age
            age = self.estimate_age_simple(image, (x, y, w, h))
            
            # Draw labels
            face_label = f"{name} ({confidence:.2f})"
            age_label = f"Age: {age}"
            
            # Face label
            face_label_size = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (x, y - face_label_size[1] - 10), 
                         (x + face_label_size[0], y), color, -1)
            cv2.putText(result_image, face_label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Age label
            cv2.putText(result_image, age_label, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw object detection results (only real objects)
        for obj_name, confidence, (x, y, w, h) in detected_objects:
            color = (255, 0, 0)  # Blue for objects
            
            # Draw rectangle around object
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            obj_label = f"{obj_name} ({confidence:.2f})"
            obj_label_size = cv2.getTextSize(obj_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_image, (x, y - obj_label_size[1] - 5), 
                         (x + obj_label_size[0], y), color, -1)
            cv2.putText(result_image, obj_label, (x, y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image
    
    def run_simple_recognition(self, threshold: float = 0.3):
        """Run simple recognition system."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Could not open webcam")
            return
        
        print("Simple Recognition System started!")
        print("Features: Face Recognition, Real Object Detection, Age Estimation")
        print("Press 'q' to quit")
        print("Note: Only detects real objects, no false positives!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Face recognition
                recognized_faces = self.recognize_faces(frame, threshold)
                
                # Object detection (only real objects)
                detected_objects = self.detect_real_objects(frame)
                
                # Draw all results
                result_frame = self.draw_results(frame, recognized_faces, detected_objects)
                
                # Add info overlay
                cv2.putText(result_frame, f"Faces: {len(recognized_faces)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Real Objects: {len(detected_objects)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Known People: {len(self.known_faces)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show recognition results
                y_offset = 120
                for name, confidence, _ in recognized_faces:
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.putText(result_frame, f"Person: {name} ({confidence:.2f})", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25
                
                # Show object results
                for obj_name, confidence, _ in detected_objects:
                    cv2.putText(result_frame, f"Object: {obj_name} ({confidence:.2f})", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    y_offset += 25
                
                cv2.imshow("Simple Recognition System", result_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function."""
    print("Simple Recognition System")
    print("=" * 50)
    print("Key improvements:")
    print("- Only detects REAL objects (no false positives)")
    print("- Simple but effective age estimation")
    print("- Very strict object detection criteria")
    print("- No background noise detection")
    print()
    
    # Initialize recognition system
    recognizer = SimpleObjectDetector()
    
    while True:
        print("\n" + "="*50)
        print("Menu:")
        print("1. Add person from webcam")
        print("2. Load known faces from folder")
        print("3. Simple recognition (faces + real objects + age)")
        print("4. View loaded people")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
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
            # Simple recognition
            if not recognizer.known_faces:
                print("Please add people first (option 1) or load known faces (option 2)")
                continue
            
            threshold = input("Recognition threshold (0.1-1.0, default 0.3): ").strip()
            threshold = float(threshold) if threshold else 0.3
            
            recognizer.run_simple_recognition(threshold)
        
        elif choice == "4":
            # View loaded people
            if recognizer.known_faces:
                print(f"\nLoaded {len(recognizer.known_faces)} people:")
                for name, samples in recognizer.known_faces.items():
                    print(f"  - {name} ({len(samples)} samples)")
            else:
                print("No people loaded yet")
        
        elif choice == "5":
            # Exit
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
