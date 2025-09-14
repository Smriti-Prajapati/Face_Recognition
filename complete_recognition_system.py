"""
Complete Recognition System
Includes: Face Recognition, Object Recognition, and Age Detection
"""

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Dict

class CompleteRecognitionSystem:
    """Complete recognition system with face, object, and age detection."""
    
    def __init__(self):
        """Initialize the complete recognition system."""
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}
        self.known_faces_folder = "known_people"
        
        # Object detection (using YOLO-like approach with OpenCV)
        self.object_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Age detection using face landmarks
        self.age_groups = ['0-2', '3-6', '7-12', '13-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+']
        
        # Create folders
        if not os.path.exists(self.known_faces_folder):
            os.makedirs(self.known_faces_folder)
            print(f"Created folder: {self.known_faces_folder}")
    
    def extract_face_features_advanced(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract advanced features from a face region."""
        x, y, w, h = face_rect
        
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        
        # Resize to standard size
        face_resized = cv2.resize(face_roi, (150, 150))
        
        # Convert to grayscale
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        face_gray = cv2.equalizeHist(face_gray)
        
        # Apply Gaussian blur
        face_gray = cv2.GaussianBlur(face_gray, (3, 3), 0)
        
        # Extract multiple feature types
        features = []
        
        # Raw pixel features
        raw_features = face_gray.flatten()
        features.extend(raw_features)
        
        # Histogram features
        hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
        
        # Edge features
        edges = cv2.Canny(face_gray, 50, 150)
        edge_features = edges.flatten()
        features.extend(edge_features)
        
        # Convert to numpy array and normalize
        features = np.array(features, dtype=np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def detect_objects_simple(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Simple object detection using color and shape analysis.
        This is a basic implementation - for production, use YOLO or similar.
        """
        detected_objects = []
        
        # Convert to different color spaces for better detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect common objects using color and shape
        objects = []
        
        # 1. Detect circular objects (bottles, cups, etc.)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                objects.append(("Bottle/Cup", 0.8, (x-r, y-r, 2*r, 2*r)))
        
        # 2. Detect rectangular objects (books, phones, etc.)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 0.5 < aspect_ratio < 2.0:  # Roughly rectangular
                    if area > 5000:
                        objects.append(("Book/Phone", 0.7, (x, y, w, h)))
                    else:
                        objects.append(("Small Object", 0.6, (x, y, w, h)))
        
        # 3. Detect faces as objects
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            objects.append(("Person", 0.9, (x, y, w, h)))
        
        # 4. Detect colored objects
        # Red objects
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                objects.append(("Red Object", 0.6, (x, y, w, h)))
        
        # Blue objects
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in blue_contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                objects.append(("Blue Object", 0.6, (x, y, w, h)))
        
        return objects
    
    def estimate_age_from_face(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> str:
        """
        Estimate age from face using simple heuristics.
        This is a basic implementation - for production, use a trained age model.
        """
        x, y, w, h = face_rect
        
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Simple age estimation based on face characteristics
        face_area = w * h
        
        # Detect wrinkles (more wrinkles = older)
        edges = cv2.Canny(face_gray, 50, 150)
        wrinkle_density = np.sum(edges) / (w * h)
        
        # Detect face smoothness
        laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        
        # Simple age estimation based on these features
        if face_area < 2000:  # Very small face
            age_group = "0-2"
        elif face_area < 4000:  # Small face
            if wrinkle_density < 0.1:
                age_group = "3-6"
            else:
                age_group = "7-12"
        elif face_area < 8000:  # Medium face
            if wrinkle_density < 0.15:
                age_group = "13-18"
            elif wrinkle_density < 0.25:
                age_group = "19-25"
            else:
                age_group = "26-35"
        else:  # Large face
            if wrinkle_density < 0.2:
                age_group = "19-25"
            elif wrinkle_density < 0.3:
                age_group = "26-35"
            elif wrinkle_density < 0.4:
                age_group = "36-45"
            elif wrinkle_density < 0.5:
                age_group = "46-55"
            elif wrinkle_density < 0.6:
                age_group = "56-65"
            else:
                age_group = "65+"
        
        return age_group
    
    def add_person_from_webcam(self, name: str, num_samples: int = 8) -> int:
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
        print("IMPORTANT: Move your head slightly between photos!")
        
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
                    age = self.estimate_age_from_face(frame, largest_face)
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
                            features = self.extract_face_features_advanced(image, largest_face)
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
            face_features = self.extract_face_features_advanced(image, face_rect)
            
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
    
    def draw_complete_results(self, image: np.ndarray, 
                            recognized_faces: List[Tuple[str, float, Tuple[int, int, int, int]]],
                            detected_objects: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """Draw all recognition results on the image."""
        result_image = image.copy()
        
        # Draw face recognition results
        for name, confidence, (x, y, w, h) in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw rectangle around face
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Estimate age
            age = self.estimate_age_from_face(image, (x, y, w, h))
            
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
        
        # Draw object detection results
        for obj_name, confidence, (x, y, w, h) in detected_objects:
            if obj_name != "Person":  # Don't draw person objects (already drawn as faces)
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
    
    def run_complete_recognition(self, threshold: float = 0.3):
        """Run complete recognition system with face, object, and age detection."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Could not open webcam")
            return
        
        print("Complete Recognition System started!")
        print("Features: Face Recognition, Object Detection, Age Estimation")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Face recognition
                recognized_faces = self.recognize_faces(frame, threshold)
                
                # Object detection
                detected_objects = self.detect_objects_simple(frame)
                
                # Draw all results
                result_frame = self.draw_complete_results(frame, recognized_faces, detected_objects)
                
                # Add info overlay
                cv2.putText(result_frame, f"Faces: {len(recognized_faces)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Objects: {len(detected_objects)}", (10, 60), 
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
                    if obj_name != "Person":
                        cv2.putText(result_frame, f"Object: {obj_name} ({confidence:.2f})", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        y_offset += 25
                
                cv2.imshow("Complete Recognition System", result_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function."""
    print("Complete Recognition System")
    print("=" * 50)
    print("Features:")
    print("- Face Recognition (identify known people)")
    print("- Object Detection (detect common objects)")
    print("- Age Estimation (estimate person's age)")
    print()
    
    # Initialize recognition system
    recognizer = CompleteRecognitionSystem()
    
    while True:
        print("\n" + "="*50)
        print("Menu:")
        print("1. Add person from webcam (with age detection)")
        print("2. Load known faces from folder")
        print("3. Complete recognition (faces + objects + age)")
        print("4. Face recognition only")
        print("5. Object detection only")
        print("6. View loaded people")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            # Add person from webcam
            name = input("Enter person's name: ").strip()
            if name:
                num_samples = input("Number of samples to capture (default 8): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 8
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
            # Complete recognition
            if not recognizer.known_faces:
                print("Please add people first (option 1) or load known faces (option 2)")
                continue
            
            threshold = input("Recognition threshold (0.1-1.0, default 0.3): ").strip()
            threshold = float(threshold) if threshold else 0.3
            
            recognizer.run_complete_recognition(threshold)
        
        elif choice == "4":
            # Face recognition only
            if not recognizer.known_faces:
                print("Please add people first (option 1) or load known faces (option 2)")
                continue
            
            threshold = input("Recognition threshold (0.1-1.0, default 0.3): ").strip()
            threshold = float(threshold) if threshold else 0.3
            
            # Simple face recognition (you can implement this)
            print("Face recognition only mode - implement if needed")
        
        elif choice == "5":
            # Object detection only
            print("Object detection only mode - implement if needed")
        
        elif choice == "6":
            # View loaded people
            if recognizer.known_faces:
                print(f"\nLoaded {len(recognizer.known_faces)} people:")
                for name, samples in recognizer.known_faces.items():
                    print(f"  - {name} ({len(samples)} samples)")
            else:
                print("No people loaded yet")
        
        elif choice == "7":
            # Exit
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
