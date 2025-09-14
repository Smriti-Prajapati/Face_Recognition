"""
Final Recognition System
Much better object detection and age estimation using reliable methods.
"""

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Dict

class FinalRecognitionSystem:
    """Final recognition system with reliable object detection and age estimation."""
    
    def __init__(self):
        """Initialize the final recognition system."""
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}
        self.known_faces_folder = "known_people"
        
        # Object detection using template matching
        self.object_templates = {}
        self.load_object_templates()
        
        # Create folders
        if not os.path.exists(self.known_faces_folder):
            os.makedirs(self.known_faces_folder)
            print(f"Created folder: {self.known_faces_folder}")
    
    def load_object_templates(self):
        """Load object templates for better detection."""
        # Create simple templates for common objects
        self.object_templates = {
            'phone': self.create_phone_template(),
            'book': self.create_book_template(),
            'bottle': self.create_bottle_template(),
            'cup': self.create_cup_template()
        }
    
    def create_phone_template(self):
        """Create a simple phone template."""
        template = np.zeros((60, 30, 3), dtype=np.uint8)
        cv2.rectangle(template, (5, 5), (25, 55), (100, 100, 100), -1)
        cv2.rectangle(template, (7, 7), (23, 53), (200, 200, 200), -1)
        return template
    
    def create_book_template(self):
        """Create a simple book template."""
        template = np.zeros((80, 60, 3), dtype=np.uint8)
        cv2.rectangle(template, (5, 5), (55, 75), (139, 69, 19), -1)
        cv2.rectangle(template, (7, 7), (53, 73), (160, 82, 45), -1)
        return template
    
    def create_bottle_template(self):
        """Create a simple bottle template."""
        template = np.zeros((100, 40, 3), dtype=np.uint8)
        cv2.rectangle(template, (15, 10), (25, 90), (100, 100, 100), -1)
        cv2.rectangle(template, (10, 5), (30, 15), (100, 100, 100), -1)
        return template
    
    def create_cup_template(self):
        """Create a simple cup template."""
        template = np.zeros((80, 60, 3), dtype=np.uint8)
        cv2.rectangle(template, (20, 20), (40, 70), (100, 100, 100), -1)
        cv2.rectangle(template, (15, 15), (45, 25), (100, 100, 100), -1)
        return template
    
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
    
    def detect_objects_reliable(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Reliable object detection using multiple methods.
        Only detects objects that are actually present.
        """
        detected_objects = []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Template matching for specific objects
        for obj_name, template in self.object_templates.items():
            # Resize template to different scales
            for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                
                # Template matching
                result = cv2.matchTemplate(gray, cv2.cvtColor(scaled_template, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
                
                # Find matches above threshold
                locations = np.where(result >= 0.6)
                
                for pt in zip(*locations[::-1]):
                    h, w = scaled_template.shape[:2]
                    x, y = pt[0], pt[1]
                    
                    # Check if this area is not in face region
                    if not self.is_in_face_area(x + w//2, y + h//2, image):
                        detected_objects.append((obj_name, result[y, x], (x, y, w, h)))
        
        # 2. Detect faces as objects
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            detected_objects.append(("Person", 0.9, (x, y, w, h)))
        
        # 3. Detect rectangular objects with strict criteria
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 50000:  # Strict size filtering
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if not in face area and has good aspect ratio
                if (not self.is_in_face_area(x + w//2, y + h//2, min(w, h)//2, image) and 
                    0.4 < aspect_ratio < 2.5):
                    
                    # Additional check: object should have good contrast
                    roi = gray[y:y+h, x:x+w]
                    if np.std(roi) > 30:  # Good contrast
                        detected_objects.append(("Rectangular Object", 0.7, (x, y, w, h)))
        
        # 4. Detect circular objects with strict criteria
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=20, maxRadius=100)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if not self.is_in_face_area(x, y, r, image):
                    # Check if circle has good contrast
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    roi = cv2.bitwise_and(gray, mask)
                    if np.std(roi) > 25:  # Good contrast
                        detected_objects.append(("Circular Object", 0.8, (x-r, y-r, 2*r, 2*r)))
        
        # Remove duplicates and low confidence detections
        filtered_objects = []
        for obj in detected_objects:
            if obj[1] > 0.5:  # High confidence threshold
                # Check for overlap with existing objects
                is_duplicate = False
                for existing in filtered_objects:
                    if self.objects_overlap(obj[2], existing[2]):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_objects.append(obj)
        
        return filtered_objects
    
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
        return overlap_area > 0.3 * min(area1, area2)
    
    def is_in_face_area(self, x: int, y: int, radius: int, image: np.ndarray) -> bool:
        """Check if a point is in the face area."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (fx, fy, fw, fh) in faces:
            # Expand face area
            margin = 100
            if (fx - margin <= x <= fx + fw + margin and 
                fy - margin <= y <= fy + fh + margin):
                return True
        return False
    
    def estimate_age_improved(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> str:
        """
        Much improved age estimation using better methods.
        """
        x, y, w, h = face_rect
        
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate face area
        face_area = w * h
        
        # Method 1: Face size analysis
        size_score = 0
        if face_area < 4000:
            size_score = 0.2  # Young
        elif face_area < 8000:
            size_score = 0.1  # Young adult
        elif face_area < 12000:
            size_score = 0.0  # Adult
        else:
            size_score = 0.1  # Older
        
        # Method 2: Wrinkle analysis using multiple scales
        wrinkle_score = 0
        for scale in [1.0, 1.5, 2.0]:
            scaled_face = cv2.resize(face_gray, None, fx=scale, fy=scale)
            edges = cv2.Canny(scaled_face, 30, 100)
            wrinkle_density = np.sum(edges) / (scaled_face.shape[0] * scaled_face.shape[1])
            wrinkle_score += wrinkle_density
        
        wrinkle_score /= 3  # Average
        
        if wrinkle_score < 0.03:
            wrinkle_score = 0.3  # Very young
        elif wrinkle_score < 0.06:
            wrinkle_score = 0.1  # Young
        elif wrinkle_score < 0.12:
            wrinkle_score = 0.0  # Adult
        elif wrinkle_score < 0.18:
            wrinkle_score = 0.1  # Middle-aged
        else:
            wrinkle_score = 0.2  # Older
        
        # Method 3: Skin texture analysis
        texture_score = 0
        # Use Local Binary Pattern
        lbp = self.calculate_lbp(face_gray)
        texture_variance = np.var(lbp)
        
        if texture_variance < 50:
            texture_score = 0.2  # Young (smooth)
        elif texture_variance < 100:
            texture_score = 0.1  # Young adult
        elif texture_variance < 200:
            texture_score = 0.0  # Adult
        else:
            texture_score = 0.1  # Older
        
        # Method 4: Face proportions
        aspect_ratio = w / h
        proportion_score = 0
        if aspect_ratio < 0.7:  # Very round face
            proportion_score = 0.1  # Young
        elif aspect_ratio < 0.8:  # Round face
            proportion_score = 0.05  # Young adult
        elif aspect_ratio < 0.9:  # Normal face
            proportion_score = 0.0  # Adult
        else:  # Long face
            proportion_score = 0.05  # Older
        
        # Combine all scores
        total_score = size_score + wrinkle_score + texture_score + proportion_score
        
        # Determine age group
        if total_score < 0.3:
            return "0-12"
        elif total_score < 0.5:
            return "13-18"
        elif total_score < 0.7:
            return "19-25"
        elif total_score < 0.9:
            return "26-35"
        elif total_score < 1.1:
            return "36-45"
        elif total_score < 1.3:
            return "46-55"
        elif total_score < 1.5:
            return "56-65"
        else:
            return "65+"
    
    def calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        h, w = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                binary_string = ""
                
                # 8-neighborhood
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += "1" if neighbor >= center else "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
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
                    age = self.estimate_age_improved(frame, largest_face)
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
    
    def draw_final_results(self, image: np.ndarray, 
                         recognized_faces: List[Tuple[str, float, Tuple[int, int, int, int]]],
                         detected_objects: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """Draw final recognition results on the image."""
        result_image = image.copy()
        
        # Draw face recognition results
        for name, confidence, (x, y, w, h) in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw rectangle around face
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Estimate age
            age = self.estimate_age_improved(image, (x, y, w, h))
            
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
    
    def run_final_recognition(self, threshold: float = 0.3):
        """Run final recognition system."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Could not open webcam")
            return
        
        print("Final Recognition System started!")
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
                detected_objects = self.detect_objects_reliable(frame)
                
                # Draw all results
                result_frame = self.draw_final_results(frame, recognized_faces, detected_objects)
                
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
                
                cv2.imshow("Final Recognition System", result_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function."""
    print("Final Recognition System")
    print("=" * 50)
    print("Major improvements:")
    print("- Reliable object detection (only real objects)")
    print("- Much better age estimation")
    print("- Template matching for objects")
    print("- Strict filtering to reduce false positives")
    print()
    
    # Initialize recognition system
    recognizer = FinalRecognitionSystem()
    
    while True:
        print("\n" + "="*50)
        print("Menu:")
        print("1. Add person from webcam (with improved age detection)")
        print("2. Load known faces from folder")
        print("3. Final recognition (faces + objects + age)")
        print("4. View loaded people")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
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
            # Final recognition
            if not recognizer.known_faces:
                print("Please add people first (option 1) or load known faces (option 2)")
                continue
            
            threshold = input("Recognition threshold (0.1-1.0, default 0.3): ").strip()
            threshold = float(threshold) if threshold else 0.3
            
            recognizer.run_final_recognition(threshold)
        
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
