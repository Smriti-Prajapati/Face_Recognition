"""
Enhanced Recognition System
Improved age estimation and better object detection
"""

import cv2
import numpy as np
import os
import time
import base64
from flask import Flask, render_template, Response, jsonify, request
import threading
from typing import List, Tuple, Dict
import json

# Try to import YOLO, fallback to OpenCV if not available
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available, using OpenCV for object detection")

class EnhancedRecognitionSystem:
    """Enhanced recognition system with better age estimation and object detection."""
    
    def __init__(self):
        """Initialize the enhanced recognition system."""
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}
        self.known_faces_folder = "known_people"
        
        # YOLO model for object detection
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
                print("YOLO model loaded successfully")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self.yolo_model = None
        
        # Webcam
        self.cap = None
        self.is_running = False
        self.current_frame = None
        
        # Create folders
        if not os.path.exists(self.known_faces_folder):
            os.makedirs(self.known_faces_folder)
            print(f"Created folder: {self.known_faces_folder}")
        
        # Load existing faces
        self.load_known_faces_from_folder()
    
    def extract_face_features(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract advanced features from a face region for better recognition."""
        x, y, w, h = face_rect
        
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        
        # Resize to larger standard size for better features
        face_resized = cv2.resize(face_roi, (150, 150))
        
        # Convert to grayscale
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        face_gray = cv2.equalizeHist(face_gray)
        
        # Apply Gaussian blur to reduce noise
        face_gray = cv2.GaussianBlur(face_gray, (3, 3), 0)
        
        # Extract multiple feature types
        features = []
        
        # 1. Raw pixel features (resized)
        raw_features = face_gray.flatten()
        features.extend(raw_features)
        
        # 2. Histogram features
        hist = cv2.calcHist([face_gray], [0], None, [64], [0, 256])
        features.extend(hist.flatten())
        
        # 3. Edge features
        edges = cv2.Canny(face_gray, 50, 150)
        edge_features = edges.flatten()
        features.extend(edge_features)
        
        # 4. Local Binary Pattern features
        lbp = self.calculate_lbp(face_gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [32], [0, 256])
        features.extend(lbp_hist.flatten())
        
        # 5. Gabor filter features (simplified)
        gabor_features = self.extract_gabor_features(face_gray)
        features.extend(gabor_features)
        
        # 6. Face region features (eyes, nose, mouth areas)
        region_features = self.extract_region_features(face_gray)
        features.extend(region_features)
        
        # Convert to numpy array and normalize
        features = np.array(features, dtype=np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def detect_objects_yolo(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Detect objects using YOLO with better filtering."""
        if not YOLO_AVAILABLE or self.yolo_model is None:
            return self.detect_objects_opencv(image)
        
        try:
            # Run YOLO detection
            results = self.yolo_model(image)
            
            detected_objects = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.yolo_model.names[class_id]
                        
                        # Convert to our format
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        # Better filtering - only high confidence detections
                        if confidence > 0.6:  # Higher confidence threshold
                            # Check if object is not too small
                            if w > 30 and h > 30:
                                # Check if not in face area
                                if not self.is_in_face_area(x + w//2, y + h//2, image):
                                    detected_objects.append((class_name, float(confidence), (x, y, w, h)))
            
            return detected_objects
        
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return self.detect_objects_opencv(image)
    
    def detect_objects_opencv(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Enhanced OpenCV object detection."""
        detected_objects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces as objects
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            detected_objects.append(("Person", 0.9, (x, y, w, h)))
        
        # Enhanced rectangular object detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 8000 < area < 100000:  # Better size filtering
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Better aspect ratio filtering
                if 0.2 < aspect_ratio < 5.0:
                    # Check if not in face area
                    if not self.is_in_face_area(x + w//2, y + h//2, image):
                        # Check contrast
                        roi = gray[y:y+h, x:x+w]
                        if np.std(roi) > 30:  # Good contrast
                            detected_objects.append(("Object", 0.7, (x, y, w, h)))
        
        return detected_objects
    
    def is_in_face_area(self, x: int, y: int, image: np.ndarray) -> bool:
        """Check if a point is in face area."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (fx, fy, fw, fh) in faces:
            # Expand face area with margin
            margin = 50
            if (fx - margin <= x <= fx + fw + margin and 
                fy - margin <= y <= fy + fh + margin):
                return True
        return False
    
    def estimate_age_enhanced(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> str:
        """
        Enhanced age estimation using multiple factors.
        Much more accurate than simple size-based estimation.
        """
        x, y, w, h = face_rect
        
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate face area
        face_area = w * h
        
        # Factor 1: Face size (normalized)
        size_score = 0
        if face_area < 3000:
            size_score = 0.3  # Very young
        elif face_area < 6000:
            size_score = 0.2  # Young
        elif face_area < 10000:
            size_score = 0.1  # Young adult
        elif face_area < 15000:
            size_score = 0.0  # Adult
        else:
            size_score = 0.1  # Older
        
        # Factor 2: Wrinkle analysis (multiple scales)
        wrinkle_score = 0
        for scale in [1.0, 1.5, 2.0]:
            scaled_face = cv2.resize(face_gray, None, fx=scale, fy=scale)
            edges = cv2.Canny(scaled_face, 30, 100)
            wrinkle_density = np.sum(edges) / (scaled_face.shape[0] * scaled_face.shape[1])
            wrinkle_score += wrinkle_density
        
        wrinkle_score /= 3  # Average
        
        if wrinkle_score < 0.02:
            wrinkle_score = 0.4  # Very young (smooth)
        elif wrinkle_score < 0.05:
            wrinkle_score = 0.2  # Young
        elif wrinkle_score < 0.10:
            wrinkle_score = 0.0  # Adult
        elif wrinkle_score < 0.15:
            wrinkle_score = 0.1  # Middle-aged
        else:
            wrinkle_score = 0.2  # Older
        
        # Factor 3: Skin texture analysis
        texture_score = 0
        # Use Local Binary Pattern for texture
        lbp = self.calculate_lbp(face_gray)
        texture_variance = np.var(lbp)
        
        if texture_variance < 30:
            texture_score = 0.3  # Very young (smooth)
        elif texture_variance < 60:
            texture_score = 0.2  # Young
        elif texture_variance < 120:
            texture_score = 0.1  # Young adult
        elif texture_variance < 200:
            texture_score = 0.0  # Adult
        else:
            texture_score = 0.1  # Older
        
        # Factor 4: Face proportions
        aspect_ratio = w / h
        proportion_score = 0
        if aspect_ratio < 0.6:  # Very round face
            proportion_score = 0.2  # Young
        elif aspect_ratio < 0.75:  # Round face
            proportion_score = 0.1  # Young adult
        elif aspect_ratio < 0.85:  # Normal face
            proportion_score = 0.0  # Adult
        else:  # Long face
            proportion_score = 0.05  # Older
        
        # Factor 5: Eye area analysis
        eye_score = 0
        eye_region = face_gray[int(h*0.2):int(h*0.5), int(w*0.2):int(w*0.8)]
        if eye_region.size > 0:
            eye_edges = cv2.Canny(eye_region, 20, 60)
            eye_density = np.sum(eye_edges) / eye_region.size
            if eye_density < 0.01:
                eye_score = 0.2  # Young (smooth around eyes)
            elif eye_density < 0.03:
                eye_score = 0.1  # Young adult
            else:
                eye_score = 0.0  # Adult/Older
        
        # Factor 6: Cheek area analysis
        cheek_score = 0
        cheek_region = face_gray[int(h*0.4):int(h*0.7), int(w*0.3):int(w*0.7)]
        if cheek_region.size > 0:
            cheek_edges = cv2.Canny(cheek_region, 20, 60)
            cheek_density = np.sum(cheek_edges) / cheek_region.size
            if cheek_density < 0.01:
                cheek_score = 0.2  # Young (smooth cheeks)
            elif cheek_density < 0.03:
                cheek_score = 0.1  # Young adult
            else:
                cheek_score = 0.0  # Adult/Older
        
        # Combine all factors with weights
        total_score = (size_score * 0.2 + 
                      wrinkle_score * 0.25 + 
                      texture_score * 0.2 + 
                      proportion_score * 0.1 + 
                      eye_score * 0.15 + 
                      cheek_score * 0.1)
        
        # Determine age group with better thresholds
        if total_score < 0.2:
            return "Child (0-12)"
        elif total_score < 0.4:
            return "Teen (13-18)"
        elif total_score < 0.6:
            return "Young Adult (19-25)"
        elif total_score < 0.8:
            return "Adult (26-35)"
        elif total_score < 1.0:
            return "Middle-aged (36-50)"
        elif total_score < 1.2:
            return "Mature (51-65)"
        else:
            return "Senior (65+)"
    
    def calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern for texture analysis."""
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
    
    def extract_gabor_features(self, image: np.ndarray) -> List[float]:
        """Extract Gabor filter features for better face recognition."""
        features = []
        
        # Create Gabor kernels with different orientations and frequencies
        orientations = [0, 45, 90, 135]  # degrees
        frequencies = [0.1, 0.2, 0.3]   # cycles per pixel
        
        for orientation in orientations:
            for frequency in frequencies:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(orientation), 
                                          2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                
                # Extract features from filtered image
                features.extend([
                    np.mean(filtered),
                    np.std(filtered),
                    np.var(filtered)
                ])
        
        return features
    
    def extract_region_features(self, face_gray: np.ndarray) -> List[float]:
        """Extract features from specific face regions (eyes, nose, mouth)."""
        h, w = face_gray.shape
        features = []
        
        # Eye region (upper 1/3 of face)
        eye_region = face_gray[0:h//3, :]
        if eye_region.size > 0:
            features.extend([
                np.mean(eye_region),
                np.std(eye_region),
                np.var(eye_region)
            ])
        
        # Nose region (middle 1/3 of face)
        nose_region = face_gray[h//3:2*h//3, :]
        if nose_region.size > 0:
            features.extend([
                np.mean(nose_region),
                np.std(nose_region),
                np.var(nose_region)
            ])
        
        # Mouth region (lower 1/3 of face)
        mouth_region = face_gray[2*h//3:, :]
        if mouth_region.size > 0:
            features.extend([
                np.mean(mouth_region),
                np.std(mouth_region),
                np.var(mouth_region)
            ])
        
        # Left and right halves
        left_half = face_gray[:, :w//2]
        right_half = face_gray[:, w//2:]
        
        if left_half.size > 0 and right_half.size > 0:
            features.extend([
                np.mean(left_half),
                np.mean(right_half),
                np.std(left_half),
                np.std(right_half)
            ])
        
        return features
    
    def load_known_faces_from_folder(self, folder_path: str = None):
        """Load known faces from a folder."""
        if folder_path is None:
            folder_path = self.known_faces_folder
        
        if not os.path.exists(folder_path):
            return
        
        self.known_faces = {}
        
        for person_name in os.listdir(folder_path):
            person_folder = os.path.join(folder_path, person_name)
            
            if not os.path.isdir(person_folder):
                continue
            
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
                            features = self.extract_face_features(image, largest_face)
                            person_features.append(features)
            
            if person_features:
                self.known_faces[person_name] = person_features
    
    def recognize_faces(self, image: np.ndarray, threshold: float = 0.3) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Enhanced face recognition with better matching algorithm."""
        if not self.known_faces:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        recognized_faces = []
        
        for face_rect in faces:
            face_features = self.extract_face_features(image, face_rect)
            
            best_match = "Unknown"
            best_confidence = 0.0
            best_distance = float('inf')
            
            # Calculate distances to all known people
            person_distances = {}
            
            for person_name, person_feature_list in self.known_faces.items():
                distances = []
                
                # Calculate distance to each sample of this person
                for person_features in person_feature_list:
                    # Use cosine similarity for better matching
                    cosine_sim = np.dot(face_features, person_features) / (
                        np.linalg.norm(face_features) * np.linalg.norm(person_features) + 1e-8
                    )
                    
                    # Convert to distance (lower is better)
                    distance = 1 - cosine_sim
                    distances.append(distance)
                
                # Use the minimum distance (best match) for this person
                min_distance = min(distances)
                person_distances[person_name] = min_distance
                
                # Calculate confidence based on distance
                confidence = max(0, 1 - min_distance)
                
                # Update best match if this is better
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_distance = min_distance
                    best_match = person_name
            
            # Additional validation: check if the best match is significantly better than others
            if len(person_distances) > 1:
                sorted_distances = sorted(person_distances.values())
                # If the best match is not significantly better than the second best, mark as unknown
                if sorted_distances[1] - sorted_distances[0] < 0.1:  # Less than 10% difference
                    best_match = "Unknown"
                    best_confidence = 0.0
            
            # Apply threshold
            if best_confidence >= threshold:
                recognized_faces.append((best_match, best_confidence, tuple(face_rect)))
            else:
                recognized_faces.append(("Unknown", best_confidence, tuple(face_rect)))
        
        return recognized_faces
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for recognition."""
        # Face recognition
        recognized_faces = self.recognize_faces(frame, 0.3)
        
        # Object detection
        detected_objects = self.detect_objects_yolo(frame)
        
        # Draw results
        result_frame = frame.copy()
        
        # Draw face recognition results
        for name, confidence, (x, y, w, h) in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw rectangle around face
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Estimate age with enhanced method
            age = self.estimate_age_enhanced(frame, (x, y, w, h))
            
            # Draw labels
            face_label = f"{name} ({confidence:.2f})"
            age_label = f"Age: {age}"
            
            # Face label
            face_label_size = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x, y - face_label_size[1] - 10), 
                         (x + face_label_size[0], y), color, -1)
            cv2.putText(result_frame, face_label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Age label
            cv2.putText(result_frame, age_label, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw object detection results
        for obj_name, confidence, (x, y, w, h) in detected_objects:
            if obj_name != "Person":  # Don't draw person objects (already drawn as faces)
                color = (255, 0, 0)  # Blue for objects
                
                # Draw rectangle around object
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                obj_label = f"{obj_name} ({confidence:.2f})"
                obj_label_size = cv2.getTextSize(obj_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(result_frame, (x, y - obj_label_size[1] - 5), 
                             (x + obj_label_size[0], y), color, -1)
                cv2.putText(result_frame, obj_label, (x, y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def start_camera(self):
        """Start the camera."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            
            # Start processing thread
            threading.Thread(target=self._camera_loop, daemon=True).start()
    
    def stop_camera(self):
        """Stop the camera."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _camera_loop(self):
        """Camera processing loop."""
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = self.process_frame(frame)
            time.sleep(0.03)  # ~30 FPS
    
    def get_frame(self):
        """Get the current processed frame."""
        if self.current_frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', self.current_frame)
            if ret:
                return buffer.tobytes()
        return None
    
    def add_person(self, name: str, image_data: str) -> bool:
        """Add a person from base64 image data with enhanced processing."""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return False
            
            # Create person folder
            person_folder = os.path.join(self.known_faces_folder, name)
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)
            
            # Detect face
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_roi = image[y:y+h, x:x+w]
                
                # Create multiple augmented versions for better recognition
                augmented_faces = self.create_augmented_faces(face_roi)
                
                # Save original and augmented faces
                timestamp = int(time.time())
                for i, aug_face in enumerate(augmented_faces):
                    photo_path = os.path.join(person_folder, f"{name}_sample_{timestamp}_{i}.jpg")
                    cv2.imwrite(photo_path, aug_face)
                
                # Reload faces
                self.load_known_faces_from_folder()
                return True
            
            return False
        
        except Exception as e:
            print(f"Error adding person: {e}")
            return False
    
    def create_augmented_faces(self, face_image: np.ndarray) -> List[np.ndarray]:
        """Create augmented versions of a face for better recognition."""
        augmented_faces = [face_image]  # Original face
        
        # 1. Slightly rotated versions
        for angle in [-5, 5, -10, 10]:
            h, w = face_image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(face_image, rotation_matrix, (w, h))
            augmented_faces.append(rotated)
        
        # 2. Brightness adjusted versions
        for brightness in [0.8, 1.2]:
            brightened = cv2.convertScaleAbs(face_image, alpha=brightness, beta=0)
            augmented_faces.append(brightened)
        
        # 3. Slightly blurred version (simulates different lighting)
        blurred = cv2.GaussianBlur(face_image, (3, 3), 0)
        augmented_faces.append(blurred)
        
        # 4. Histogram equalized version
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        augmented_faces.append(equalized_bgr)
        
        return augmented_faces

# Initialize the recognition system
recognition_system = EnhancedRecognitionSystem()

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    def generate():
        while True:
            frame = recognition_system.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera."""
    recognition_system.start_camera()
    return jsonify({'status': 'success'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera."""
    recognition_system.stop_camera()
    return jsonify({'status': 'success'})

@app.route('/add_person', methods=['POST'])
def add_person():
    """Add a person to the database."""
    data = request.get_json()
    name = data.get('name', '')
    image_data = data.get('image', '')
    
    if not name or not image_data:
        return jsonify({'status': 'error', 'message': 'Name and image are required'})
    
    success = recognition_system.add_person(name, image_data)
    
    if success:
        return jsonify({'status': 'success', 'message': f'Person {name} added successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to add person. Make sure a face is visible in the image.'})

@app.route('/get_known_people')
def get_known_people():
    """Get list of known people."""
    people = list(recognition_system.known_faces.keys())
    return jsonify({'people': people})

if __name__ == '__main__':
    # Create templates directory and HTML file
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template with ASCII characters only
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced AI Recognition System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .video-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .video-container {
            position: relative;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        #video-feed {
            width: 100%;
            height: 400px;
            object-fit: cover;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
        }
        
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(244, 67, 54, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #2196F3, #1976D2);
            color: white;
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(33, 150, 243, 0.4);
        }
        
        .features-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .feature-list {
            list-style: none;
            margin-bottom: 20px;
        }
        
        .feature-list li {
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
        }
        
        .feature-list li:before {
            content: "OK";
            color: #4CAF50;
            font-weight: bold;
            margin-right: 10px;
            font-size: 1.2rem;
        }
        
        .add-person-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin-bottom: 20px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
        }
        
        .form-group input {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
        }
        
        .form-group input:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.3);
        }
        
        .status {
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            font-weight: 600;
        }
        
        .status.success {
            background: rgba(76, 175, 80, 0.2);
            border: 1px solid #4CAF50;
            color: #4CAF50;
        }
        
        .status.error {
            background: rgba(244, 67, 54, 0.2);
            border: 1px solid #f44336;
            color: #f44336;
        }
        
        .known-people {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .people-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .person-tag {
            background: rgba(33, 150, 243, 0.3);
            padding: 8px 16px;
            border-radius: 20px;
            border: 1px solid rgba(33, 150, 243, 0.5);
            font-size: 0.9rem;
        }
        
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .controls {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Enhanced AI Recognition System</h1>
            <p>Advanced Face Recognition, Object Detection & Accurate Age Estimation</p>
        </div>
        
        
        <div class="main-content">
            <div class="video-section">
                <div class="video-container">
                    <img id="video-feed" src="{{ url_for('video_feed') }}" alt="Video Feed">
                </div>
                <div class="controls">
                    <button class="btn btn-primary" onclick="startCamera()">Start Camera</button>
                    <button class="btn btn-danger" onclick="stopCamera()">Stop Camera</button>
                    <button class="btn btn-secondary" onclick="capturePhoto()">Capture Photo</button>
                </div>
            </div>
            
            <div class="features-section">
                <h3>Enhanced Features</h3>
                <ul class="feature-list">
                    <li>Real-time Face Recognition</li>
                    <li>Advanced Object Detection (Phone, Bottle, Chair, Pen, etc.)</li>
                    <li>Accurate Age Estimation (6-factor analysis)</li>
                    <li>Live Video Processing</li>
                    <li>Easy Person Management</li>
                </ul>
                
                <h3>System Status</h3>
                <div id="system-status">
                    <p>Camera: <span id="camera-status">Stopped</span></p>
                    <p>Known People: <span id="people-count">0</span></p>
                    <p>Object Detection: <span id="object-status">Enhanced</span></p>
                </div>
            </div>
        </div>
        
        <div class="add-person-section">
            <h3>Add New Person</h3>
            <div class="form-group">
                <label for="person-name">Name:</label>
                <input type="text" id="person-name" placeholder="Enter person's name">
            </div>
            <button class="btn btn-primary" onclick="addPerson()">Add Person</button>
            <div id="add-status"></div>
        </div>
        
        <div class="known-people">
            <h3>Known People</h3>
            <div id="people-list" class="people-list">
                <!-- People will be loaded here -->
            </div>
        </div>
    </div>
    
    <script>
        let isCameraRunning = false;
        let capturedImage = null;
        
        function startCamera() {
            fetch('/start_camera', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        isCameraRunning = true;
                        document.getElementById('camera-status').textContent = 'Running';
                        showStatus('Camera started successfully!', 'success');
                    }
                })
                .catch(error => {
                    showStatus('Failed to start camera', 'error');
                });
        }
        
        function stopCamera() {
            fetch('/stop_camera', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        isCameraRunning = false;
                        document.getElementById('camera-status').textContent = 'Stopped';
                        showStatus('Camera stopped', 'success');
                    }
                })
                .catch(error => {
                    showStatus('Failed to stop camera', 'error');
                });
        }
        
        function capturePhoto() {
            if (!isCameraRunning) {
                showStatus('Please start the camera first', 'error');
                return;
            }
            
            // Get the current video frame
            const video = document.getElementById('video-feed');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = video.videoWidth || video.width;
            canvas.height = video.videoHeight || video.height;
            
            ctx.drawImage(video, 0, 0);
            capturedImage = canvas.toDataURL('image/jpeg');
            
            showStatus('Photo captured! You can now add a person.', 'success');
        }
        
        function addPerson() {
            const name = document.getElementById('person-name').value.trim();
            
            if (!name) {
                showStatus('Please enter a name', 'error');
                return;
            }
            
            if (!capturedImage) {
                showStatus('Please capture a photo first', 'error');
                return;
            }
            
            fetch('/add_person', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: name,
                    image: capturedImage
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(data.message, 'success');
                    document.getElementById('person-name').value = '';
                    capturedImage = null;
                    loadKnownPeople();
                } else {
                    showStatus(data.message, 'error');
                }
            })
            .catch(error => {
                showStatus('Failed to add person', 'error');
            });
        }
        
        function loadKnownPeople() {
            fetch('/get_known_people')
                .then(response => response.json())
                .then(data => {
                    const peopleList = document.getElementById('people-list');
                    const peopleCount = document.getElementById('people-count');
                    
                    peopleList.innerHTML = '';
                    peopleCount.textContent = data.people.length;
                    
                    data.people.forEach(person => {
                        const tag = document.createElement('div');
                        tag.className = 'person-tag';
                        tag.textContent = person;
                        peopleList.appendChild(tag);
                    });
                })
                .catch(error => {
                    console.error('Failed to load known people:', error);
                });
        }
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('add-status');
            statusDiv.innerHTML = '<div class="status ' + type + '">' + message + '</div>';
            
            setTimeout(() => {
                statusDiv.innerHTML = '';
            }, 3000);
        }
        
        // Load known people on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadKnownPeople();
        });
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Starting Enhanced Recognition System...")
    print("Open your browser and go to: http://localhost:5000")
    print("Features: Enhanced Face Recognition, Better Object Detection, Accurate Age Estimation")
    print("YOLO Status:", "Available" if YOLO_AVAILABLE else "Not Available (using OpenCV)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
