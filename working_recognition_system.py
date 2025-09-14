"""
Working Recognition System
Simplified but reliable face recognition
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

class WorkingRecognitionSystem:
    """Working recognition system with reliable face recognition."""
    
    def __init__(self):
        """Initialize the working recognition system."""
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
    
    def extract_face_features_simple(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract simple but reliable features from a face region."""
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
    
    def detect_objects_yolo(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Detect objects using YOLO."""
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
                        
                        # Filter out low confidence detections
                        if confidence > 0.5:
                            detected_objects.append((class_name, float(confidence), (x, y, w, h)))
            
            return detected_objects
        
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return self.detect_objects_opencv(image)
    
    def detect_objects_opencv(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Fallback object detection using OpenCV."""
        detected_objects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces as objects
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            detected_objects.append(("Person", 0.9, (x, y, w, h)))
        
        return detected_objects
    
    def estimate_age(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> str:
        """Simple but effective age estimation."""
        x, y, w, h = face_rect
        face_area = w * h
        
        # Extract face region for analysis
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistent analysis
        face_resized = cv2.resize(face_gray, (100, 100))
        
        # Factor 1: Face size relative to image
        image_area = image.shape[0] * image.shape[1]
        size_ratio = face_area / image_area
        
        # Factor 2: Simple wrinkle detection
        edges = cv2.Canny(face_resized, 50, 150)
        wrinkle_score = np.sum(edges) / (100 * 100)
        
        # Factor 3: Face shape (rounder = younger)
        aspect_ratio = w / h
        
        # Factor 4: Eye area smoothness
        eye_region = face_resized[20:50, 20:80]
        if eye_region.size > 0:
            eye_edges = cv2.Canny(eye_region, 30, 100)
            eye_smoothness = 1 - (np.sum(eye_edges) / eye_region.size)
        else:
            eye_smoothness = 0.5
        
        # Factor 5: Overall skin texture
        laplacian_var = cv2.Laplacian(face_resized, cv2.CV_64F).var()
        texture_score = min(laplacian_var / 1000, 1.0)  # Normalize to 0-1
        
        # Calculate age based on simple rules
        age_points = 0
        
        # Size factor (larger face = older)
        if size_ratio > 0.12:
            age_points += 3
        elif size_ratio > 0.08:
            age_points += 2
        elif size_ratio > 0.05:
            age_points += 1
        
        # Wrinkle factor
        if wrinkle_score > 0.08:
            age_points += 4
        elif wrinkle_score > 0.05:
            age_points += 3
        elif wrinkle_score > 0.03:
            age_points += 2
        elif wrinkle_score > 0.01:
            age_points += 1
        
        # Texture factor (more texture = older)
        if texture_score > 0.7:
            age_points += 2
        elif texture_score > 0.4:
            age_points += 1
        
        # Eye smoothness factor
        if eye_smoothness < 0.3:
            age_points += 2
        elif eye_smoothness < 0.5:
            age_points += 1
        
        # Face shape factor (rounder = younger)
        if aspect_ratio > 0.85:
            age_points -= 1  # Rounder face, younger
        elif aspect_ratio < 0.65:
            age_points += 1  # Longer face, older
        
        # Determine age group
        if age_points <= 1:
            return "Child (0-12)"
        elif age_points <= 3:
            return "Teen (13-18)"
        elif age_points <= 5:
            return "Young Adult (19-25)"
        elif age_points <= 7:
            return "Adult (26-35)"
        elif age_points <= 9:
            return "Middle-aged (36-50)"
        elif age_points <= 11:
            return "Mature (51-65)"
        else:
            return "Senior (65+)"
    
    def detect_gender(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> str:
        """Enhanced gender detection based on multiple facial features."""
        x, y, w, h = face_rect
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistent analysis
        face_resized = cv2.resize(face_gray, (100, 100))
        
        # Factor 1: Face shape analysis
        aspect_ratio = w / h
        
        # Factor 2: Jawline analysis (lower 1/3 of face)
        jaw_region = face_resized[65:100, 20:80]
        if jaw_region.size > 0:
            jaw_edges = cv2.Canny(jaw_region, 30, 100)
            jaw_angularity = np.sum(jaw_edges)
        else:
            jaw_angularity = 0
        
        # Factor 3: Cheekbone analysis (middle region)
        cheek_region = face_resized[40:70, 25:75]
        if cheek_region.size > 0:
            cheek_edges = cv2.Canny(cheek_region, 30, 100)
            cheek_prominence = np.sum(cheek_edges)
        else:
            cheek_prominence = 0
        
        # Factor 4: Forehead analysis (upper 1/3)
        forehead_region = face_resized[10:40, 30:70]
        if forehead_region.size > 0:
            forehead_edges = cv2.Canny(forehead_region, 30, 100)
            forehead_width = np.sum(forehead_edges)
        else:
            forehead_width = 0
        
        # Calculate gender score
        gender_score = 0
        
        # Face shape factor (rounder = more female)
        if aspect_ratio > 0.85:
            gender_score += 0.3  # Rounder face
        elif aspect_ratio < 0.7:
            gender_score -= 0.2  # More rectangular face
        
        # Jawline factor (more angular = more male)
        if jaw_angularity > 200:
            gender_score -= 0.3  # Angular jawline
        elif jaw_angularity < 100:
            gender_score += 0.2  # Softer jawline
        
        # Cheekbone factor (more prominent = more male)
        if cheek_prominence > 150:
            gender_score -= 0.2
        elif cheek_prominence < 80:
            gender_score += 0.1
        
        # Forehead factor (wider = more male)
        if forehead_width > 120:
            gender_score -= 0.2
        elif forehead_width < 80:
            gender_score += 0.1
        
        # Determine gender based on score
        if gender_score > 0.1:
            return "Female"
        else:
            return "Male"
    
    def detect_emotion(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> str:
        """Detect emotion based on facial features."""
        x, y, w, h = face_rect
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for analysis
        face_resized = cv2.resize(face_gray, (100, 100))
        
        # Analyze mouth region (lower 1/3)
        mouth_region = face_resized[65:100, 25:75]
        if mouth_region.size > 0:
            mouth_edges = cv2.Canny(mouth_region, 30, 100)
            mouth_curvature = np.sum(mouth_edges)
            
            # Analyze eye region (upper 1/3)
            eye_region = face_resized[20:50, 20:80]
            if eye_region.size > 0:
                eye_edges = cv2.Canny(eye_region, 30, 100)
                eye_activity = np.sum(eye_edges)
                
                # Simple emotion detection
                if mouth_curvature > 200 and eye_activity > 150:
                    return "Happy"
                elif mouth_curvature < 100 and eye_activity < 100:
                    return "Neutral"
                elif mouth_curvature < 50:
                    return "Sad"
                else:
                    return "Surprised"
        
        return "Neutral"
    
    def detect_glasses(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> str:
        """Detect if person is wearing glasses."""
        x, y, w, h = face_rect
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Focus on eye region
        eye_region = face_gray[int(h*0.2):int(h*0.5), int(w*0.2):int(w*0.8)]
        
        if eye_region.size > 0:
            # Look for horizontal lines (glasses frames)
            edges = cv2.Canny(eye_region, 50, 150)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            
            if np.sum(horizontal_lines) > 100:
                return "With Glasses"
        
        return "No Glasses"
    
    def detect_face_orientation(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> str:
        """Detect face orientation."""
        x, y, w, h = face_rect
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Analyze face symmetry
        left_half = face_gray[:, :w//2]
        right_half = face_gray[:, w//2:]
        
        if left_half.size > 0 and right_half.size > 0:
            left_mean = np.mean(left_half)
            right_mean = np.mean(right_half)
            
            diff = abs(left_mean - right_mean)
            
            if diff < 10:
                return "Front"
            elif left_mean > right_mean:
                return "Left"
            else:
                return "Right"
        
        return "Front"
    
    def load_known_faces_from_folder(self, folder_path: str = None):
        """Load known faces from a folder."""
        if folder_path is None:
            folder_path = self.known_faces_folder
        
        if not os.path.exists(folder_path):
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
                            print(f"  Loaded {filename}")
                        else:
                            print(f"  No face found in {filename}")
            
            if person_features:
                self.known_faces[person_name] = person_features
                print(f"  Added {person_name} with {len(person_features)} face samples")
            else:
                print(f"  No valid faces found for {person_name}")
        
        print(f"Loaded {len(self.known_faces)} people")
    
    def recognize_faces(self, image: np.ndarray, threshold: float = 0.3) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Simple but reliable face recognition."""
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
                    # Simple Euclidean distance
                    distance = np.linalg.norm(face_features - person_features)
                    min_distance = min(min_distance, distance)
                
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_match = person_name
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = max(0, 1 - best_distance)
            
            if confidence >= threshold:
                recognized_faces.append((best_match, confidence, tuple(face_rect)))
            else:
                recognized_faces.append(("Unknown", confidence, tuple(face_rect)))
        
        return recognized_faces
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for recognition with enhanced features."""
        # Face recognition
        recognized_faces = self.recognize_faces(frame, 0.2)  # Lower threshold
        
        # Object detection
        detected_objects = self.detect_objects_yolo(frame)
        
        # Draw results
        result_frame = frame.copy()
        
        # Draw face recognition results with enhanced features
        for name, confidence, (x, y, w, h) in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw rectangle around face
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Get all face analysis features
            age = self.estimate_age(frame, (x, y, w, h))
            gender = self.detect_gender(frame, (x, y, w, h))
            emotion = self.detect_emotion(frame, (x, y, w, h))
            glasses = self.detect_glasses(frame, (x, y, w, h))
            orientation = self.detect_face_orientation(frame, (x, y, w, h))
            
            # Draw main label
            face_label = f"{name} ({confidence:.2f})"
            face_label_size = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x, y - face_label_size[1] - 10), 
                         (x + face_label_size[0], y), color, -1)
            cv2.putText(result_frame, face_label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw feature labels in a column
            labels = [
                f"Age: {age}",
                f"Gender: {gender}",
                f"Emotion: {emotion}",
                f"Glasses: {glasses}",
                f"Orientation: {orientation}"
            ]
            
            y_offset = y + h + 15
            for i, label in enumerate(labels):
                # Choose color based on feature type
                if "Age:" in label:
                    label_color = (255, 255, 0)  # Yellow
                elif "Gender:" in label:
                    label_color = (255, 0, 255)  # Magenta
                elif "Emotion:" in label:
                    if "Happy" in label:
                        label_color = (0, 255, 0)  # Green
                    elif "Sad" in label:
                        label_color = (0, 0, 255)  # Red
                    else:
                        label_color = (255, 255, 255)  # White
                elif "Glasses:" in label:
                    label_color = (0, 255, 255)  # Cyan
                else:
                    label_color = (128, 128, 128)  # Gray
                
                cv2.putText(result_frame, label, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
                y_offset += 15
        
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
        """Add a person from base64 image data."""
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
                # Save image
                photo_path = os.path.join(person_folder, f"{name}_sample_{int(time.time())}.jpg")
                cv2.imwrite(photo_path, image)
                
                # Reload faces
                self.load_known_faces_from_folder()
                return True
            
            return False
        
        except Exception as e:
            print(f"Error adding person: {e}")
            return False

# Initialize the recognition system
recognition_system = WorkingRecognitionSystem()

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
    
    # Create the HTML template
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Recognition System</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            position: relative;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: -20px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: linear-gradient(90deg, #4CAF50, #2196F3, #FF9800);
            border-radius: 2px;
        }
        
        .header h1 {
            font-size: 3.5rem;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            background: linear-gradient(45deg, #fff, #e0e0e0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            font-size: 1.3rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .video-section {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 25px;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .video-container {
            position: relative;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 25px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            border: 2px solid rgba(255, 255, 255, 0.1);
        }
        
        #video-feed {
            width: 100%;
            height: 450px;
            object-fit: cover;
            transition: transform 0.3s ease;
        }
        
        .video-container:hover #video-feed {
            transform: scale(1.02);
        }
        
        .video-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(to bottom, rgba(0,0,0,0.3) 0%, transparent 30%, transparent 70%, rgba(0,0,0,0.3) 100%);
            display: flex;
            align-items: flex-end;
            justify-content: center;
            padding: 20px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .video-container:hover .video-overlay {
            opacity: 1;
        }
        
        .camera-controls {
            display: flex;
            gap: 10px;
        }
        
        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 50px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .btn:hover::before {
            left: 100%;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        }
        
        .btn-danger:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(244, 67, 54, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #2196F3, #1976D2);
            color: white;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        }
        
        .btn-secondary:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(33, 150, 243, 0.4);
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 25px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 25px;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h3 {
            color: #fff;
            margin-bottom: 20px;
            font-size: 1.4rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .feature-list {
            list-style: none;
            margin-bottom: 20px;
        }
        
        .feature-list li {
            padding: 12px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            transition: all 0.3s ease;
        }
        
        .feature-list li:hover {
            padding-left: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            margin: 0 -10px;
            padding-right: 10px;
        }
        
        .feature-list li:before {
            content: "✓";
            color: #4CAF50;
            font-weight: bold;
            margin-right: 12px;
            font-size: 1.2rem;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }
        
        .status-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .status-item .label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .status-item .value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #4CAF50;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #fff;
        }
        
        .form-group input {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            transition: all 0.3s ease;
        }
        
        .form-group input:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.3);
            transform: translateY(-2px);
        }
        
        .status {
            padding: 15px;
            border-radius: 12px;
            margin: 15px 0;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
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
        
        .people-list {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        
        .person-tag {
            background: linear-gradient(45deg, rgba(33, 150, 243, 0.3), rgba(33, 150, 243, 0.1));
            padding: 10px 20px;
            border-radius: 25px;
            border: 1px solid rgba(33, 150, 243, 0.5);
            font-size: 0.9rem;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .person-tag:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
        }
        
        .debug-info {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 152, 0, 0.1));
            border: 1px solid rgba(255, 193, 7, 0.3);
        }
        
        .debug-info h3 {
            color: #FFD700;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .debug-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .debug-stat {
            background: rgba(255, 255, 255, 0.05);
            padding: 12px;
            border-radius: 10px;
            text-align: center;
        }
        
        .debug-stat .label {
            font-size: 0.8rem;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .debug-stat .value {
            font-size: 1rem;
            font-weight: 600;
            color: #FFD700;
        }
        
        .features-showcase {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1));
            border: 1px solid rgba(76, 175, 80, 0.3);
        }
        
        .features-showcase h3 {
            color: #4CAF50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 10px;
            display: block;
        }
        
        .feature-name {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 5px;
            color: #fff;
        }
        
        .feature-desc {
            font-size: 0.9rem;
            opacity: 0.8;
            color: #e0e0e0;
        }
        
        .add-person-info {
            background: rgba(33, 150, 243, 0.1);
            border: 1px solid rgba(33, 150, 243, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .add-person-info p {
            margin-bottom: 10px;
            color: #2196F3;
            font-weight: 600;
        }
        
        .add-person-info ol {
            margin-left: 20px;
            color: #e0e0e0;
        }
        
        .add-person-info li {
            margin-bottom: 5px;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .floating {
            animation: floating 3s ease-in-out infinite;
        }
        
        @keyframes floating {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2.5rem;
            }
            
            .controls {
                flex-direction: column;
            }
            
            .status-grid {
                grid-template-columns: 1fr;
            }
            
            .debug-stats {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-brain"></i> Face Recognition System</h1>
            <p>Advanced Face Recognition, Object Detection & Age Estimation</p>
        </div>
        
        <div class="card debug-info">
            <h3><i class="fas fa-cog"></i> System Status</h3>
            <div class="debug-stats">
                <div class="debug-stat">
                    <div class="label">Known People</div>
                    <div class="value" id="people-count">0</div>
                </div>
                <div class="debug-stat">
                    <div class="label">Recognition Threshold</div>
                    <div class="value">0.2</div>
                </div>
                <div class="debug-stat">
                    <div class="label">Face Detection</div>
                    <div class="value">Haar Cascade</div>
                </div>
                <div class="debug-stat">
                    <div class="label">Object Detection</div>
                    <div class="value" id="object-status">Ready</div>
                </div>
            </div>
        </div>
        
    
        
        <div class="main-content">
            <div class="video-section">
                <div class="video-container">
                    <img id="video-feed" src="{{ url_for('video_feed') }}" alt="Video Feed">
                    <div class="video-overlay">
                        <div class="camera-controls">
                            <button class="btn btn-primary" onclick="startCamera()">
                                <i class="fas fa-play"></i> Start
                            </button>
                            <button class="btn btn-danger" onclick="stopCamera()">
                                <i class="fas fa-stop"></i> Stop
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="card">
                    <h3><i class="fas fa-star"></i> Features</h3>
                    <ul class="feature-list">
                        <li>Real-time Face Recognition</li>
                        <li>Object Detection (80+ types)</li>
                        <li>Accurate Age Estimation</li>
                        <li>Live Video Processing</li>
                        <li>Easy Person Management</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3><i class="fas fa-chart-line"></i> Live Status</h3>
                    <div class="status-grid">
                        <div class="status-item">
                            <div class="label">Camera</div>
                            <div class="value" id="camera-status">Stopped</div>
                        </div>
                        <div class="status-item">
                            <div class="label">Recognition</div>
                            <div class="value" id="recognition-status">Ready</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3><i class="fas fa-user-plus"></i> Add New Person</h3>
            <div class="add-person-info">
                <p><i class="fas fa-info-circle"></i> To add a new person:</p>
                <ol>
                    <li>Start the camera</li>
                    <li>Position the person's face in the camera view</li>
                    <li>Enter their name below and click "Add Person"</li>
                    <li>The system will automatically capture and save their face</li>
                </ol>
            </div>
            <div class="form-group">
                <label for="person-name">Name:</label>
                <input type="text" id="person-name" placeholder="Enter person's name">
            </div>
            <button class="btn btn-primary" onclick="addPerson()">
                <i class="fas fa-plus"></i> Add Person
            </button>
            <div id="add-status"></div>
        </div>
        
        <div class="card">
            <h3><i class="fas fa-users"></i> Known People</h3>
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
                        document.getElementById('recognition-status').textContent = 'Active';
                        document.getElementById('camera-status').classList.add('pulse');
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
                        document.getElementById('recognition-status').textContent = 'Ready';
                        document.getElementById('camera-status').classList.remove('pulse');
                        showStatus('Camera stopped', 'success');
                    }
                })
                .catch(error => {
                    showStatus('Failed to stop camera', 'error');
                });
        }
        
        function addPerson() {
            const name = document.getElementById('person-name').value.trim();
            
            if (!name) {
                showStatus('Please enter a name', 'error');
                return;
            }
            
            if (!isCameraRunning) {
                showStatus('Please start the camera first', 'error');
                return;
            }
            
            // Get the current video frame automatically
            const video = document.getElementById('video-feed');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = video.videoWidth || video.width;
            canvas.height = video.videoHeight || video.height;
            
            ctx.drawImage(video, 0, 0);
            const capturedImage = canvas.toDataURL('image/jpeg');
            
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
            const icon = type === 'success' ? 'fas fa-check-circle' : 'fas fa-exclamation-circle';
            statusDiv.innerHTML = '<div class="status ' + type + '"><i class="' + icon + '"></i>' + message + '</div>';
            
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
    
    print("Starting Working Recognition System...")
    print("Open your browser and go to: http://localhost:5000")
    print("Features: Reliable Face Recognition, Object Detection, Age Estimation")
    print("YOLO Status:", "Available" if YOLO_AVAILABLE else "Not Available (using OpenCV)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
