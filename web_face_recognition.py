"""
Web Face Recognition and Object Detection System
Beautiful web GUI with real-time video processing
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

class WebRecognitionSystem:
    """Web-based recognition system with YOLO object detection."""
    
    def __init__(self):
        """Initialize the web recognition system."""
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}
        self.known_faces_folder = "known_people"
        
        # YOLO model for object detection
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
                print("✓ YOLO model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load YOLO model: {e}")
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
        """Extract features from a face region."""
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
        
        # Simple rectangular object detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 0.3 < aspect_ratio < 3.0:
                    detected_objects.append(("Object", 0.7, (x, y, w, h)))
        
        return detected_objects
    
    def estimate_age(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> str:
        """Estimate age from face."""
        x, y, w, h = face_rect
        face_area = w * h
        
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
        """Recognize faces in an image."""
        if not self.known_faces:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        recognized_faces = []
        
        for face_rect in faces:
            face_features = self.extract_face_features(image, face_rect)
            
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
            
            # Estimate age
            age = self.estimate_age(frame, (x, y, w, h))
            
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
recognition_system = WebRecognitionSystem()

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
            content: "✓";
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
            <h1>🤖 AI Recognition System</h1>
            <p>Real-time Face Recognition, Object Detection & Age Estimation</p>
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
                <h3>🎯 Features</h3>
                <ul class="feature-list">
                    <li>Real-time Face Recognition</li>
                    <li>Object Detection (Phone, Bottle, Chair, Pen, etc.)</li>
                    <li>Age Estimation</li>
                    <li>Live Video Processing</li>
                    <li>Easy Person Management</li>
                </ul>
                
                <h3>📊 System Status</h3>
                <div id="system-status">
                    <p>Camera: <span id="camera-status">Stopped</span></p>
                    <p>Known People: <span id="people-count">0</span></p>
                    <p>Object Detection: <span id="object-status">Ready</span></p>
                </div>
            </div>
        </div>
        
        <div class="add-person-section">
            <h3>👤 Add New Person</h3>
            <div class="form-group">
                <label for="person-name">Name:</label>
                <input type="text" id="person-name" placeholder="Enter person's name">
            </div>
            <button class="btn btn-primary" onclick="addPerson()">Add Person</button>
            <div id="add-status"></div>
        </div>
        
        <div class="known-people">
            <h3>👥 Known People</h3>
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
            statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
            
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
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    print("🌐 Starting Web Recognition System...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🎯 Features: Face Recognition, Object Detection, Age Estimation")
    print("🔧 YOLO Status:", "Available" if YOLO_AVAILABLE else "Not Available (using OpenCV)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
