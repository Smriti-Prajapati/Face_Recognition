# Face Detection System

A comprehensive Python-based face detection and recognition system with multiple detection methods, facial landmark analysis, and a user-friendly GUI interface.

## Features

- **Basic Face Detection**: Using OpenCV Haar Cascade classifiers
- **Advanced Image Processing**: Multi-angle face detection, eye detection, smile detection
- **Real-time Video Detection**: Webcam and video file processing
- **Facial Landmark Detection**: 68-point facial landmark detection using dlib
- **Face Recognition**: Train and recognize faces using face_recognition library
- **GUI Interface**: Comprehensive graphical user interface for all features
- **Batch Processing**: Process multiple images at once

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenCV
- dlib
- face_recognition
- tkinter (usually included with Python)

### Setup

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dlib shape predictor (for landmark detection):
   - Download `shape_predictor_68_face_landmarks.dat` from [dlib.net](http://dlib.net/files/)
   - Place it in the project directory

## Quick Start

### Command Line Usage

#### Basic Face Detection
```python
from face_detection_basic import FaceDetector

# Initialize detector
detector = FaceDetector()

# Load and process image
image = cv2.imread("your_image.jpg")
result_image, faces = detector.detect_and_draw(image)

# Display results
cv2.imshow("Face Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### Advanced Image Analysis
```python
from image_face_detection import ImageFaceDetector

# Initialize detector
detector = ImageFaceDetector()

# Analyze image
results = detector.analyze_image("your_image.jpg")
print(f"Found {results['total_faces']} faces")
```

#### Real-time Video Detection
```python
from video_face_detection import VideoFaceDetector

# Initialize detector
detector = VideoFaceDetector()

# Start webcam detection
detector.run_webcam_detection()
```

#### Face Recognition
```python
from face_recognition import FaceRecognizer

# Initialize recognizer
recognizer = FaceRecognizer()

# Add faces to database
recognizer.add_face("John", cv2.imread("john.jpg"))
recognizer.add_face("Jane", cv2.imread("jane.jpg"))

# Recognize faces in new image
result_image, recognized_faces = recognizer.recognize_and_draw(image)
```

### GUI Application

Run the graphical interface:

```bash
python face_detection_gui.py
```

The GUI provides:
- **Image Detection Tab**: Process static images with various detection options
- **Video Detection Tab**: Real-time webcam and video file processing
- **Face Recognition Tab**: Manage face database and recognize faces
- **Facial Landmarks Tab**: Analyze facial landmarks and emotions
- **Settings Tab**: Configure application settings
- **Log Tab**: View processing logs and messages

## Module Documentation

### face_detection_basic.py
Core face detection functionality using Haar Cascade classifiers.

**Key Classes:**
- `FaceDetector`: Basic face detection with OpenCV

**Key Methods:**
- `detect_faces()`: Detect faces in an image
- `draw_faces()`: Draw rectangles around detected faces
- `detect_and_draw()`: Combined detection and drawing

### image_face_detection.py
Advanced image processing with multiple detection types.

**Key Classes:**
- `ImageFaceDetector`: Extended face detector with multiple features

**Key Methods:**
- `detect_faces_multiple_angles()`: Detect frontal and profile faces
- `detect_eyes()`: Detect eyes within face regions
- `detect_smiles()`: Detect smiles within face regions
- `analyze_image()`: Comprehensive image analysis
- `batch_process_images()`: Process multiple images

### video_face_detection.py
Real-time video processing capabilities.

**Key Classes:**
- `VideoFaceDetector`: Real-time video face detection

**Key Methods:**
- `run_webcam_detection()`: Start webcam face detection
- `run_video_file_detection()`: Process video files
- `process_frame()`: Process individual video frames

### face_landmarks.py
Facial landmark detection and analysis.

**Key Classes:**
- `FaceLandmarkDetector`: Facial landmark detection using dlib

**Key Methods:**
- `detect_landmarks()`: Detect 68 facial landmarks
- `calculate_eye_aspect_ratio()`: Calculate EAR for blink detection
- `detect_blinks()`: Detect eye blinks
- `calculate_head_pose()`: Estimate head pose angles
- `analyze_face_emotions()`: Analyze facial expressions

### face_recognition.py
Face recognition and identification system.

**Key Classes:**
- `FaceRecognizer`: Face recognition and database management

**Key Methods:**
- `add_face()`: Add a face to the recognition database
- `recognize_faces()`: Recognize faces in an image
- `train_from_video()`: Train recognition from video files
- `get_face_database_info()`: Get database statistics

## Configuration

### Detection Parameters

- **Scale Factor**: How much the image size is reduced at each scale (default: 1.1)
- **Min Neighbors**: Minimum neighbors for face detection (default: 5)
- **Min Size**: Minimum face size in pixels (default: 30x30)

### Recognition Parameters

- **Tolerance**: Face recognition tolerance (lower = more strict, default: 0.6)
- **Encoding File**: Path to save face encodings (default: face_encodings.pkl)

## Examples

### Example 1: Basic Face Detection
```python
import cv2
from face_detection_basic import FaceDetector

# Load image
image = cv2.imread("group_photo.jpg")

# Detect faces
detector = FaceDetector()
faces = detector.detect_faces(image)

# Draw rectangles around faces
result = detector.draw_faces(image, faces)

# Save result
cv2.imwrite("faces_detected.jpg", result)
print(f"Found {len(faces)} faces")
```

### Example 2: Comprehensive Image Analysis
```python
from image_face_detection import ImageFaceDetector

detector = ImageFaceDetector()

# Analyze single image
results = detector.analyze_image("photo.jpg")
print(f"Analysis Results:")
print(f"  Total faces: {results['total_faces']}")
print(f"  Eyes detected: {results['eyes_detected']}")
print(f"  Smiles detected: {results['smiles_detected']}")

# Batch process directory
results = detector.batch_process_images("input_images/", "output_images/")
```

### Example 3: Face Recognition System
```python
from face_recognition import FaceRecognizer

recognizer = FaceRecognizer()

# Add known faces
recognizer.add_face("Alice", cv2.imread("alice.jpg"))
recognizer.add_face("Bob", cv2.imread("bob.jpg"))

# Recognize in new image
image = cv2.imread("test_image.jpg")
result_image, recognized = recognizer.recognize_and_draw(image)

for name, confidence, location in recognized:
    print(f"Found {name} with confidence {confidence:.2f}")
```

### Example 4: Facial Landmark Analysis
```python
from face_landmarks import FaceLandmarkDetector

detector = FaceLandmarkDetector()

# Analyze facial landmarks
results = detector.comprehensive_analysis("portrait.jpg")

for analysis in results['face_analyses']:
    print(f"Face {analysis['face_id'] + 1}:")
    print(f"  Blink info: {analysis['blink_info']}")
    print(f"  Head pose: {analysis['head_pose']}")
    print(f"  Emotions: {analysis['emotions']}")
```

## Troubleshooting

### Common Issues

1. **dlib Installation Issues**:
   - On Windows: Use conda instead of pip
   - On Linux: Install cmake and build tools first

2. **Shape Predictor Not Found**:
   - Download `shape_predictor_68_face_landmarks.dat` from dlib.net
   - Place in project directory

3. **Camera Access Issues**:
   - Check camera permissions
   - Try different camera indices (0, 1, 2, etc.)

4. **Memory Issues with Large Images**:
   - Resize images before processing
   - Use batch processing for multiple images

### Performance Tips

1. **For Real-time Detection**:
   - Use smaller image resolutions
   - Adjust detection parameters for speed vs accuracy
   - Process every nth frame instead of every frame

2. **For Accuracy**:
   - Use higher resolution images
   - Adjust scale factor and min neighbors
   - Ensure good lighting conditions

3. **For Face Recognition**:
   - Use multiple training images per person
   - Ensure good quality, well-lit training images
   - Adjust tolerance based on your needs

## File Structure

```
face_detection_system/
├── face_detection_basic.py      # Basic face detection
├── image_face_detection.py      # Advanced image processing
├── video_face_detection.py      # Real-time video processing
├── face_landmarks.py            # Facial landmark detection
├── face_recognition.py          # Face recognition system
├── face_detection_gui.py        # GUI application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── examples/                    # Example scripts
    ├── basic_example.py
    ├── image_analysis_example.py
    ├── video_example.py
    └── recognition_example.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenCV for computer vision capabilities
- dlib for facial landmark detection
- face_recognition library for face recognition
- The computer vision community for algorithms and techniques
