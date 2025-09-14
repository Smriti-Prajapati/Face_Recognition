"""
Facial Landmark Detection Module
This module provides detailed facial landmark detection using dlib.
"""

import cv2
import numpy as np
import dlib
import os
from typing import List, Tuple, Optional, Dict, Any
from face_detection_basic import FaceDetector

class FaceLandmarkDetector(FaceDetector):
    """Face detector with facial landmark detection capabilities."""
    
    def __init__(self, cascade_path: Optional[str] = None, 
                 predictor_path: Optional[str] = None):
        """
        Initialize the face landmark detector.
        
        Args:
            cascade_path: Path to Haar cascade XML file
            predictor_path: Path to dlib shape predictor file
        """
        super().__init__(cascade_path)
        
        # Initialize dlib face detector and landmark predictor
        self.dlib_detector = dlib.get_frontal_face_detector()
        
        if predictor_path and os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
        else:
            # Try to use default predictor (you may need to download this)
            default_predictor = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(default_predictor):
                self.predictor = dlib.shape_predictor(default_predictor)
            else:
                print("Warning: No shape predictor found. Landmark detection will be disabled.")
                print("Please download shape_predictor_68_face_landmarks.dat from dlib.net")
                self.predictor = None
    
    def detect_landmarks(self, image: np.ndarray, 
                        face_rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Detect facial landmarks for a given face rectangle.
        
        Args:
            image: Input image
            face_rect: Face rectangle (x, y, w, h)
            
        Returns:
            Array of landmark coordinates or None if detection failed
        """
        if self.predictor is None:
            return None
        
        # Convert face rectangle to dlib format
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect landmarks
        landmarks = self.predictor(gray, dlib_rect)
        
        # Convert to numpy array
        landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        return landmarks_array
    
    def detect_all_landmarks(self, image: np.ndarray, 
                           faces: List[Tuple[int, int, int, int]]) -> List[Optional[np.ndarray]]:
        """
        Detect landmarks for all faces in the image.
        
        Args:
            image: Input image
            faces: List of face rectangles
            
        Returns:
            List of landmark arrays for each face
        """
        landmarks_list = []
        
        for face_rect in faces:
            landmarks = self.detect_landmarks(image, face_rect)
            landmarks_list.append(landmarks)
        
        return landmarks_list
    
    def draw_landmarks(self, image: np.ndarray, 
                      landmarks: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0),
                      radius: int = 2) -> np.ndarray:
        """
        Draw facial landmarks on the image.
        
        Args:
            image: Input image
            landmarks: Array of landmark coordinates
            color: Color of the landmark points
            radius: Radius of the landmark points
            
        Returns:
            Image with landmarks drawn
        """
        result_image = image.copy()
        
        for (x, y) in landmarks:
            cv2.circle(result_image, (x, y), radius, color, -1)
        
        return result_image
    
    def draw_landmark_connections(self, image: np.ndarray, 
                                landmarks: np.ndarray,
                                color: Tuple[int, int, int] = (0, 255, 0),
                                thickness: int = 1) -> np.ndarray:
        """
        Draw connections between facial landmarks.
        
        Args:
            image: Input image
            landmarks: Array of landmark coordinates
            color: Color of the connections
            thickness: Thickness of the connections
            
        Returns:
            Image with landmark connections drawn
        """
        result_image = image.copy()
        
        # Define facial feature connections (68-point model)
        connections = [
            # Jaw line
            list(range(0, 17)),
            # Right eyebrow
            list(range(17, 22)),
            # Left eyebrow
            list(range(22, 27)),
            # Nose
            list(range(27, 31)) + list(range(31, 36)),
            # Right eye
            list(range(36, 42)) + [36],
            # Left eye
            list(range(42, 48)) + [42],
            # Outer lip
            list(range(48, 60)) + [48],
            # Inner lip
            list(range(60, 68)) + [60]
        ]
        
        for connection in connections:
            for i in range(len(connection) - 1):
                pt1 = tuple(landmarks[connection[i]].astype(int))
                pt2 = tuple(landmarks[connection[i + 1]].astype(int))
                cv2.line(result_image, pt1, pt2, color, thickness)
        
        return result_image
    
    def get_landmark_regions(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract specific facial regions from landmarks.
        
        Args:
            landmarks: Array of 68 facial landmarks
            
        Returns:
            Dictionary containing different facial regions
        """
        regions = {
            'jaw': landmarks[0:17],
            'right_eyebrow': landmarks[17:22],
            'left_eyebrow': landmarks[22:27],
            'nose': landmarks[27:36],
            'right_eye': landmarks[36:42],
            'left_eye': landmarks[42:48],
            'outer_mouth': landmarks[48:60],
            'inner_mouth': landmarks[60:68]
        }
        
        return regions
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        Args:
            eye_landmarks: Landmarks for one eye (6 points)
            
        Returns:
            Eye aspect ratio
        """
        # Calculate distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def detect_blinks(self, landmarks: np.ndarray, 
                     threshold: float = 0.25) -> Dict[str, bool]:
        """
        Detect if eyes are closed (blinking).
        
        Args:
            landmarks: Array of facial landmarks
            threshold: EAR threshold for blink detection
            
        Returns:
            Dictionary with blink status for each eye
        """
        regions = self.get_landmark_regions(landmarks)
        
        left_ear = self.calculate_eye_aspect_ratio(regions['left_eye'])
        right_ear = self.calculate_eye_aspect_ratio(regions['right_eye'])
        
        return {
            'left_eye_closed': left_ear < threshold,
            'right_eye_closed': right_ear < threshold,
            'both_eyes_closed': left_ear < threshold and right_ear < threshold,
            'left_ear': left_ear,
            'right_ear': right_ear
        }
    
    def calculate_head_pose(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate head pose angles (pitch, yaw, roll).
        
        Args:
            landmarks: Array of facial landmarks
            
        Returns:
            Dictionary with head pose angles
        """
        # Key points for head pose estimation
        nose_tip = landmarks[30]
        chin = landmarks[8]
        left_eye = landmarks[36]
        right_eye = landmarks[45]
        
        # Calculate basic angles
        # Roll (head tilt)
        eye_vector = right_eye - left_eye
        roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
        
        # Yaw (head turn left/right)
        nose_chin_vector = chin - nose_tip
        yaw = np.arctan2(nose_chin_vector[0], nose_chin_vector[1]) * 180 / np.pi
        
        # Pitch (head nod up/down) - simplified calculation
        eye_center = (left_eye + right_eye) / 2
        pitch_vector = nose_tip - eye_center
        pitch = np.arctan2(pitch_vector[1], pitch_vector[0]) * 180 / np.pi
        
        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll
        }
    
    def analyze_face_emotions(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Analyze facial expressions based on landmarks.
        
        Args:
            landmarks: Array of facial landmarks
            
        Returns:
            Dictionary with emotion scores
        """
        regions = self.get_landmark_regions(landmarks)
        
        # Calculate mouth opening
        mouth_height = np.linalg.norm(landmarks[51] - landmarks[57])
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        mouth_ratio = mouth_height / mouth_width
        
        # Calculate eyebrow raise
        left_eyebrow_center = np.mean(regions['left_eyebrow'], axis=0)
        right_eyebrow_center = np.mean(regions['right_eyebrow'], axis=0)
        eyebrow_center = (left_eyebrow_center + right_eyebrow_center) / 2
        
        left_eye_center = np.mean(regions['left_eye'], axis=0)
        right_eye_center = np.mean(regions['right_eye'], axis=0)
        eye_center = (left_eye_center + right_eye_center) / 2
        
        eyebrow_raise = eyebrow_center[1] - eye_center[1]
        
        # Simple emotion scoring (normalized)
        emotions = {
            'smile': min(mouth_ratio * 2, 1.0),
            'surprise': min(eyebrow_raise / 20, 1.0),
            'frown': max(0, 1 - mouth_ratio * 2),
            'neutral': 0.5
        }
        
        return emotions
    
    def comprehensive_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive facial analysis.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with complete analysis results
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        if not faces:
            return {'faces_detected': 0}
        
        results = {
            'faces_detected': len(faces),
            'face_analyses': []
        }
        
        for i, face_rect in enumerate(faces):
            # Detect landmarks
            landmarks = self.detect_landmarks(image, face_rect)
            
            if landmarks is not None:
                # Get facial regions
                regions = self.get_landmark_regions(landmarks)
                
                # Detect blinks
                blink_info = self.detect_blinks(landmarks)
                
                # Calculate head pose
                head_pose = self.calculate_head_pose(landmarks)
                
                # Analyze emotions
                emotions = self.analyze_face_emotions(landmarks)
                
                face_analysis = {
                    'face_id': i,
                    'face_rectangle': face_rect,
                    'landmarks': landmarks.tolist(),
                    'regions': {k: v.tolist() for k, v in regions.items()},
                    'blink_info': blink_info,
                    'head_pose': head_pose,
                    'emotions': emotions
                }
                
                results['face_analyses'].append(face_analysis)
        
        return results

def main():
    """Example usage of the FaceLandmarkDetector class."""
    # Initialize detector
    detector = FaceLandmarkDetector()
    
    # Load an image
    image_path = "sample_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Please provide a valid image path.")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}")
        return
    
    # Perform comprehensive analysis
    results = detector.comprehensive_analysis(image)
    
    print("=== Facial Landmark Analysis Results ===")
    print(f"Faces detected: {results['faces_detected']}")
    
    for analysis in results['face_analyses']:
        print(f"\nFace {analysis['face_id'] + 1}:")
        print(f"  Rectangle: {analysis['face_rectangle']}")
        print(f"  Landmarks: {len(analysis['landmarks'])} points")
        print(f"  Blink info: {analysis['blink_info']}")
        print(f"  Head pose: {analysis['head_pose']}")
        print(f"  Emotions: {analysis['emotions']}")
    
    # Draw landmarks on image
    if results['face_analyses']:
        result_image = image.copy()
        
        for analysis in results['face_analyses']:
            landmarks = np.array(analysis['landmarks'])
            
            # Draw face rectangle
            x, y, w, h = analysis['face_rectangle']
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw landmarks
            result_image = detector.draw_landmarks(result_image, landmarks)
            
            # Draw landmark connections
            result_image = detector.draw_landmark_connections(result_image, landmarks)
        
        # Save result
        cv2.imwrite("landmark_analysis_result.jpg", result_image)
        print("\nResult saved as 'landmark_analysis_result.jpg'")
        
        # Display result
        cv2.imshow("Facial Landmark Analysis", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
