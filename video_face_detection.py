"""
Real-time Video Face Detection Module
This module provides live face detection capabilities for webcam and video files.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Callable
from face_detection_basic import FaceDetector

class VideoFaceDetector(FaceDetector):
    """Real-time face detector for video streams and webcam."""
    
    def __init__(self, cascade_path: Optional[str] = None):
        """Initialize the video face detector."""
        super().__init__(cascade_path)
        self.is_running = False
        self.fps_counter = 0
        self.fps_start_time = 0
        self.current_fps = 0
        
        # Performance tracking
        self.detection_times = []
        self.max_detection_time = 0.1  # Maximum detection time in seconds
        
    def calculate_fps(self):
        """Calculate and update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_info_overlay(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                         show_fps: bool = True, show_face_count: bool = True) -> np.ndarray:
        """
        Draw information overlay on the image.
        
        Args:
            image: Input image
            faces: List of detected faces
            show_fps: Whether to show FPS counter
            show_face_count: Whether to show face count
            
        Returns:
            Image with overlay information
        """
        result_image = image.copy()
        
        # Draw FPS counter
        if show_fps:
            cv2.putText(result_image, f"FPS: {self.current_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw face count
        if show_face_count:
            cv2.putText(result_image, f"Faces: {len(faces)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw detection time
        if self.detection_times:
            avg_detection_time = np.mean(self.detection_times[-10:])  # Average of last 10 detections
            cv2.putText(result_image, f"Detection: {avg_detection_time:.3f}s", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_image
    
    def detect_faces_optimized(self, image: np.ndarray, 
                              scale_factor: float = 1.1, 
                              min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Optimized face detection for real-time processing.
        
        Args:
            image: Input image
            scale_factor: Scale factor for detection
            min_neighbors: Minimum neighbors for detection
            
        Returns:
            List of detected face rectangles
        """
        start_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Track detection time
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        
        # Keep only recent detection times
        if len(self.detection_times) > 50:
            self.detection_times = self.detection_times[-50:]
        
        return faces.tolist()
    
    def process_frame(self, frame: np.ndarray, 
                     draw_rectangles: bool = True,
                     draw_info: bool = True) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Process a single frame for face detection.
        
        Args:
            frame: Input frame
            draw_rectangles: Whether to draw rectangles around faces
            draw_info: Whether to draw information overlay
            
        Returns:
            Tuple of (processed_frame, detected_faces)
        """
        # Detect faces
        faces = self.detect_faces_optimized(frame)
        
        # Draw rectangles around faces
        if draw_rectangles:
            frame = self.draw_faces(frame, faces, color=(0, 255, 0), thickness=2)
        
        # Draw information overlay
        if draw_info:
            frame = self.draw_info_overlay(frame, faces)
        
        return frame, faces
    
    def run_webcam_detection(self, camera_index: int = 0, 
                           window_name: str = "Face Detection",
                           callback: Optional[Callable] = None) -> None:
        """
        Run real-time face detection on webcam.
        
        Args:
            camera_index: Camera index (usually 0 for default camera)
            window_name: Name of the display window
            callback: Optional callback function for face detection events
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.fps_start_time = time.time()
        
        print("Face detection started. Press 'q' to quit, 's' to save screenshot.")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame, faces = self.process_frame(frame)
                
                # Call callback function if provided
                if callback:
                    callback(faces, processed_frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"Screenshot saved: {screenshot_path}")
                elif key == ord('r'):
                    # Reset detection times
                    self.detection_times = []
                    print("Detection times reset")
        
        except KeyboardInterrupt:
            print("Face detection interrupted by user")
        
        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            print("Face detection stopped")
    
    def run_video_file_detection(self, video_path: str, 
                               output_path: Optional[str] = None,
                               window_name: str = "Video Face Detection",
                               callback: Optional[Callable] = None) -> None:
        """
        Run face detection on a video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            window_name: Name of the display window
            callback: Optional callback function for face detection events
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.is_running = True
        self.fps_start_time = time.time()
        
        print(f"Processing video: {video_path}")
        print("Press 'q' to quit, 'p' to pause/resume, 's' to save current frame")
        
        paused = False
        
        try:
            while self.is_running:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video reached")
                        break
                    
                    # Process frame
                    processed_frame, faces = self.process_frame(frame)
                    
                    # Call callback function if provided
                    if callback:
                        callback(faces, processed_frame)
                    
                    # Write frame to output video
                    if writer:
                        writer.write(processed_frame)
                    
                    # Display frame
                    cv2.imshow(window_name, processed_frame)
                    
                    # Calculate FPS
                    self.calculate_fps()
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('s'):
                    # Save current frame
                    screenshot_path = f"frame_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"Frame saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("Video processing interrupted by user")
        
        finally:
            self.is_running = False
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("Video processing completed")
    
    def stop_detection(self):
        """Stop the face detection process."""
        self.is_running = False

def face_detection_callback(faces: List[Tuple[int, int, int, int]], frame: np.ndarray):
    """
    Example callback function for face detection events.
    
    Args:
        faces: List of detected faces
        frame: Current frame
    """
    if faces:
        print(f"Detected {len(faces)} face(s) in current frame")

def main():
    """Example usage of the VideoFaceDetector class."""
    # Initialize detector
    detector = VideoFaceDetector()
    
    print("Face Detection System")
    print("1. Webcam detection")
    print("2. Video file detection")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Webcam detection
        try:
            detector.run_webcam_detection(callback=face_detection_callback)
        except Exception as e:
            print(f"Error with webcam detection: {str(e)}")
    
    elif choice == "2":
        # Video file detection
        video_path = input("Enter path to video file: ").strip()
        if not video_path:
            video_path = "sample_video.mp4"
        
        output_path = input("Enter output video path (optional): ").strip()
        if not output_path:
            output_path = None
        
        try:
            detector.run_video_file_detection(video_path, output_path, callback=face_detection_callback)
        except Exception as e:
            print(f"Error with video file detection: {str(e)}")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
