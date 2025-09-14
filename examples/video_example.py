"""
Video Face Detection Example
This example demonstrates real-time video face detection.
"""

import cv2
import time
from video_face_detection import VideoFaceDetector

def face_detection_callback(faces, frame):
    """Callback function for face detection events."""
    if faces:
        print(f"Detected {len(faces)} face(s) in current frame")

def main():
    """Run video face detection example."""
    print("Video Face Detection Example")
    print("=" * 40)
    
    # Initialize detector
    detector = VideoFaceDetector()
    
    print("Choose detection mode:")
    print("1. Webcam detection")
    print("2. Video file detection")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Webcam detection
        print("\nStarting webcam detection...")
        print("Press 'q' to quit, 's' to save screenshot, 'r' to reset detection times")
        
        try:
            detector.run_webcam_detection(callback=face_detection_callback)
        except Exception as e:
            print(f"Error with webcam detection: {str(e)}")
            print("Make sure your camera is connected and not being used by another application.")
    
    elif choice == "2":
        # Video file detection
        video_path = input("Enter path to video file: ").strip()
        
        if not video_path:
            print("No video path provided. Using default sample video.")
            video_path = "sample_video.mp4"
        
        output_path = input("Enter output video path (optional): ").strip()
        if not output_path:
            output_path = None
        
        print(f"\nProcessing video: {video_path}")
        print("Press 'q' to quit, 'p' to pause/resume, 's' to save current frame")
        
        try:
            detector.run_video_file_detection(video_path, output_path, callback=face_detection_callback)
        except Exception as e:
            print(f"Error with video file detection: {str(e)}")
            print("Make sure the video file exists and is in a supported format.")
    
    else:
        print("Invalid choice")

def performance_test():
    """Test detection performance with different parameters."""
    print("\nPerformance Test")
    print("=" * 40)
    
    # Initialize detector
    detector = VideoFaceDetector()
    
    # Test different scale factors
    scale_factors = [1.05, 1.1, 1.2, 1.3]
    
    print("Testing different scale factors...")
    
    for scale_factor in scale_factors:
        print(f"\nTesting scale factor: {scale_factor}")
        
        # Create a test image (you would load a real image here)
        test_image = cv2.imread("sample_image.jpg")
        if test_image is None:
            print("No test image found. Skipping performance test.")
            break
        
        # Measure detection time
        start_time = time.time()
        faces = detector.detect_faces_optimized(test_image, scale_factor=scale_factor)
        detection_time = time.time() - start_time
        
        print(f"  Detection time: {detection_time:.3f} seconds")
        print(f"  Faces found: {len(faces)}")
        print(f"  FPS (estimated): {1.0 / detection_time:.1f}")

def custom_processing_example():
    """Example of custom frame processing."""
    print("\nCustom Processing Example")
    print("=" * 40)
    
    # Initialize detector
    detector = VideoFaceDetector()
    
    # Load a video file
    video_path = "sample_video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    print(f"Processing video: {video_path}")
    print("Press 'q' to quit")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame
            if frame_count % 5 == 0:
                # Detect faces
                faces = detector.detect_faces_optimized(frame)
                
                # Draw custom annotations
                result_frame = frame.copy()
                for i, (x, y, w, h) in enumerate(faces):
                    # Draw rectangle
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw face number
                    cv2.putText(result_frame, f"Face {i+1}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw face area
                    area = w * h
                    cv2.putText(result_frame, f"Area: {area}", (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw frame info
                cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result_frame, f"Faces: {len(faces)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Custom Processing", result_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    main()
    
    # Uncomment to run additional examples
    # performance_test()
    # custom_processing_example()
