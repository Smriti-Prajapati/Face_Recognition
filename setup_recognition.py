"""
Setup script for Face Recognition System
This script helps you set up the face recognition system step by step.
"""

import os
import cv2

def check_opencv():
    """Check if OpenCV is installed."""
    try:
        print(f"OpenCV version: {cv2.__version__}")
        return True
    except ImportError:
        print("OpenCV not found. Please install it first:")
        print("pip install opencv-python")
        return False

def create_sample_directories():
    """Create sample directories for the recognition system."""
    directories = [
        "known_faces",
        "test_images", 
        "recognition_results"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def test_webcam():
    """Test if webcam is working."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Webcam not available")
        return False
    
    print("Webcam test - Press any key to close")
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Webcam Test", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Webcam is working!")
        cap.release()
        return True
    else:
        print("Could not read from webcam")
        cap.release()
        return False

def main():
    """Main setup function."""
    print("Face Recognition System Setup")
    print("=" * 40)
    
    # Check OpenCV
    if not check_opencv():
        return
    
    # Create directories
    print("\nCreating directories...")
    create_sample_directories()
    
    # Test webcam
    print("\nTesting webcam...")
    webcam_ok = test_webcam()
    
    print("\n" + "=" * 40)
    print("Setup Complete!")
    print("=" * 40)
    
    if webcam_ok:
        print("✓ OpenCV is working")
        print("✓ Webcam is working")
        print("✓ Directories created")
        print("\nYou can now run the face recognition system:")
        print("python face_recognition_system.py")
    else:
        print("✓ OpenCV is working")
        print("✗ Webcam not available (you can still use image recognition)")
        print("✓ Directories created")
        print("\nYou can still run the face recognition system for image recognition:")
        print("python face_recognition_system.py")
    
    print("\nNext steps:")
    print("1. Add some people to the database using option 1 or 2")
    print("2. Test recognition with option 3 or 4")
    print("3. Check database info with option 5")

if __name__ == "__main__":
    main()
