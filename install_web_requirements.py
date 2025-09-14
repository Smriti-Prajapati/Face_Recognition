"""
Installation script for Web Face Recognition System
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages."""
    print("🌐 Installing Web Face Recognition System...")
    print("=" * 50)
    
    # Core packages
    packages = [
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
        "Flask>=2.3.0",
        "Werkzeug>=2.3.0",
        "Pillow>=9.0.0",
        "requests>=2.28.0"
    ]
    
    # Optional YOLO package
    yolo_package = "ultralytics>=8.0.0"
    
    print("Installing core packages...")
    success_count = 0
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n✓ Installed {success_count}/{len(packages)} core packages")
    
    # Try to install YOLO
    print("\nInstalling YOLO for object detection...")
    if install_package(yolo_package):
        print("✓ YOLO installed successfully - Full object detection available!")
    else:
        print("⚠ YOLO installation failed - Will use OpenCV fallback")
        print("  You can still use the system, but object detection will be limited")
    
    print("\n" + "=" * 50)
    print("🎉 Installation complete!")
    print("\nTo run the web application:")
    print("  python web_face_recognition.py")
    print("\nThen open your browser and go to:")
    print("  http://localhost:5000")
    print("\nFeatures:")
    print("  ✓ Real-time face recognition")
    print("  ✓ Object detection (phone, bottle, chair, pen, etc.)")
    print("  ✓ Age estimation")
    print("  ✓ Beautiful web interface")
    print("  ✓ Easy person management")

if __name__ == "__main__":
    main()
