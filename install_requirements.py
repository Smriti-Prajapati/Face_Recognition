"""
Alternative installation script for Python 3.13 compatibility
This script installs packages one by one to avoid dependency conflicts
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package with error handling."""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages."""
    print("Installing Face Detection System Dependencies")
    print("=" * 50)
    
    # List of packages to install (in order of dependency)
    packages = [
        "numpy",
        "opencv-python",
        "Pillow",
        "matplotlib",
        "dlib",
        "face-recognition"
    ]
    
    # Alternative packages for problematic ones
    alternative_packages = {
        "dlib": "dlib-binary",  # Pre-compiled binary version
        "face-recognition": "face-recognition-models"  # Alternative if main package fails
    }
    
    failed_packages = []
    
    for package in packages:
        success = install_package(package)
        if not success:
            # Try alternative package if available
            if package in alternative_packages:
                print(f"Trying alternative package for {package}...")
                alt_success = install_package(alternative_packages[package])
                if not alt_success:
                    failed_packages.append(package)
            else:
                failed_packages.append(package)
    
    print("\n" + "=" * 50)
    if failed_packages:
        print("Failed to install the following packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nYou may need to install these manually or use conda instead of pip.")
        print("For dlib, try: conda install -c conda-forge dlib")
        print("For face-recognition, try: pip install face-recognition --no-deps")
    else:
        print("✓ All packages installed successfully!")
        print("\nNext steps:")
        print("1. Download shape_predictor_68_face_landmarks.dat from dlib.net")
        print("2. Place it in your project directory")
        print("3. Run: python face_detection_gui.py")

if __name__ == "__main__":
    main()
