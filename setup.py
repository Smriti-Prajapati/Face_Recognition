"""
Setup script for Face Detection System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="face-detection-system",
    version="1.0.0",
    author="Face Detection Team",
    author_email="contact@facedetection.com",
    description="A comprehensive Python-based face detection and recognition system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/face-detection-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "face-detection-gui=face_detection_gui:main",
            "face-detection-basic=examples.basic_example:main",
            "face-detection-image=examples.image_analysis_example:main",
            "face-detection-video=examples.video_example:main",
            "face-detection-recognition=examples.recognition_example:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.dat", "*.xml", "*.json"],
    },
    keywords="face detection, face recognition, computer vision, opencv, dlib, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/face-detection-system/issues",
        "Source": "https://github.com/yourusername/face-detection-system",
        "Documentation": "https://face-detection-system.readthedocs.io/",
    },
)
