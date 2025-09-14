"""
Face Detection GUI Application
A comprehensive graphical user interface for the face detection system.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
import json
from typing import Optional, Dict, Any

# Import our face detection modules
from face_detection_basic import FaceDetector
from image_face_detection import ImageFaceDetector
from video_face_detection import VideoFaceDetector
from face_landmarks import FaceLandmarkDetector
from face_recognition import FaceRecognizer

class FaceDetectionGUI:
    """Main GUI application for face detection system."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Face Detection System")
        self.root.geometry("1200x800")
        
        # Initialize detectors
        self.basic_detector = FaceDetector()
        self.image_detector = ImageFaceDetector()
        self.video_detector = VideoFaceDetector()
        self.landmark_detector = FaceLandmarkDetector()
        self.recognizer = FaceRecognizer()
        
        # Current image and results
        self.current_image = None
        self.current_results = None
        self.is_processing = False
        
        # Setup GUI
        self.setup_gui()
        
        # Load settings
        self.load_settings()
    
    def setup_gui(self):
        """Setup the GUI components."""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_image_tab()
        self.create_video_tab()
        self.create_recognition_tab()
        self.create_landmarks_tab()
        self.create_settings_tab()
        self.create_log_tab()
    
    def create_image_tab(self):
        """Create the image processing tab."""
        image_frame = ttk.Frame(self.notebook)
        self.notebook.add(image_frame, text="Image Detection")
        
        # Left panel for controls
        left_panel = ttk.Frame(image_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # File operations
        file_frame = ttk.LabelFrame(left_panel, text="File Operations")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Save Result", command=self.save_result).pack(fill=tk.X, pady=2)
        
        # Detection options
        detection_frame = ttk.LabelFrame(left_panel, text="Detection Options")
        detection_frame.pack(fill=tk.X, pady=5)
        
        self.detect_faces_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(detection_frame, text="Detect Faces", variable=self.detect_faces_var).pack(anchor=tk.W)
        
        self.detect_eyes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(detection_frame, text="Detect Eyes", variable=self.detect_eyes_var).pack(anchor=tk.W)
        
        self.detect_smiles_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(detection_frame, text="Detect Smiles", variable=self.detect_smiles_var).pack(anchor=tk.W)
        
        self.detect_landmarks_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(detection_frame, text="Detect Landmarks", variable=self.detect_landmarks_var).pack(anchor=tk.W)
        
        # Detection parameters
        params_frame = ttk.LabelFrame(left_panel, text="Parameters")
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="Scale Factor:").pack(anchor=tk.W)
        self.scale_factor_var = tk.DoubleVar(value=1.1)
        scale_frame = ttk.Frame(params_frame)
        scale_frame.pack(fill=tk.X)
        ttk.Scale(scale_frame, from_=1.05, to=1.5, variable=self.scale_factor_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(scale_frame, textvariable=self.scale_factor_var).pack(side=tk.RIGHT)
        
        ttk.Label(params_frame, text="Min Neighbors:").pack(anchor=tk.W)
        self.min_neighbors_var = tk.IntVar(value=5)
        neighbors_frame = ttk.Frame(params_frame)
        neighbors_frame.pack(fill=tk.X)
        ttk.Scale(neighbors_frame, from_=1, to=20, variable=self.min_neighbors_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(neighbors_frame, textvariable=self.min_neighbors_var).pack(side=tk.RIGHT)
        
        # Process button
        ttk.Button(left_panel, text="Process Image", command=self.process_image).pack(fill=tk.X, pady=10)
        
        # Results display
        results_frame = ttk.LabelFrame(left_panel, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, width=30)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for image display
        right_panel = ttk.Frame(image_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image display
        self.image_label = ttk.Label(right_panel, text="No image loaded")
        self.image_label.pack(fill=tk.BOTH, expand=True)
    
    def create_video_tab(self):
        """Create the video processing tab."""
        video_frame = ttk.Frame(self.notebook)
        self.notebook.add(video_frame, text="Video Detection")
        
        # Controls
        controls_frame = ttk.Frame(video_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Start Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Stop Detection", command=self.stop_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Load Video File", command=self.load_video_file).pack(side=tk.LEFT, padx=5)
        
        # Video display
        self.video_label = ttk.Label(video_frame, text="Video display will appear here")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video info
        self.video_info_label = ttk.Label(video_frame, text="No video loaded")
        self.video_info_label.pack(fill=tk.X, padx=5, pady=5)
    
    def create_recognition_tab(self):
        """Create the face recognition tab."""
        recognition_frame = ttk.Frame(self.notebook)
        self.notebook.add(recognition_frame, text="Face Recognition")
        
        # Left panel for database management
        left_panel = ttk.Frame(recognition_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Database operations
        db_frame = ttk.LabelFrame(left_panel, text="Database Management")
        db_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(db_frame, text="Add Faces from Directory", command=self.add_faces_from_directory).pack(fill=tk.X, pady=2)
        ttk.Button(db_frame, text="Add Face from Image", command=self.add_face_from_image).pack(fill=tk.X, pady=2)
        ttk.Button(db_frame, text="Train from Video", command=self.train_from_video).pack(fill=tk.X, pady=2)
        ttk.Button(db_frame, text="View Database Info", command=self.view_database_info).pack(fill=tk.X, pady=2)
        ttk.Button(db_frame, text="Clear Database", command=self.clear_database).pack(fill=tk.X, pady=2)
        
        # Recognition settings
        recog_frame = ttk.LabelFrame(left_panel, text="Recognition Settings")
        recog_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(recog_frame, text="Tolerance:").pack(anchor=tk.W)
        self.tolerance_var = tk.DoubleVar(value=0.6)
        tolerance_frame = ttk.Frame(recog_frame)
        tolerance_frame.pack(fill=tk.X)
        ttk.Scale(tolerance_frame, from_=0.1, to=1.0, variable=self.tolerance_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(tolerance_frame, textvariable=self.tolerance_var).pack(side=tk.RIGHT)
        
        ttk.Button(recog_frame, text="Recognize in Image", command=self.recognize_image).pack(fill=tk.X, pady=5)
        
        # Database info display
        self.db_info_text = scrolledtext.ScrolledText(left_panel, height=15, width=30)
        self.db_info_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Right panel for image display
        right_panel = ttk.Frame(recognition_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.recognition_image_label = ttk.Label(right_panel, text="No image loaded")
        self.recognition_image_label.pack(fill=tk.BOTH, expand=True)
    
    def create_landmarks_tab(self):
        """Create the facial landmarks tab."""
        landmarks_frame = ttk.Frame(self.notebook)
        self.notebook.add(landmarks_frame, text="Facial Landmarks")
        
        # Left panel for controls
        left_panel = ttk.Frame(landmarks_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Landmark options
        options_frame = ttk.LabelFrame(left_panel, text="Landmark Options")
        options_frame.pack(fill=tk.X, pady=5)
        
        self.show_landmarks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Landmarks", variable=self.show_landmarks_var).pack(anchor=tk.W)
        
        self.show_connections_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Connections", variable=self.show_connections_var).pack(anchor=tk.W)
        
        self.analyze_emotions_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Analyze Emotions", variable=self.analyze_emotions_var).pack(anchor=tk.W)
        
        self.detect_blinks_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Detect Blinks", variable=self.detect_blinks_var).pack(anchor=tk.W)
        
        ttk.Button(left_panel, text="Analyze Landmarks", command=self.analyze_landmarks).pack(fill=tk.X, pady=10)
        
        # Analysis results
        self.landmark_results_text = scrolledtext.ScrolledText(left_panel, height=15, width=30)
        self.landmark_results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Right panel for image display
        right_panel = ttk.Frame(landmarks_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.landmark_image_label = ttk.Label(right_panel, text="No image loaded")
        self.landmark_image_label.pack(fill=tk.BOTH, expand=True)
    
    def create_settings_tab(self):
        """Create the settings tab."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # General settings
        general_frame = ttk.LabelFrame(settings_frame, text="General Settings")
        general_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(general_frame, text="Default Image Directory:").pack(anchor=tk.W)
        self.image_dir_var = tk.StringVar()
        dir_frame = ttk.Frame(general_frame)
        dir_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(dir_frame, textvariable=self.image_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="Browse", command=self.browse_image_dir).pack(side=tk.RIGHT, padx=5)
        
        # Video settings
        video_frame = ttk.LabelFrame(settings_frame, text="Video Settings")
        video_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(video_frame, text="Camera Index:").pack(anchor=tk.W)
        self.camera_index_var = tk.IntVar(value=0)
        ttk.Spinbox(video_frame, from_=0, to=5, textvariable=self.camera_index_var).pack(anchor=tk.W)
        
        ttk.Label(video_frame, text="Video Resolution:").pack(anchor=tk.W)
        self.resolution_var = tk.StringVar(value="640x480")
        resolution_combo = ttk.Combobox(video_frame, textvariable=self.resolution_var, 
                                      values=["320x240", "640x480", "800x600", "1024x768"])
        resolution_combo.pack(anchor=tk.W)
        
        # Save/Load settings
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)
        ttk.Button(settings_frame, text="Load Settings", command=self.load_settings).pack(pady=5)
    
    def create_log_tab(self):
        """Create the log tab."""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Log")
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(log_frame, height=25, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(log_controls, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_controls, text="Save Log", command=self.save_log).pack(side=tk.LEFT, padx=5)
    
    def log_message(self, message: str):
        """Add a message to the log."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        """Clear the log."""
        self.log_text.delete(1.0, tk.END)
    
    def save_log(self):
        """Save the log to a file."""
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log_message(f"Log saved to {filename}")
    
    def load_image(self):
        """Load an image for processing."""
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if filename:
            self.current_image = cv2.imread(filename)
            if self.current_image is not None:
                self.display_image(self.current_image, self.image_label)
                self.log_message(f"Loaded image: {filename}")
            else:
                messagebox.showerror("Error", "Could not load image")
    
    def display_image(self, image: np.ndarray, label_widget: ttk.Label, max_size: tuple = (600, 400)):
        """Display an image in a label widget."""
        # Resize image if too large
        height, width = image.shape[:2]
        if width > max_size[0] or height > max_size[1]:
            scale = min(max_size[0] / width, max_size[1] / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        label_widget.configure(image=photo)
        label_widget.image = photo  # Keep a reference
    
    def process_image(self):
        """Process the current image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.is_processing:
            messagebox.showwarning("Warning", "Already processing an image")
            return
        
        self.is_processing = True
        self.log_message("Processing image...")
        
        # Run processing in a separate thread
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Process image in a separate thread."""
        try:
            results = {}
            result_image = self.current_image.copy()
            
            # Basic face detection
            if self.detect_faces_var.get():
                faces = self.image_detector.detect_faces(
                    self.current_image,
                    scale_factor=self.scale_factor_var.get(),
                    min_neighbors=self.min_neighbors_var.get()
                )
                results['faces'] = faces
                result_image = self.image_detector.draw_faces(result_image, faces)
            
            # Eye detection
            if self.detect_eyes_var.get() and 'faces' in results:
                eyes_per_face = self.image_detector.detect_eyes(self.current_image, results['faces'])
                results['eyes'] = eyes_per_face
                for eyes in eyes_per_face:
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(result_image, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            
            # Smile detection
            if self.detect_smiles_var.get() and 'faces' in results:
                smiles_per_face = self.image_detector.detect_smiles(self.current_image, results['faces'])
                results['smiles'] = smiles_per_face
                for smiles in smiles_per_face:
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(result_image, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            
            # Landmark detection
            if self.detect_landmarks_var.get() and 'faces' in results:
                landmarks_per_face = self.landmark_detector.detect_all_landmarks(self.current_image, results['faces'])
                results['landmarks'] = landmarks_per_face
                for landmarks in landmarks_per_face:
                    if landmarks is not None:
                        result_image = self.landmark_detector.draw_landmarks(result_image, landmarks)
                        result_image = self.landmark_detector.draw_landmark_connections(result_image, landmarks)
            
            # Update GUI in main thread
            self.root.after(0, self._update_results, result_image, results)
            
        except Exception as e:
            self.root.after(0, self._handle_error, str(e))
        finally:
            self.is_processing = False
    
    def _update_results(self, result_image: np.ndarray, results: Dict[str, Any]):
        """Update the GUI with processing results."""
        self.current_results = results
        
        # Display result image
        self.display_image(result_image, self.image_label)
        
        # Update results text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Processing Results:\n\n")
        
        if 'faces' in results:
            self.results_text.insert(tk.END, f"Faces detected: {len(results['faces'])}\n")
            for i, (x, y, w, h) in enumerate(results['faces']):
                self.results_text.insert(tk.END, f"  Face {i+1}: x={x}, y={y}, w={w}, h={h}\n")
        
        if 'eyes' in results:
            total_eyes = sum(len(eyes) for eyes in results['eyes'])
            self.results_text.insert(tk.END, f"Eyes detected: {total_eyes}\n")
        
        if 'smiles' in results:
            total_smiles = sum(len(smiles) for smiles in results['smiles'])
            self.results_text.insert(tk.END, f"Smiles detected: {total_smiles}\n")
        
        if 'landmarks' in results:
            valid_landmarks = sum(1 for lm in results['landmarks'] if lm is not None)
            self.results_text.insert(tk.END, f"Landmarks detected: {valid_landmarks}\n")
        
        self.log_message("Image processing completed")
    
    def _handle_error(self, error_message: str):
        """Handle processing errors."""
        messagebox.showerror("Error", f"Processing failed: {error_message}")
        self.log_message(f"Error: {error_message}")
        self.is_processing = False
    
    def save_result(self):
        """Save the current result image."""
        if self.current_results is None:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        if filename:
            # Get the result image from the current display
            # This is a simplified version - in practice, you'd store the result image
            messagebox.showinfo("Info", "Result saving not fully implemented in this demo")
    
    def start_webcam(self):
        """Start webcam face detection."""
        self.log_message("Starting webcam detection...")
        # This would start the video detection in a separate thread
        messagebox.showinfo("Info", "Webcam detection not fully implemented in this demo")
    
    def stop_detection(self):
        """Stop video detection."""
        self.video_detector.stop_detection()
        self.log_message("Video detection stopped")
    
    def load_video_file(self):
        """Load a video file for processing."""
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if filename:
            self.log_message(f"Video file selected: {filename}")
            # This would start video processing in a separate thread
    
    def add_faces_from_directory(self):
        """Add faces from a directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.log_message(f"Adding faces from directory: {directory}")
            results = self.recognizer.add_faces_from_directory(directory)
            self.log_message(f"Added {results['success']} faces, {results['failed']} failed")
    
    def add_face_from_image(self):
        """Add a face from an image."""
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if filename:
            name = tk.simpledialog.askstring("Input", "Enter person's name:")
            if name:
                image = cv2.imread(filename)
                if image is not None:
                    if self.recognizer.add_face(name, image):
                        self.log_message(f"Added face for {name}")
                    else:
                        self.log_message(f"Failed to add face for {name}")
    
    def train_from_video(self):
        """Train face recognition from video."""
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if filename:
            name = tk.simpledialog.askstring("Input", "Enter person's name:")
            if name:
                self.log_message(f"Training from video for {name}...")
                faces_added = self.recognizer.train_from_video(filename, name)
                self.log_message(f"Added {faces_added} faces for {name}")
    
    def view_database_info(self):
        """View face recognition database information."""
        info = self.recognizer.get_face_database_info()
        self.db_info_text.delete(1.0, tk.END)
        self.db_info_text.insert(tk.END, json.dumps(info, indent=2))
        self.log_message("Database info updated")
    
    def clear_database(self):
        """Clear the face recognition database."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the database?"):
            self.recognizer.clear_database()
            self.log_message("Database cleared")
            self.view_database_info()
    
    def recognize_image(self):
        """Recognize faces in the current image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.log_message("Recognizing faces...")
        result_image, recognized_faces = self.recognizer.recognize_and_draw(
            self.current_image, self.tolerance_var.get()
        )
        
        self.display_image(result_image, self.recognition_image_label)
        
        # Update results
        self.log_message(f"Recognized {len(recognized_faces)} faces:")
        for name, confidence, location in recognized_faces:
            self.log_message(f"  {name}: {confidence:.2f}")
    
    def analyze_landmarks(self):
        """Analyze facial landmarks in the current image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.log_message("Analyzing facial landmarks...")
        results = self.landmark_detector.comprehensive_analysis(self.current_image)
        
        # Display results
        self.landmark_results_text.delete(1.0, tk.END)
        self.landmark_results_text.insert(tk.END, f"Landmark Analysis Results:\n\n")
        self.landmark_results_text.insert(tk.END, f"Faces detected: {results['faces_detected']}\n\n")
        
        for analysis in results['face_analyses']:
            self.landmark_results_text.insert(tk.END, f"Face {analysis['face_id'] + 1}:\n")
            self.landmark_results_text.insert(tk.END, f"  Landmarks: {len(analysis['landmarks'])} points\n")
            self.landmark_results_text.insert(tk.END, f"  Blink info: {analysis['blink_info']}\n")
            self.landmark_results_text.insert(tk.END, f"  Head pose: {analysis['head_pose']}\n")
            self.landmark_results_text.insert(tk.END, f"  Emotions: {analysis['emotions']}\n\n")
        
        # Draw landmarks on image
        if results['face_analyses']:
            result_image = self.current_image.copy()
            for analysis in results['face_analyses']:
                landmarks = np.array(analysis['landmarks'])
                if self.show_landmarks_var.get():
                    result_image = self.landmark_detector.draw_landmarks(result_image, landmarks)
                if self.show_connections_var.get():
                    result_image = self.landmark_detector.draw_landmark_connections(result_image, landmarks)
            
            self.display_image(result_image, self.landmark_image_label)
        
        self.log_message("Landmark analysis completed")
    
    def browse_image_dir(self):
        """Browse for image directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.image_dir_var.set(directory)
    
    def save_settings(self):
        """Save application settings."""
        settings = {
            'image_directory': self.image_dir_var.get(),
            'camera_index': self.camera_index_var.get(),
            'resolution': self.resolution_var.get()
        }
        
        with open('face_detection_settings.json', 'w') as f:
            json.dump(settings, f, indent=2)
        
        self.log_message("Settings saved")
    
    def load_settings(self):
        """Load application settings."""
        try:
            with open('face_detection_settings.json', 'r') as f:
                settings = json.load(f)
            
            self.image_dir_var.set(settings.get('image_directory', ''))
            self.camera_index_var.set(settings.get('camera_index', 0))
            self.resolution_var.set(settings.get('resolution', '640x480'))
            
            self.log_message("Settings loaded")
        except FileNotFoundError:
            self.log_message("No settings file found, using defaults")
        except Exception as e:
            self.log_message(f"Error loading settings: {str(e)}")

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = FaceDetectionGUI(root)
    
    # Add import for simpledialog
    import tkinter.simpledialog
    
    root.mainloop()

if __name__ == "__main__":
    main()
