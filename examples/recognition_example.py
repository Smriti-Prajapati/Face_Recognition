"""
Face Recognition Example
This example demonstrates face recognition and identification capabilities.
"""

import cv2
import os
import json
from face_recognition import FaceRecognizer

def main():
    """Run face recognition example."""
    print("Face Recognition Example")
    print("=" * 40)
    
    # Initialize recognizer
    recognizer = FaceRecognizer()
    
    print("Face Recognition System")
    print("1. Add faces from directory")
    print("2. Add face from image")
    print("3. Recognize faces in image")
    print("4. Train from video")
    print("5. View database info")
    print("6. Test recognition accuracy")
    
    choice = input("Enter your choice (1-6): ").strip()
    
    if choice == "1":
        add_faces_from_directory(recognizer)
    elif choice == "2":
        add_face_from_image(recognizer)
    elif choice == "3":
        recognize_faces_in_image(recognizer)
    elif choice == "4":
        train_from_video(recognizer)
    elif choice == "5":
        view_database_info(recognizer)
    elif choice == "6":
        test_recognition_accuracy(recognizer)
    else:
        print("Invalid choice")

def add_faces_from_directory(recognizer):
    """Add faces from a directory of images."""
    print("\nAdd Faces from Directory")
    print("-" * 30)
    
    directory = input("Enter directory path: ").strip()
    
    if not directory or not os.path.exists(directory):
        print("Invalid directory path")
        return
    
    print(f"Adding faces from: {directory}")
    results = recognizer.add_faces_from_directory(directory)
    
    print(f"Results:")
    print(f"  Successfully added: {results['success']}")
    print(f"  Failed: {results['failed']}")
    
    # Show database info
    info = recognizer.get_face_database_info()
    print(f"\nDatabase now contains {info['total_faces']} faces for {info['unique_people']} people")

def add_face_from_image(recognizer):
    """Add a face from a single image."""
    print("\nAdd Face from Image")
    print("-" * 30)
    
    name = input("Enter person's name: ").strip()
    image_path = input("Enter image path: ").strip()
    
    if not name or not image_path or not os.path.exists(image_path):
        print("Invalid input")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image")
        return
    
    print(f"Adding face for {name}...")
    if recognizer.add_face(name, image):
        print(f"Successfully added face for {name}")
    else:
        print(f"Failed to add face for {name}")

def recognize_faces_in_image(recognizer):
    """Recognize faces in an image."""
    print("\nRecognize Faces in Image")
    print("-" * 30)
    
    image_path = input("Enter image path: ").strip()
    
    if not image_path or not os.path.exists(image_path):
        print("Invalid image path")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image")
        return
    
    # Get tolerance
    tolerance = float(input("Enter recognition tolerance (0.1-1.0, default 0.6): ") or "0.6")
    
    print("Recognizing faces...")
    result_image, recognized_faces = recognizer.recognize_and_draw(image, tolerance)
    
    print(f"\nRecognition Results:")
    print(f"Found {len(recognized_faces)} face(s)")
    
    for i, (name, confidence, location) in enumerate(recognized_faces):
        print(f"  Face {i+1}: {name} (confidence: {confidence:.2f})")
    
    # Save result
    output_path = "recognition_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"\nResult saved as: {output_path}")
    
    # Display result
    cv2.imshow("Face Recognition", result_image)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def train_from_video(recognizer):
    """Train face recognition from a video file."""
    print("\nTrain from Video")
    print("-" * 30)
    
    video_path = input("Enter video path: ").strip()
    person_name = input("Enter person's name: ").strip()
    max_frames = int(input("Enter max frames to process (default 50): ") or "50")
    
    if not video_path or not person_name or not os.path.exists(video_path):
        print("Invalid input")
        return
    
    print(f"Training from video: {video_path}")
    print(f"Person: {person_name}")
    print(f"Max frames: {max_frames}")
    
    faces_added = recognizer.train_from_video(video_path, person_name, max_frames)
    print(f"Added {faces_added} faces for {person_name}")

def view_database_info(recognizer):
    """View face recognition database information."""
    print("\nDatabase Information")
    print("-" * 30)
    
    info = recognizer.get_face_database_info()
    
    print(f"Total faces: {info['total_faces']}")
    print(f"Unique people: {info['unique_people']}")
    print(f"People in database: {', '.join(info['people'])}")
    
    print("\nFace counts per person:")
    for person, count in info['face_counts'].items():
        print(f"  {person}: {count} faces")
    
    # Save info to file
    with open("database_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nDatabase info saved to: database_info.json")

def test_recognition_accuracy(recognizer):
    """Test recognition accuracy with known faces."""
    print("\nTest Recognition Accuracy")
    print("-" * 30)
    
    # This is a simplified test - in practice, you'd have a test dataset
    test_dir = input("Enter test directory path: ").strip()
    
    if not test_dir or not os.path.exists(test_dir):
        print("Invalid test directory")
        return
    
    print("Testing recognition accuracy...")
    
    correct_predictions = 0
    total_predictions = 0
    
    # Process test images
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Extract expected name from filename
            expected_name = os.path.splitext(filename)[0]
            
            image_path = os.path.join(test_dir, filename)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Recognize faces
                recognized_faces = recognizer.recognize_faces(image)
                
                if recognized_faces:
                    predicted_name = recognized_faces[0][0]  # First face
                    confidence = recognized_faces[0][1]
                    
                    print(f"  {filename}: Expected {expected_name}, Got {predicted_name} (confidence: {confidence:.2f})")
                    
                    if predicted_name == expected_name:
                        correct_predictions += 1
                    
                    total_predictions += 1
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\nAccuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    else:
        print("No test images found")

def create_sample_database():
    """Create a sample database for testing."""
    print("\nCreate Sample Database")
    print("-" * 30)
    
    # This would create a sample database with known faces
    # In practice, you'd have a dataset of labeled faces
    
    print("To create a sample database:")
    print("1. Create a directory called 'known_faces'")
    print("2. Add images named after the person (e.g., 'john.jpg', 'jane.jpg')")
    print("3. Run the 'Add faces from directory' option")
    print("4. Use the 'Recognize faces in image' option to test")

def batch_recognition_example():
    """Example of batch recognition processing."""
    print("\nBatch Recognition Example")
    print("-" * 30)
    
    recognizer = FaceRecognizer()
    
    # Process multiple images
    input_dir = "test_images"
    output_dir = "recognition_results"
    
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing images in: {input_dir}")
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Recognize faces
                result_image, recognized_faces = recognizer.recognize_and_draw(image)
                
                # Save result
                output_path = os.path.join(output_dir, f"recognized_{filename}")
                cv2.imwrite(output_path, result_image)
                
                print(f"Processed {filename}: {len(recognized_faces)} faces")

if __name__ == "__main__":
    main()
    
    # Uncomment to run additional examples
    # create_sample_database()
    # batch_recognition_example()
