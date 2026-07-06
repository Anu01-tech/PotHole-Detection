import os
import sys
import time
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Fix Windows console unicode issues
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

class PotholeDetector:
    """
    YOLOv8 Pothole Detector class that supports inference on single images,
    videos, and batches of images.
    """
    def __init__(self, model_path='models/yolov8/weights/best.pt'):
        """Initializes the detector and loads the YOLOv8 model."""
        if not os.path.exists(model_path):
            # If the best model is not found, fallback to the pre-trained nano model
            print(f"⚠️  Trained weights not found at {model_path}. Falling back to pre-trained 'yolov8n.pt'.")
            self.model = YOLO('yolov8n.pt')
        else:
            print(f"📥 Loading trained YOLOv8 model from: {model_path}")
            self.model = YOLO(model_path)
            
        # Ensure output directory exists
        self.output_dir = "results/detections"
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_image(self, image_path, conf_threshold=0.5, save=True):
        """
        Runs pothole detection on a single image.
        
        Args:
            image_path (str): Path to the input image file.
            conf_threshold (float): Confidence score threshold to filter weak detections.
            save (bool): If True, saves the annotated image to results/detections/.
            
        Returns:
            annotated_image (numpy.ndarray): OpenCV image with overlays.
            pothole_count (int): Number of detected potholes.
            inference_time_ms (float): Inference time in milliseconds.
        """
        # Read the image or use it directly if it's already a numpy array
        if isinstance(image_path, np.ndarray):
            img = image_path.copy()
            filename = "uploaded_image.jpg"
        else:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Failed to load image at: {image_path}")
                return None, 0, 0.0
            filename = os.path.basename(image_path)

        # Track inference time
        start_time = time.time()
        # Run inference using the model. We specify imgsz=320 as the model was trained on 320x320.
        results = self.model(img, conf=conf_threshold, imgsz=320, verbose=False)
        inference_time_ms = (time.time() - start_time) * 1000.0

        pothole_count = 0
        
        # YOLOv8 returns a list of Results objects (one per image)
        for r in results:
            boxes = r.boxes
            pothole_count += len(boxes)
            
            for box in boxes:
                # Extract coordinates as integers
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                
                # Draw GREEN bounding box (BGR color: 0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence label just above the box
                label = f"Pothole: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)  # Label background
                cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw summary panel overlay in the top-left corner
        # Add a semi-transparent black rectangle background for high legibility
        cv2.rectangle(img, (10, 10), (280, 75), (0, 0, 0), -1)
        # Overlay counts and timing
        cv2.putText(img, f"Potholes Found: {pothole_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"Inference Time: {inference_time_ms:.1f} ms", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Save annotated image if requested
        if save:
            # Add a 'det_' prefix to distinguish detections
            output_path = os.path.join(self.output_dir, f"det_{filename}")
            cv2.imwrite(output_path, img)
            
        return img, pothole_count, inference_time_ms

    def detect_video(self, video_path, conf_threshold=0.5, save=True):
        """
        Runs pothole detection on a video file frame-by-frame and displays real-time FPS.
        
        Args:
            video_path (str): Path to the input video file.
            conf_threshold (float): Confidence score threshold.
            save (bool): If True, saves the processed video to results/detections/.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video at: {video_path}")
            return

        # Read video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 30.0  # Fallback FPS
            
        print(f"🎞️  Processing Video: {video_path} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")
        
        writer = None
        if save:
            filename = os.path.basename(video_path)
            output_path = os.path.join(self.output_dir, f"det_{filename}")
            # Use 'mp4v' or 'MJPG' codec. 'mp4v' is highly portable for MP4 files.
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # We'll use a tqdm progress bar to monitor progress in the console
        pbar = tqdm(total=total_frames, desc="Processing Video Frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # Run inference
            results = self.model(frame, conf=conf_threshold, imgsz=320, verbose=False)
            
            # Calculate metrics
            end_time = time.time()
            inference_time = end_time - start_time
            actual_fps = 1.0 / inference_time if inference_time > 0 else fps
            
            pothole_count = 0
            for r in results:
                boxes = r.boxes
                pothole_count += len(boxes)
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    
                    # Draw green bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw confidence score
                    label = f"Pothole: {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw visual metrics overlay
            cv2.rectangle(frame, (10, 10), (220, 75), (0, 0, 0), -1)
            cv2.putText(frame, f"Potholes: {pothole_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {actual_fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if writer:
                writer.write(frame)
                
            pbar.update(1)
            
        pbar.close()
        cap.release()
        if writer:
            writer.release()
            print(f"💾 Processed video saved to: {output_path}")

    def batch_detect(self, folder_path, conf_threshold=0.5):
        """
        Processes all images in a folder and saves the outputs.
        
        Args:
            folder_path (str): Path to the folder containing test images.
            conf_threshold (float): Confidence threshold.
        """
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return
            
        # Get list of valid images
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                       if f.lower().endswith(valid_extensions)]
                       
        if not image_files:
            print(f"⚠️  No images found in: {folder_path}")
            return
            
        print(f"📂 Batch processing {len(image_files)} images from: {folder_path}...")
        
        total_potholes = 0
        total_time_ms = 0.0
        
        for img_path in tqdm(image_files, desc="Batch Detecting"):
            _, count, t_ms = self.detect_image(img_path, conf_threshold=conf_threshold, save=True)
            total_potholes += count
            total_time_ms += t_ms
            
        avg_time = total_time_ms / len(image_files)
        avg_potholes = total_potholes / len(image_files)
        
        print("\n📊 Batch Processing Summary:")
        print(f"   Processed Images:   {len(image_files)}")
        print(f"   Total Potholes:     {total_potholes}")
        print(f"   Avg Potholes/Img:   {avg_potholes:.2f}")
        print(f"   Avg Inference Time: {avg_time:.1f} ms/image")
        print(f"💾 All detections saved in: {self.output_dir}")


def generate_test_media():
    """Generates synthetic media (test_image.jpg and road_video.mp4) if not present."""
    from prepare_dataset import create_synthetic_road_image
    
    # 1. Generate test_image.jpg
    if not os.path.exists("test_image.jpg"):
        img, _ = create_synthetic_road_image()
        cv2.imwrite("test_image.jpg", img)
        print("Generated synthetic 'test_image.jpg' for inference.")
        
    # 2. Generate road_video.mp4 (moving lane markings and moving potholes)
    if not os.path.exists("road_video.mp4"):
        print("Generating synthetic 'road_video.mp4' (this will take a moment)...")
        # Define video properties
        width, height = 640, 640
        fps = 30
        duration_sec = 3
        total_frames = fps * duration_sec
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter("road_video.mp4", fourcc, fps, (width, height))
        
        # Potholes starting coordinates and radii
        # We simulate them moving down the screen to represent driving forward
        potholes = [
            {'cx': 200, 'cy': 100, 'rx': 30, 'ry': 20, 'speed': 6},
            {'cx': 450, 'cy': 250, 'rx': 40, 'ry': 25, 'speed': 8},
            {'cx': 300, 'cy': -50, 'rx': 25, 'ry': 15, 'speed': 5}
        ]
        
        from prepare_dataset import generate_rough_ellipse_points
        
        for frame_idx in range(total_frames):
            # Asphalt background
            img = np.full((height, width, 3), 80, dtype=np.uint8)
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Dashed center line that shifts downwards over time to simulate motion
            lane_color = (220, 220, 220)
            lane_width = 8
            lane_length = 40
            gap_length = 30
            y_offset = (frame_idx * 10) % (lane_length + gap_length)
            
            for y in range(-lane_length + y_offset, height, lane_length + gap_length):
                cv2.rectangle(img, (320 - lane_width // 2, y), (320 + lane_width // 2, y + lane_length), lane_color, -1)
                
            # Side solid road lines
            cv2.line(img, (60, 0), (60, height), (200, 200, 200), 4)
            cv2.line(img, (580, 0), (580, height), (200, 200, 200), 4)
            
            # Draw moving potholes
            for p in potholes:
                # Update pothole y coordinate
                current_cy = p['cy'] + frame_idx * p['speed']
                # If pothole moves off the screen, wrap it around
                if current_cy > height + 50:
                    current_cy = -50
                    p['cx'] = np.random.randint(100, 540)
                
                # Make the pothole size scale up slightly as it gets "closer" (lower on the screen)
                scale = 1.0 + (current_cy / float(height)) * 0.5
                rx_scaled = int(p['rx'] * scale)
                ry_scaled = int(p['ry'] * scale)
                
                # Generate points only if within visual bounds
                if -50 < current_cy < height + 50:
                    pts = generate_rough_ellipse_points(p['cx'], current_cy, rx_scaled, ry_scaled, num_points=24, noise_level=0.15)
                    # Rim
                    cv2.drawContours(img, [pts], -1, (50, 50, 50), thickness=2)
                    # Cavity
                    cv2.fillPoly(img, [pts], (25, 25, 25))
                    
            video_writer.write(img)
            
        video_writer.release()
        print("Generated synthetic 'road_video.mp4'.")


def main():
    print("🚗 Pothole Detection System - YOLOv8 Inference Mode")
    print("-" * 50)
    
    # 1. Generate test media (synthetic test_image.jpg and road_video.mp4) if not present
    # This guarantees that the script is completely runnable immediately without downloading external data
    generate_test_media()
    print("-" * 50)

    # 2. Instantiate detector
    # By default, loads models/yolov8/weights/best.pt
    detector = PotholeDetector()
    print("-" * 50)

    # 3. Detect on single image
    print("🖼️  Running detection on 'test_image.jpg'...")
    _, count, inference_time = detector.detect_image("test_image.jpg", conf_threshold=0.4, save=True)
    print(f"   Done! Found {count} potholes in {inference_time:.1f} ms.")
    print(f"   Output saved to: results/detections/det_test_image.jpg")
    print("-" * 50)

    # 4. Batch process test split folder
    test_folder = "dataset/images/test"
    if os.path.exists(test_folder):
        detector.batch_detect(test_folder, conf_threshold=0.4)
    else:
        print(f"⚠️  Test folder not found at '{test_folder}'. Make sure to run prepare_dataset.py.")
    print("-" * 50)

    # 5. Detect on video
    print("🎥 Running detection on 'road_video.mp4'...")
    detector.detect_video("road_video.mp4", conf_threshold=0.4, save=True)
    print("-" * 50)
    
    print("\n🎉 Detection pipeline completed successfully! Check the 'results/detections/' folder.")

if __name__ == "__main__":
    main()
