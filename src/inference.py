"""
inference.py

This module contains the PotholeDetector class, which encapsulates object detection
logic for both YOLOv8 and YOLOv11 models. It supports:
1. Loading .pt (PyTorch) or .onnx (ONNX Runtime) weights.
2. Single-image inference, bounding box visualization, and confidence scoring.
3. Batch directory inference with JSON logging.
4. Video processing with real-time FPS counter overlays.
5. Adjustable Confidence (conf) and Intersection over Union (IoU) thresholds.

Usage:
    from src.inference import PotholeDetector
    
    # Initialize detector
    detector = PotholeDetector(model_path="models/yolov11/weights/best.pt")
    
    # Run on single image
    annotated_img, results = detector.predict_image("dataset/images/test/road_test_000.jpg")
"""

import os
import cv2
import json
import time
import numpy as np
from ultralytics import YOLO

class PotholeDetector:
    def __init__(self, model_path):
        """
        Initializes the pothole detector by loading the model.
        Supports both .pt and .onnx formats.
        """
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        print(f"Loading detector model: {model_path} ...")
        # Ultralytics natively handles loading both .pt and .onnx files
        self.model = YOLO(model_path)
        print(f"Model loaded successfully!")
        
    def predict_image(self, img_input, conf_threshold=0.25, iou_threshold=0.45):
        """
        Runs inference on a single image.
        
        Args:
            img_input: Can be a path string (str) or a pre-loaded BGR image (np.ndarray).
            conf_threshold: Confidence threshold for predictions (0.0 to 1.0).
            iou_threshold: Overlap threshold for Non-Maximum Suppression (NMS).
            
        Returns:
            annotated_img: Image with bounding boxes and labels drawn.
            detections: List of dicts containing bounding box coords, confidence, and class name.
        """
        # Load image if a path is provided
        if isinstance(img_input, str):
            image = cv2.imread(img_input)
            if image is None:
                raise ValueError(f"Could not read image from path: {img_input}")
        else:
            image = img_input.copy()
            
        # Run prediction
        # stream=False returns a list of Results objects
        results_list = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        results = results_list[0]
        detections = []
        
        # Parse detections
        # results.boxes contains bounding boxes metadata
        if results.boxes is not None:
            for box in results.boxes:
                # Convert coords from tensor to numpy array
                # coords are in xyxy format (xmin, ymin, xmax, ymax)
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_id]
                
                detections.append({
                    "bbox": [round(coord, 2) for coord in xyxy],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name
                })
                
        # Draw bounding boxes on the image
        annotated_img = image.copy()
        for det in detections:
            xmin, ymin, xmax, ymax = map(int, det["bbox"])
            conf = det["confidence"]
            cls_name = det["class_name"]
            
            # Choose border color (Pothole = Reddish Orange)
            color = (0, 140, 255) # BGR format (orange-ish)
            
            # Draw rectangle bounding box
            cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Render label string
            label = f"{cls_name} {conf:.2f}"
            
            # Put label text above bounding box
            # Calculate text size to draw a background rectangle for label legibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background rectangle
            cv2.rectangle(
                annotated_img, 
                (xmin, ymin - text_h - 6), 
                (xmin + text_w + 4, ymin), 
                color, 
                -1 # Filled rectangle
            )
            
            # Draw white text over label background
            cv2.putText(
                annotated_img, 
                label, 
                (xmin + 2, ymin - 4), 
                font, 
                font_scale, 
                (255, 255, 255), 
                thickness, 
                cv2.LINE_AA
            )
            
        return annotated_img, detections
        
    def predict_folder(self, input_dir, output_dir, conf_threshold=0.25, iou_threshold=0.45, save_json=True):
        """
        Runs batch inference on all images in a directory.
        Saves annotated images to output_dir and writes statistics to a JSON log.
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        image_exts = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_exts)]
        
        print(f"Batch processing {len(image_files)} images from: {input_dir}")
        
        stats = {}
        total_inference_time = 0
        total_potholes = 0
        
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(input_dir, img_file)
            
            # Measure inference time
            t0 = time.time()
            annotated_img, detections = self.predict_image(
                img_path, 
                conf_threshold=conf_threshold, 
                iou_threshold=iou_threshold
            )
            t_inf = (time.time() - t0) * 1000 # convert to milliseconds
            total_inference_time += t_inf
            total_potholes += len(detections)
            
            # Save annotated image
            out_path = os.path.join(output_dir, img_file)
            cv2.imwrite(out_path, annotated_img)
            
            stats[img_file] = {
                "num_detections": len(detections),
                "inference_time_ms": round(t_inf, 2),
                "detections": detections
            }
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
                print(f"  Processed {idx + 1}/{len(image_files)} images...")
                
        # Consolidated Stats
        avg_inf_time = total_inference_time / len(image_files) if image_files else 0
        summary = {
            "model_path": self.model_path,
            "total_images": len(image_files),
            "total_detections": total_potholes,
            "average_inference_time_ms": round(avg_inf_time, 2),
            "average_fps": round(1000.0 / avg_inf_time, 2) if avg_inf_time > 0 else 0,
            "image_details": stats
        }
        
        if save_json:
            json_path = os.path.join(output_dir, "detection_statistics.json")
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"  + Saved detection statistics to: {json_path}")
            
        return summary
        
    def predict_video(self, video_path, output_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Processes a video file frame by frame, draws bounding boxes,
        overlays an FPS counter, and writes the output to a new video file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        
        # Read video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"  - Dimensions: {width}x{height} | Original FPS: {fps} | Total Frames: {total_frames}")
        
        # Define VideoWriter to save processed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        total_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            t0 = time.time()
            
            # Run inference
            annotated_frame, detections = self.predict_image(
                frame, 
                conf_threshold=conf_threshold, 
                iou_threshold=iou_threshold
            )
            
            elapsed = time.time() - t0
            total_time += elapsed
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            
            # Overlay Info: Bounding boxes count and FPS
            overlay_text = f"Potholes: {len(detections)} | FPS: {current_fps:.1f}"
            cv2.putText(
                annotated_frame, 
                overlay_text, 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 255, 0), # Bright Green for visibility
                2, 
                cv2.LINE_AA
            )
            
            # Write processed frame
            out.write(annotated_frame)
            
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"  Processed frame {frame_idx}/{total_frames} ({progress:.1f}%)")
                
        # Clean up
        cap.release()
        out.release()
        
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        print(f"  + Video processing complete! Saved to: {output_path}")
        print(f"  - Average FPS during processing: {avg_fps:.1f}")
        
        return {
            "output_path": output_path,
            "total_frames": frame_idx,
            "average_fps": round(avg_fps, 2)
        }
