import os
import sys
import math
import cv2
import numpy as np
from ultralytics import YOLO

# Fix Windows console unicode issues (ensures we can print UTF-8 characters like emoji, 📏, 🕳️)
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

class PotholeAreaCalculator:
    """
    Class responsible for calculating the physical, real-world area of 
    potholes detected by YOLOv8 using camera calibration systems.
    """
    def __init__(self, model_path='models/yolov8/weights/best.pt'):
        # 1. Load the trained model weights. Fall back to pretrained yolov8n.pt if not found.
        if not os.path.exists(model_path):
            print(f"⚠️  Trained weights not found at {model_path}. Falling back to pre-trained 'yolov8n.pt'.")
            self.model = YOLO('yolov8n.pt')
        else:
            print(f"📥 Loading trained YOLOv8 model from: {model_path}")
            self.model = YOLO(model_path)
            
        # Default calibration ratio (1.0 means 1 pixel = 1 cm, which is just a placeholder)
        self.pixels_per_cm = 1.0
        
        # Ensure our results save directory exists
        self.output_dir = "results/detections"
        os.makedirs(self.output_dir, exist_ok=True)

    def set_manual_calibration(self, camera_height_cm=100, camera_fov_degrees=60, image_size_pixels=640):
        """
        Method 1: Manual Calibration.
        Calculates the pixels_per_cm ratio using the physical setup parameters of the camera.
        """
        # Convert field of view degrees to radians for Python's math.tan function
        fov_radians = math.radians(camera_fov_degrees)
        
        # Calculate the real-world horizontal width of the ground visible to the camera
        # Formula: ground_width = 2 * height * tan(FOV / 2)
        ground_width_cm = 2.0 * camera_height_cm * math.tan(fov_radians / 2.0)
        
        # Calculate ratio of pixels per physical centimeter
        self.pixels_per_cm = image_size_pixels / ground_width_cm
        
        print("\n📏 Manual Calibration Completed:")
        print(f"   Camera Height: {camera_height_cm} cm")
        print(f"   Field of View: {camera_fov_degrees}°")
        print(f"   Estimated Ground Coverage Width: {ground_width_cm:.2f} cm")
        print(f"   Calculated Ratio: {self.pixels_per_cm:.4f} pixels per cm")
        
        return self.pixels_per_cm

    def calibrate_with_reference(self, reference_size_cm, reference_line_pixels):
        """
        Method 2: Reference-Object Calibration.
        Calculates pixels_per_cm ratio directly using a reference object of a known size in the image.
        """
        if reference_size_cm <= 0 or reference_line_pixels <= 0:
            raise ValueError("Reference dimensions must be greater than zero.")
            
        # Calculate ratio: pixels_per_cm = pixel length / physical length
        self.pixels_per_cm = reference_line_pixels / reference_size_cm
        
        print("\n📏 Reference Calibration Completed:")
        print(f"   Reference Object Size: {reference_size_cm} cm")
        print(f"   Reference Size in Pixels: {reference_line_pixels} px")
        print(f"   Calculated Ratio: {self.pixels_per_cm:.4f} pixels per cm")
        
        return self.pixels_per_cm

    def calculate_pothole_area(self, bbox_xyxy, image_shape):
        """
        Computes the real-world physical width, height, and area of a pothole
        bounding box using the calibrated pixels_per_cm ratio.
        """
        # Extract coordinates from bounding box tensor
        x1, y1, x2, y2 = bbox_xyxy
        
        # Calculate size in pixels
        pixel_width = x2 - x1
        pixel_height = y2 - y1
        
        # Convert dimensions to centimeters using the pixels_per_cm ratio
        real_width_cm = pixel_width / self.pixels_per_cm
        real_height_cm = pixel_height / self.pixels_per_cm
        
        # Calculate Area using Rectangular formula (Reference)
        rect_area_cm2 = real_width_cm * real_height_cm
        
        # Calculate Area using Ellipse formula (Primary, since potholes are round-ish)
        # Formula: Area = pi * (width / 2) * (height / 2)
        ellipse_area_cm2 = math.pi * (real_width_cm / 2.0) * (real_height_cm / 2.0)
        
        # Classify the size based on the primary ellipse area
        if ellipse_area_cm2 < 100:
            size_category = "Small"
        elif ellipse_area_cm2 < 500:
            size_category = "Medium"
        elif ellipse_area_cm2 < 1000:
            size_category = "Large"
        else:
            size_category = "Very Large"
            
        return {
            "pixel_width": float(pixel_width),
            "pixel_height": float(pixel_height),
            "real_width_cm": float(real_width_cm),
            "real_height_cm": float(real_height_cm),
            "ellipse_area_cm2": float(ellipse_area_cm2),
            "rect_area_cm2": float(rect_area_cm2),
            "size_category": size_category
        }

    def detect_with_area(self, image_path, conf_threshold=0.5):
        """
        Runs YOLOv8 detection and applies area calculations to each pothole,
        drawing bounding boxes and area labels on the image.
        """
        # Load image from file path or use directly if it is a numpy array (from web app upload)
        if isinstance(image_path, np.ndarray):
            img = image_path.copy()
            filename = "uploaded_image.jpg"
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"❌ Failed to load image at: {image_path}")
            filename = os.path.basename(image_path)

        # Retrieve image shape (height, width, channels)
        h_img, w_img, _ = img.shape
        
        # Run inference using the loaded model
        # We specify imgsz=320 to match the training resolution
        results = self.model(img, conf=conf_threshold, imgsz=320, verbose=False)
        
        potholes_data = []
        total_damaged_area = 0.0
        
        # Extract the results from the model
        result = results[0]
        boxes = result.boxes
        
        print(f"\n🔍 Analyzing: {filename}")
        print(f"✅ Found {len(boxes)} potholes")
        
        for idx, box in enumerate(boxes):
            # Extract coordinates as integers
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            
            # Calculate physical dimensions
            area_stats = self.calculate_pothole_area([x1, y1, x2, y2], img.shape)
            area_val = area_stats["ellipse_area_cm2"]
            category = area_stats["size_category"]
            total_damaged_area += area_val
            
            # Save data to return list
            potholes_data.append({
                "id": idx + 1,
                "confidence": conf,
                "box_coords": [x1, y1, x2, y2],
                "area_stats": area_stats
            })
            
            # Print individual detection stats
            print(f"\n🕳️  Pothole #{idx + 1}:")
            print(f"   Confidence: {conf*100:.1f}%")
            print(f"   Pixel size: {int(area_stats['pixel_width'])}x{int(area_stats['pixel_height'])} px")
            print(f"   Real size: {area_stats['real_width_cm']:.1f}x{area_stats['real_height_cm']:.1f} cm")
            print(f"   Area: {area_val:.1f} cm²")
            print(f"   Category: {category}")
            
            # 1. Draw a green bounding box around the pothole (BGR: 0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 2. Draw the label "P1: 156 cm² (Medium)" at the top of the box
            label = f"P{idx + 1}: {area_val:.0f} cm2 ({category})"
            (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            # Draw a solid background rectangle behind the text label for visibility
            cv2.rectangle(img, (x1, y1 - h_lbl - 5), (x1 + w_lbl + 5, y1), (0, 255, 0), -1)
            # Write the black text over the solid green label background
            cv2.putText(img, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            
        # 3. Draw a summary panel in the top-left corner
        # Add a semi-transparent black rectangle background
        cv2.rectangle(img, (10, 10), (380, 75), (0, 0, 0), -1)
        summary_text1 = f"Total: {len(boxes)} potholes"
        summary_text2 = f"Damaged Area: {total_damaged_area:.1f} cm2"
        cv2.putText(img, summary_text1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, summary_text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        
        print(f"\n📊 Summary:")
        print(f"   Total potholes: {len(boxes)}")
        print(f"   Total damaged area: {total_damaged_area:.1f} cm²")
        
        return img, potholes_data

if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE 1: POTHOLE AREA CALCULATION DEMO")
    print("=" * 60)
    
    # Instantiate the calculator using our trained model path
    calculator = PotholeAreaCalculator('models/yolov8/weights/best.pt')
    
    # Calibrate using standard camera setup parameters:
    # Camera height = 100cm, FOV = 60 degrees, and image size = 640px
    calculator.set_manual_calibration(
        camera_height_cm=100,
        camera_fov_degrees=60,
        image_size_pixels=640
    )
    
    # Run the calculator on our test image
    try:
        annotated_img, detections = calculator.detect_with_area('test_image.jpg')
        
        # Save output image
        output_file = 'results/detections/area_analysis.jpg'
        cv2.imwrite(output_file, annotated_img)
        print(f"\n💾 Annotated output image saved successfully to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("Tip: Make sure 'test_image.jpg' exists in your project root directory.")
