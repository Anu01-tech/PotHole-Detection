import os
import sys
import json
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Add the current directory (src) to sys.path to resolve imports when running from other roots (like Streamlit)
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from feature_area_calculation import PotholeAreaCalculator

# Fix Windows console unicode issues (ensures we can print UTF-8 characters like emoji, 🔴, 🟠, 🟡, 🟢)
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

class PotholeSeverityClassifier:
    """
    Class responsible for estimating pothole depth, calculating risk scores,
    classifying severity levels, and generating inspection reports.
    """
    def __init__(self, model_path='models/yolov8/weights/best.pt'):
        # 1. Load the trained model weights. Fall back to pretrained yolov8n.pt if not found.
        if not os.path.exists(model_path):
            print(f"⚠️  Trained weights not found at {model_path}. Falling back to pre-trained 'yolov8n.pt'.")
            self.model = YOLO('yolov8n.pt')
        else:
            print(f"📥 Loading trained YOLOv8 model from: {model_path}")
            self.model = YOLO(model_path)
            
        # Initialize an area calculator instance for size calibration
        self.area_calculator = PotholeAreaCalculator(model_path)
        self.pixels_per_cm = 1.0 # Set from external calibration
        
        # Ensure our results save directory exists
        self.output_dir = "results/detections"
        os.makedirs(self.output_dir, exist_ok=True)

    def estimate_depth_from_shadow(self, image, bbox):
        """
        Part B: Depth Estimation.
        Approximates pothole depth from image shadow contrast and average brightness in the bounding box.
        """
        x1, y1, x2, y2 = bbox
        
        # Clip bounding box coordinates to ensure they fall within the image boundary
        h_img, w_img, _ = image.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        # Crop the pothole region
        crop = image[y1:y2, x1:x2]
        
        # If crop is empty, return default depth
        if crop.size == 0:
            return 1.0
            
        # Convert cropped image to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness (0-255)
        avg_brightness = np.mean(gray)
        
        # Calculate min and max brightness values
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        
        # Formula explained:
        # 1. Darkness factor: Deeper holes have darker interiors.
        #    We compute how much darker it is than mid-gray (128). Max factor capped at 1.0.
        darkness_factor = max(0, (128.0 - avg_brightness) / 128.0)
        
        # 2. Contrast factor: Deeper holes create sharper, high-contrast shadow boundaries.
        #    Calculated as difference between brightest and darkest pixels.
        contrast_factor = (max_val - min_val) / 255.0
        
        # 3. Estimated Depth = (darkness_factor * 10) + (contrast_factor * 5)
        estimated_depth = (darkness_factor * 10.0) + (contrast_factor * 5.0)
        
        # Ensure estimated depth is within reasonable visual range (0.5cm to 12.0cm)
        estimated_depth = max(0.5, min(12.0, estimated_depth))
        
        return float(estimated_depth)

    def calculate_risk_score(self, area_cm2, depth_cm, position='center'):
        """
        Part C: Risk Score Calculation.
        Calculates a numeric risk score out of 100 based on area, depth, and road placement.
        """
        # Risk factors:
        # - SIZE: 40% weight. Capped at 1500 cm2 for maximum size risk points (40)
        size_score = min(40.0, (area_cm2 / 1500.0) * 40.0)
        
        # - DEPTH: 40% weight. Capped at 10 cm for maximum depth risk points (40)
        depth_score = min(40.0, (depth_cm / 10.0) * 40.0)
        
        # - POSITION: 20% weight.
        position_mapping = {
            'center': 20.0,     # High risk: directly in vehicle wheel path
            'edge': 15.0,       # Medium-high risk: dangerous for motorcycles and cyclists
            'shoulder': 5.0     # Low risk: lower vehicle traffic
        }
        position_score = position_mapping.get(position.lower(), 5.0)
        
        # Combine weighted scores to get risk score out of 100
        risk_score = size_score + depth_score + position_score
        
        return float(risk_score)

    def get_severity_level(self, area_cm2, depth_cm):
        """
        Evaluates Area and Depth parameters against standard severity thresholds,
        and assigns the HIGHER of the two severity levels to be conservative.
        """
        # 1. Evaluate Area severity
        if area_cm2 < 100.0:
            area_severity = "LOW"
        elif area_cm2 < 500.0:
            area_severity = "MEDIUM"
        elif area_cm2 < 1000.0:
            area_severity = "HIGH"
        else:
            area_severity = "CRITICAL"
            
        # 2. Evaluate Depth severity
        if depth_cm < 2.0:
            depth_severity = "LOW"
        elif depth_cm < 5.0:
            depth_severity = "MEDIUM"
        elif depth_cm < 8.0:
            depth_severity = "HIGH"
        else:
            depth_severity = "CRITICAL"
            
        # 3. Take the higher of the two severities
        severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        
        if severity_rank[area_severity] >= severity_rank[depth_severity]:
            final_severity = area_severity
        else:
            final_severity = depth_severity
            
        # Assign action recommendation and fix timeframe parameters based on final severity
        severity_metadata = {
            "LOW": {
                "color": (0, 255, 0),       # Green
                "action": "Monitor condition",
                "timeframe": "3 months",
                "priority_label": "Priority 4"
            },
            "MEDIUM": {
                "color": (0, 255, 255),     # Yellow (BGR: 0, 255, 255)
                "action": "Schedule repair",
                "timeframe": "1 month",
                "priority_label": "Priority 3"
            },
            "HIGH": {
                "color": (0, 165, 255),     # Orange (BGR: 0, 165, 255)
                "action": "Urgent repair needed",
                "timeframe": "1 week",
                "priority_label": "Priority 2"
            },
            "CRITICAL": {
                "color": (0, 0, 255),       # Red
                "action": "IMMEDIATE REPAIR REQUIRED",
                "timeframe": "24 hours",
                "priority_label": "Priority 1"
            }
        }
        
        meta = severity_metadata[final_severity]
        return final_severity, meta

    def analyze_image(self, image_path, conf_threshold=0.5, road_position='center'):
        """
        Part D: Complete Analysis.
        Detects potholes, estimates area & depth, scores risks, and classifies severity.
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
            
        # Ensure our area calculator has the same pixels_per_cm ratio as configured
        self.area_calculator.pixels_per_cm = self.pixels_per_cm
        
        # Run inference using the loaded model
        results = self.model(img, conf=conf_threshold, imgsz=320, verbose=False)
        
        potholes_analysis = []
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        result = results[0]
        boxes = result.boxes
        
        print(f"\n🔍 Severity Analysis: {filename}")
        print("=" * 40)
        
        for idx, box in enumerate(boxes):
            # Extract coordinates as integers
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            
            # 1. Calculate area in cm²
            area_stats = self.area_calculator.calculate_pothole_area([x1, y1, x2, y2], img.shape)
            area_val = area_stats["ellipse_area_cm2"]
            
            # 2. Estimate depth from shadow analysis
            estimated_depth = self.estimate_depth_from_shadow(img, [x1, y1, x2, y2])
            
            # 3. Classify severity level based on area and depth
            severity_level, meta = self.get_severity_level(area_val, estimated_depth)
            severity_counts[severity_level] += 1
            
            # 4. Calculate risk score
            risk_score = self.calculate_risk_score(area_val, estimated_depth, road_position)
            
            # Store details for report generation
            potholes_analysis.append({
                "id": idx + 1,
                "confidence": conf,
                "box_coords": [x1, y1, x2, y2],
                "area_cm2": float(area_val),
                "estimated_depth_cm": float(estimated_depth),
                "severity": severity_level,
                "risk_score": float(risk_score),
                "recommendation": meta["action"],
                "timeframe": meta["timeframe"],
                "priority": meta["priority_label"]
            })
            
            # Print detailed console outputs
            print(f"\n🕳️  Pothole #{idx + 1}:")
            print(f"   Confidence: {conf*100:.1f}%")
            print(f"   Area: {area_val:.1f} cm²")
            print(f"   Estimated Depth: ~{estimated_depth:.1f} cm")
            print(f"   Severity: {severity_level}")
            print(f"   Risk Score: {risk_score:.1f}/100")
            print(f"   -> {meta['action']} (within {meta['timeframe']})")
            
            # 5. Color-code bounding box on the output image based on severity
            cv2.rectangle(img, (x1, y1), (x2, y2), meta["color"], 2)
            
            # 6. Draw the label text "P1: HIGH (Risk:75) - 650cm²"
            label = f"P{idx + 1}: {severity_level} (Risk:{risk_score:.0f}) - {area_val:.0f}cm2"
            (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            # Draw a solid background rectangle in severity color
            cv2.rectangle(img, (x1, y1 - h_lbl - 5), (x1 + w_lbl + 5, y1), meta["color"], -1)
            # Draw black text on top of the solid color background
            cv2.putText(img, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
            
        # Determine overall road assessment category
        if severity_counts["CRITICAL"] > 0:
            overall_assessment = "CRITICAL"
        elif severity_counts["HIGH"] > 0:
            overall_assessment = "POOR"
        elif severity_counts["MEDIUM"] > 0:
            overall_assessment = "FAIR"
        else:
            overall_assessment = "GOOD"
            
        # Draw a summary panel on the top-left of the image
        # Add a semi-transparent black rectangle background
        cv2.rectangle(img, (10, 10), (390, 110), (0, 0, 0), -1)
        cv2.putText(img, f"Overall Status: {overall_assessment}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display the counts of each severity level
        summary_details = f"R:{severity_counts['CRITICAL']}  O:{severity_counts['HIGH']}  Y:{severity_counts['MEDIUM']}  G:{severity_counts['LOW']}"
        cv2.putText(img, summary_details, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Display recommendations
        if overall_assessment in ["CRITICAL", "POOR"]:
            rec_str = "⚠️ URGENT REPAIRS MANDATED"
            rec_color = (0, 0, 255) # Red text
        elif overall_assessment == "FAIR":
            rec_str = "📅 SCHEDULE MAINTENANCE"
            rec_color = (0, 255, 255) # Yellow text
        else:
            rec_str = "🟢 ROAD HEALTH IS SATISFACTORY"
            rec_color = (0, 255, 0) # Green text
        cv2.putText(img, rec_str, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rec_color, 1, cv2.LINE_AA)
        
        # 7. Draw a Legend in the bottom-left corner of the image
        h_img, w_img, _ = img.shape
        cv2.rectangle(img, (10, h_img - 110), (220, h_img - 10), (0, 0, 0), -1)
        cv2.putText(img, "SEVERITY LEGEND", (20, h_img - 92), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        legend_items = [
            ("Critical (24h)", (0, 0, 255)),
            ("High (1wk)", (0, 165, 255)),
            ("Medium (1mo)", (0, 255, 255)),
            ("Low (Monitor)", (0, 255, 0))
        ]
        for idx, (label_text, color_bgr) in enumerate(legend_items):
            y_offset = h_img - 72 + (idx * 16)
            # Draw a colored square representation
            cv2.rectangle(img, (20, y_offset - 8), (30, y_offset + 2), color_bgr, -1)
            # Write the legend label next to the square
            cv2.putText(img, label_text, (38, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            
        print("\n📊 SEVERITY SUMMARY:")
        print(f"   🔴 Critical: {severity_counts['CRITICAL']}")
        print(f"   🟠 High:     {severity_counts['HIGH']}")
        print(f"   🟡 Medium:   {severity_counts['MEDIUM']}")
        print(f"   🟢 Low:      {severity_counts['LOW']}")
        print(f"\n📋 OVERALL ASSESSMENT: {overall_assessment}")
        
        analysis_summary = {
            "image": filename,
            "overall_assessment": overall_assessment,
            "severity_summary": severity_counts,
            "potholes": potholes_analysis
        }
        
        return img, analysis_summary

    def generate_report(self, analysis_results, output_path='results/detections/severity_report.json'):
        """
        Part E: Report Generation.
        Generates a professional, structured JSON report documenting the analysis details.
        """
        # Build global list of actionable recommendations for the report
        recommendations = []
        summary = analysis_results["severity_summary"]
        
        if summary["CRITICAL"] > 0:
            recommendations.append("🚨 IMMEDIATE: Critical potholes detected. Close road sections or apply patches within 24 hours.")
        if summary["HIGH"] > 0:
            recommendations.append("⚠️ URGENT: High severity potholes found. Dispatch maintenance crew within 1 week.")
        if summary["MEDIUM"] > 0:
            recommendations.append("📅 SCHEDULE: Medium severity potholes noted. Schedule repairs within the next 30 days.")
        if len(recommendations) == 0:
            recommendations.append("🟢 MONITOR: Only low severity potholes detected. Monitor during routine patrol schedules.")
            
        report_data = {
            "report_title": "Pothole Severity Analysis & Road Health Assessment",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": analysis_results["image"],
            "total_potholes": sum(summary.values()),
            "severity_summary": summary,
            "overall_assessment": analysis_results["overall_assessment"],
            "potholes": analysis_results["potholes"],
            "recommendations": recommendations
        }
        
        # Save JSON file with clean indent indentation formatting
        try:
            # Ensure folder exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4)
            print(f"\n📄 Report saved successfully: {output_path}")
        except Exception as e:
            print(f"\n❌ Error generating report: {e}")
            
        return report_data

if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE 2: POTHOLE SEVERITY CLASSIFICATION DEMO")
    print("=" * 60)
    
    # Instantiate the classifier using our trained model path
    classifier = PotholeSeverityClassifier('models/yolov8/weights/best.pt')
    
    # Configure the pixels_per_cm ratio from our Feature 1 manual calibration
    # Camera at 100cm height and 60 degree FOV gives ~5.54 pixels per cm
    classifier.pixels_per_cm = 5.54
    
    # Run the severity classifier on our test image
    try:
        annotated_img, analysis = classifier.analyze_image('test_image.jpg')
        
        # Generate the JSON report
        report = classifier.generate_report(analysis)
        
        # Save annotated image
        output_file = 'results/detections/severity_analysis.jpg'
        cv2.imwrite(output_file, annotated_img)
        print(f"\n💾 Annotated output image saved successfully to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("Tip: Make sure 'test_image.jpg' exists in your project root directory.")
