import os
import sys
import time
import tempfile
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Add project root directory to sys.path to ensure src imports work properly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the PotholeDetector class from our core detect.py
from src.detect import PotholeDetector

# Fix Windows console unicode issues
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# --- Page Configurations & Styling ---
st.set_page_config(
    page_title="Pothole Detection Hub - YOLOv8",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (dark mode themed headers with clean visual cards)
st.markdown("""
    <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #1e3d59 0%, #ff6e40 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2px;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 25px;
        }
        .metric-container {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            flex: 1;
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 5px solid #ff6e40;
            text-align: center;
        }
        .metric-val {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1e3d59;
        }
        .metric-lbl {
            font-size: 0.85rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #1e3d59 0%, #17b978 100%);
            color: white;
            border: None;
            padding: 10px 20px;
            font-weight: bold;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            opacity: 0.9;
            transform: translateY(-2px);
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.markdown("<div style='text-align: center;'><img src='https://img.icons8.com/color/96/road-worker.png' width='80'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='text-align: center;'>YOLOv8 Config Panel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Model Path Setup
model_path = "models/yolov8/weights/best.pt"
model_exists = os.path.exists(model_path)

if not model_exists:
    st.sidebar.warning("⚠️ Trained weights 'best.pt' not found. App will fallback to pre-trained weights.")
    
# Detection threshold sliders
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=1.00,
    value=0.30,
    step=0.01,
    help="Minimum score required to classify a bounding box as a pothole."
)

analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    options=["Standard Detection", "Area Calculation", "Severity Classification"],
    help="Select standard detection or advanced area/severity analysis features."
)



# --- Main Interface ---
st.markdown("<div class='main-title'>Road Pothole Detection Hub</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Road Inspection & Verification Dashboard powered by YOLOv8</div>", unsafe_allow_html=True)

# Cache detector instantiation to prevent reloading model on every parameter change
@st.cache_resource
def get_detector(weights):
    return PotholeDetector(weights)

detector = get_detector(model_path)

# Upload Section
uploaded_file = st.file_uploader(
    "Upload a road image or video to perform real-time pothole detection:",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # --- IMAGE PIPELINE ---
    if file_extension in ["jpg", "jpeg", "png"]:
        # Decode uploaded image
        image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        # Run detection based on the selected mode
        analysis_summary = None
        if analysis_mode == "Standard Detection":
            annotated_bgr, count, latency = detector.detect_image(
                img_bgr, 
                conf_threshold=conf_threshold, 
                save=False
            )
        elif analysis_mode == "Area Calculation":
            # Manual calibration (100cm height, 60 degrees FOV)
            detector.area_calculator.set_manual_calibration(100, 60, 640)
            annotated_bgr, potholes_data = detector.area_calculator.detect_with_area(
                img_bgr, 
                conf_threshold=conf_threshold
            )
            count = len(potholes_data)
        else: # Severity Classification
            detector.severity_classifier.pixels_per_cm = detector.area_calculator.pixels_per_cm
            annotated_bgr, analysis_summary = detector.severity_classifier.analyze_image(
                img_bgr, 
                conf_threshold=conf_threshold
            )
            count = len(analysis_summary["potholes"])
        
        # Layout columns
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Convert BGR to RGB for Streamlit display
            img_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Annotated Detection Output", use_column_width=True)
            
        with col2:
            st.markdown("### Performance & Metrics")
            
            # Display stats in custom CSS cards
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{count}</div>
                    <div class='metric-lbl'>Potholes Detected</div>
                </div>
            """, unsafe_allow_html=True)
            
            # If Severity Classification is chosen, generate the JSON report and display it
            if analysis_mode == "Severity Classification" and analysis_summary is not None:
                report = detector.severity_classifier.generate_report(analysis_summary)
                st.markdown("---")
                st.markdown("### Severity Report")
                st.json(report)
                
            st.markdown("---")
            
            # Allow user to download the processed image
            _, encoded_img = cv2.imencode('.jpg', annotated_bgr)
            st.download_button(
                label="📥 Download Annotated Image",
                data=encoded_img.tobytes(),
                file_name=f"det_{uploaded_file.name}",
                mime="image/jpeg"
            )

    # --- VIDEO PIPELINE ---
    elif file_extension == "mp4":
        st.markdown("### Video Inference Preview")
        
        # Create a temporary file to save the uploaded video bytes
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create layout columns
        col1, col2 = st.columns([3, 1])
        
        with col1:
            frame_placeholder = st.empty()
            
        with col2:
            st.markdown("### Real-Time Analytics")
            pothole_card = st.empty()
            progress_bar = st.progress(0.0)
            
        frame_idx = 0
        
        # Loop over frames and display in real time
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # Perform detection
            results = detector.model(frame, conf=conf_threshold, imgsz=320, verbose=False)
            
            latency = time.time() - start_time
            fps = 1.0 / latency if latency > 0 else 30.0
            
            pothole_count = 0
            for r in results:
                boxes = r.boxes
                pothole_count += len(boxes)
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    
                    # Draw green box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw label
                    label = f"Pothole: {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw visual metrics on the frame (only Potholes Count panel)
            cv2.rectangle(frame, (10, 10), (220, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"Potholes: {pothole_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Update streamlit view (Convert BGR to RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_column_width=True)
            
            # Update metrics cards
            pothole_card.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{pothole_count}</div>
                    <div class='metric-lbl'>Current Potholes</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Update progress
            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            
        cap.release()
        os.unlink(tfile.name)
        st.success("🎉 Video processing complete!")

else:
    # App landing message when no file is uploaded
    st.info("👋 Welcome! Please upload a road image (.jpg, .png) or video (.mp4) from the file uploader above to begin detection.")
