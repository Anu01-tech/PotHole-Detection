"""
streamlit_app.py

This Streamlit web application provides an interactive user interface for the
Pothole Detection System. Features include:
1. File upload for images and videos (.jpg, .jpeg, .png, .mp4).
2. Two operational modes: Single Model view vs. Side-by-Side Model comparison.
3. Sliders for real-time adjustments of Confidence and IoU NMS thresholds.
4. Live processing with visual feedback, inference latency logs, and FPS counters.
5. Export capabilities to download annotated results.

Usage:
    # Run from the project root directory:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import cv2
import tempfile
import numpy as np
from PIL import Image
import streamlit as st

# Add project root to python path to ensure imports work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.inference import PotholeDetector

# --- Page Configurations & Styling ---
st.set_page_config(
    page_title="Pothole Detection Hub: YOLOv11 vs. YOLOv8",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
    <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #1f385c 0%, #e67e22 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 5px solid #1f385c;
            text-align: center;
        }
        .metric-card.v8 { border-left-color: #4A90E2; }
        .metric-card.v11 { border-left-color: #E67E22; }
        .metric-val {
            font-size: 1.6rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-lbl {
            font-size: 0.9rem;
            color: #7f8c8d;
            text-transform: uppercase;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.image("https://img.icons8.com/color/96/road-worker.png", width=80)
st.sidebar.title("Configuration Panel")

# Mode Selection
app_mode = st.sidebar.selectbox(
    "Select App Mode",
    ["Single Model Inference", "Side-by-Side Comparison"]
)

# Model paths
yolov8_path = "models/yolov8/weights/best.pt"
yolov11_path = "models/yolov11/weights/best.pt"

# Validate model weight files
models_exist = os.path.exists(yolov8_path) and os.path.exists(yolov11_path)

if not models_exist:
    st.sidebar.error("⚠️ Model weights not found! Please run the training scripts first.")
    
# Confidence and IoU NMS Threshold Sliders
# Default to 0.05 since the models are trained for 5 validation epochs on CPU
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=1.00,
    value=0.05,
    step=0.01,
    help="Minimum confidence score required to display a detection. Lower value for under-trained models."
)

iou_threshold = st.sidebar.slider(
    "NMS IoU Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.45,
    step=0.05,
    help="Intersection over Union (IoU) overlap limit. Lower values suppress overlapping boxes more aggressively."
)

# Sidebar System Info
st.sidebar.markdown("---")
st.sidebar.subheader("System Info")
st.sidebar.info(
    "💡 **Quick Dry-Run Mode Active**\n\n"
    "The models were trained for **5 epochs** on a synthetic dataset for CPU compatibility. "
    "To visualize detections, keep the **Confidence Threshold** low (e.g., `0.01 - 0.08`)."
)

# --- Main Board Header ---
st.markdown("<div class='main-title'>Road Pothole Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Comparative Analysis Dashboard: YOLOv11 (Original) vs. YOLOv8 (Baseline)</div>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a road image or video for pothole detection",
    type=["jpg", "jpeg", "png", "mp4"]
)

# Helper function to cache model loading
@st.cache_resource
def load_detector(model_path):
    return PotholeDetector(model_path)

# --- App Execution Flow ---
if uploaded_file is not None and models_exist:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    # Initialize detectors
    detector_v8 = load_detector(yolov8_path)
    detector_v11 = load_detector(yolov11_path)
    
    # ------------------ IMAGE PROCESSING ------------------
    if file_type in ['jpg', 'jpeg', 'png']:
        # Load uploaded image bytes
        image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        # Single Model Inference Mode
        if app_mode == "Single Model Inference":
            selected_model_name = st.selectbox("Choose Model", ["YOLOv11 Nano", "YOLOv8 Nano"])
            
            detector = detector_v11 if selected_model_name == "YOLOv11 Nano" else detector_v8
            
            st.write("Running detection...")
            
            # Predict
            t0 = time.time()
            annotated_bgr, detections = detector.predict_image(
                img_bgr, 
                conf_threshold=conf_threshold, 
                iou_threshold=iou_threshold
            )
            latency = (time.time() - t0) * 1000 # ms
            fps = 1000.0 / latency if latency > 0 else 0
            
            # Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Convert BGR to RGB for Streamlit display
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption=f"Processed image ({selected_model_name})", use_column_width=True)
                
            with col2:
                st.subheader("Performance Metrics")
                
                # Metric Cards
                st.markdown(f"""
                    <div class='metric-card {"v11" if "11" in selected_model_name else "v8"}'>
                        <div class='metric-val'>{len(detections)}</div>
                        <div class='metric-lbl'>Potholes Detected</div>
                    </div>
                    <br>
                    <div class='metric-card {"v11" if "11" in selected_model_name else "v8"}'>
                        <div class='metric-val'>{latency:.1f} ms</div>
                        <div class='metric-lbl'>Inference Latency</div>
                    </div>
                    <br>
                    <div class='metric-card {"v11" if "11" in selected_model_name else "v8"}'>
                        <div class='metric-val'>{fps:.1f} FPS</div>
                        <div class='metric-lbl'>Equivalent Speed</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Raw detections json
                st.markdown("---")
                st.subheader("Raw JSON Output")
                st.json(detections)
                
                # Download Button
                is_success, buffer = cv2.imencode(".jpg", annotated_bgr)
                if is_success:
                    st.download_button(
                        label=f"📥 Download Annotated Image",
                        data=buffer.tobytes(),
                        file_name=f"potholes_{selected_model_name.replace(' ', '_').lower()}_{uploaded_file.name}",
                        mime="image/jpeg"
                    )
                    
        # Side-by-Side Comparison Mode
        else:
            st.write("Running dual model detection comparison...")
            
            # Predict YOLOv8
            t0 = time.time()
            annotated_bgr_v8, detections_v8 = detector_v8.predict_image(
                img_bgr, 
                conf_threshold=conf_threshold, 
                iou_threshold=iou_threshold
            )
            latency_v8 = (time.time() - t0) * 1000
            
            # Predict YOLOv11
            t1 = time.time()
            annotated_bgr_v11, detections_v11 = detector_v11.predict_image(
                img_bgr, 
                conf_threshold=conf_threshold, 
                iou_threshold=iou_threshold
            )
            latency_v11 = (time.time() - t1) * 1000
            
            # Layout: columns
            col_v8, col_v11 = st.columns(2)
            
            with col_v8:
                st.subheader("YOLOv8 Nano")
                rgb_v8 = cv2.cvtColor(annotated_bgr_v8, cv2.COLOR_BGR2RGB)
                st.image(rgb_v8, use_column_width=True)
                
                st.markdown(f"""
                    <div class='metric-card v8'>
                        <div class='metric-val'>{len(detections_v8)} potholes</div>
                        <div class='metric-lbl'>Latency: {latency_v8:.1f} ms | FPS: {1000.0/latency_v8:.1f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_v11:
                st.subheader("YOLOv11 Nano (Winner)")
                rgb_v11 = cv2.cvtColor(annotated_bgr_v11, cv2.COLOR_BGR2RGB)
                st.image(rgb_v11, use_column_width=True)
                
                st.markdown(f"""
                    <div class='metric-card v11'>
                        <div class='metric-val'>{len(detections_v11)} potholes</div>
                        <div class='metric-lbl'>Latency: {latency_v11:.1f} ms | FPS: {1000.0/latency_v11:.1f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            # Stitched comparison download
            st.markdown("---")
            h, w, c = img_bgr.shape
            canvas = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
            canvas[:, :w] = annotated_bgr_v8
            canvas[:, w + 10:] = annotated_bgr_v11
            
            is_success, buffer = cv2.imencode(".jpg", canvas)
            if is_success:
                st.download_button(
                    label="📥 Download Side-by-Side Comparison Plot",
                    data=buffer.tobytes(),
                    file_name=f"yolov8_vs_yolov11_{uploaded_file.name}",
                    mime="image/jpeg"
                )
                
    # ------------------ VIDEO PROCESSING ------------------
    elif file_type == 'mp4':
        st.info("🎥 Video Processing Active. Uploaded file registered.")
        
        # Save uploaded file to temp path
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Output temp path
        out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_temp.close()
        
        if app_mode == "Single Model Inference":
            selected_model_name = st.selectbox("Choose Model", ["YOLOv11 Nano", "YOLOv8 Nano"])
            detector = detector_v11 if selected_model_name == "YOLOv11 Nano" else detector_v8
            
            if st.button("🚀 Process and Render Video"):
                with st.spinner("Processing video frame-by-frame on CPU... please wait."):
                    progress_bar = st.progress(0.0)
                    
                    # Open video stream
                    cap = cv2.VideoCapture(tfile.name)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # VideoWriter
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(out_temp.name, fourcc, fps, (width, height))
                    
                    frame_idx = 0
                    total_time = 0
                    
                    # Metric holders
                    potholes_per_frame = []
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_idx += 1
                        t_start = time.time()
                        
                        # Inference
                        annotated_frame, detections = detector.predict_image(
                            frame, 
                            conf_threshold=conf_threshold, 
                            iou_threshold=iou_threshold
                        )
                        elapsed = time.time() - t_start
                        total_time += elapsed
                        potholes_per_frame.append(len(detections))
                        
                        # Draw overlay
                        cv2.putText(
                            annotated_frame,
                            f"{selected_model_name} | Potholes: {len(detections)} | FPS: {1.0/elapsed:.1f}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA
                        )
                        
                        out.write(annotated_frame)
                        
                        # Update progress bar
                        progress_bar.progress(frame_idx / total_frames)
                        
                    cap.release()
                    out.release()
                    
                    avg_fps = frame_idx / total_time if total_time > 0 else 0
                    avg_potholes = np.mean(potholes_per_frame) if potholes_per_frame else 0
                    
                    st.success("Video processing complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-val'>{avg_potholes:.1f}</div>
                                <div class='metric-lbl'>Avg Potholes Detected</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-val'>{avg_fps:.1f} FPS</div>
                                <div class='metric-lbl'>Average Process Speed</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    # Download video
                    with open(out_temp.name, "rb") as f:
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=f.read(),
                            file_name=f"potholes_{selected_model_name.replace(' ', '_').lower()}_{uploaded_file.name}",
                            mime="video/mp4"
                        )
        else:
            st.warning("⚠️ Side-by-Side video stitching is not supported directly in the browser due to CPU encoding times. "
                       "Please use 'Single Model Inference' mode for video analysis.")
            
        # Clean up temp files
        os.unlink(tfile.name)
        
else:
    # --- Landing Page Info ---
    st.info("ℹ️ Upload an image or video above to begin pothole detection predictions.")
    
    # Display some project metrics details
    st.markdown("### Model Architectural Spec Sheet")
    st.table([
        {"Model": "YOLOv8 Nano (Base)", "Parameters": "3.01 Million", "FLOPs": "8.1 GFLOPs", "Size": "5.96 MB", "Backbone": "C2f"},
        {"Model": "YOLOv11 Nano (New)", "Parameters": "2.58 Million", "FLOPs": "6.3 GFLOPs", "Size": "5.21 MB", "Backbone": "C3k2 (Winner)"}
    ])
