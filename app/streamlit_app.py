import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import tempfile

# Page config
st.set_page_config(page_title="Pothole Detection", layout="wide")

st.title("🚧 Pothole Detection System")
st.write("Upload an image to detect potholes")

# Find the model file
@st.cache_resource
def load_model():
    model_paths = [
        "models/yolov8/weights/best.pt",
        "yolov8n.pt",
        "yolo11n.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            st.success(f"✅ Loading model from: {path}")
            return YOLO(path)
    
    st.error("⚠️ No model file found! Please upload a YOLO model.")
    return None

model = load_model()

# File upload
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
)

if uploaded_file is not None and model is not None:
    # Read image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Display original
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Image", use_container_width=True)
    
    # Run detection
    with st.spinner("🔍 Detecting potholes..."):
        results = model(image_np)
    
    # Display results
    if results:
        result = results[0]
        
        # Get annotated image
        annotated = result.plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        col2.image(annotated_rgb, caption="Detection Results", use_container_width=True)
        
        # Show details
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            st.success(f"✅ Found {len(boxes)} pothole(s)!")
            
            # Show confidence scores
            for i, box in enumerate(boxes):
                conf = float(box.conf[0]) * 100
                cls = int(box.cls[0])
                st.write(f"Pothole {i+1}: {conf:.1f}% confidence")
        else:
            st.info("✅ No potholes detected in this image")
    else:
        st.error("❌ No detection results")

# Footer
st.markdown("---")
st.caption("Built with YOLO and Streamlit 🚀")
