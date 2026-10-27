import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="RGB–Thermal Object Detection",
    layout="centered"
)

st.title("🔥 RGB–Thermal Object Detection")
st.caption("YOLOv11 + CBAM + VQC Fusion")

# ----------------------------
# Load model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/abl_vqc_real/weights/best.pt")

model = load_model()

# ----------------------------
# Upload image
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Input Image")
    st.image(image, use_column_width=True)

    if st.button("🔍 Detect Objects"):
        with st.spinner("Running detection..."):
            results = model(image_np, conf=0.25)

            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.subheader("Detection Result")
        st.image(annotated, use_column_width=True)

