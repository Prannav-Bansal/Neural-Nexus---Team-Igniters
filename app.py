from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from src.inference import predict_image


st.set_page_config(page_title="Neural Nexus Disaster Management", layout="wide")
st.title("Neural Nexus Disaster Management")
st.caption("Upload a disaster image to get disaster type, severity, confidence, and Grad-CAM focus.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".png") as handle:
        handle.write(uploaded.read())
        temp_path = handle.name

    result = predict_image(temp_path)
    image = Image.open(temp_path)

    left, right = st.columns(2)
    with left:
        st.image(image, caption="Input image", use_container_width=True)
        st.metric("Disaster Type", result["predicted_disaster_type"])
        st.metric("Severity", result["predicted_severity_level"])
        st.metric("Confidence", f"{result['confidence_score']:.2%}")
        st.json(result)
    with right:
        st.image(result["gradcam_path"], caption="Grad-CAM", use_container_width=True)

else:
    st.info("Upload an image to run the trained model.")
