import time

import streamlit as st
from PIL import Image

from helpers import MemesClassification


# Load model
@st.cache_resource
def load_model(model_path, image_size=224, device_inference="cpu"):
    model = MemesClassification(
        model_path, (image_size, image_size), device_inference.lower()
    )
    return model


# Set page config
st.set_page_config(
    page_title="Memes Classification",
    page_icon="😱",
)

device_inference = st.sidebar.selectbox("Select device", ("CPU", "CUDA"))
path_model = st.sidebar.text_input(
    "Path to model", "./saved_model_memes_classification.onnx"
)
model = load_model(path_model, device_inference=device_inference)

# Main page
st.title("Memes Classification")
st.write(
    """
      Memes classification is the process of categorizing internet memes into different groups based on their features, such as the image, caption, and context. This task is commonly performed using machine learning techniques, where an algorithm is trained on a large dataset of labeled memes to identify the characteristics that define different meme categories. The goal of memes classification is to enable the automatic organization and discovery of memes, as well as to support the development of meme-related applications such as search engines, recommendation systems, and analysis tools. The output of a memes classification model it's binary, true, or false.
"""
)
st.markdown("  ")

# Run load model
uploaded_file = st.file_uploader(
    "Upload image file", type=["jpg", "jpeg", "png", "bmp", "tiff"]
)
if uploaded_file is not None:
    uploaded_file = Image.open(uploaded_file).convert("RGB")
    st.markdown("  ")
    st.write("Source Image")
    st.image(uploaded_file)

    predict_button = st.button("Predict image")
    st.markdown("  ")

    if predict_button:
        with st.spinner("Wait for it..."):
            start_time = time.perf_counter()
            predict_label = model.predict(uploaded_file)
            st.write(
                f"Inference time: {(time.perf_counter() - start_time):.3f} seconds"
            )
            st.write(f"Predict label: {predict_label}")
