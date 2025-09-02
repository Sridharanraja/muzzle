import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLOv11 classification model
model = YOLO("./models/best_25.pt")

st.title("Muzzle Identification")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    results = model.predict(image)

    # Extract top class with highest confidence
    top_result = results[0].probs
    class_id = int(top_result.top1)
    confidence = float(top_result.top1conf)
    class_name = model.names[class_id]

    # Display result
    st.success(f"Predicted Class: **{class_name}**") #(Confidence: {confidence:.2f})
