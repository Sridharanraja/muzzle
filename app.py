import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import io, base64, random
from pymongo import MongoClient

# -----------------------
# Connect to MongoDB
# -----------------------
@st.cache_resource
def get_db():
    client = MongoClient(st.secrets["mongodb"]["uri"])
    db = client["cattle_db"]
    return db

db = get_db()
cattle_collection = db["cattle_images"]

# -----------------------
# Load Data.csv
# -----------------------
@st.cache_data
def load_csv():
    return pd.read_csv("./file/data.csv")  # Ensure this file is in your repo

cattle_df = load_csv()

# -----------------------
# Load YOLOv11 classification model
# -----------------------
model = YOLO("./models/best_25.pt")

st.title("🐄 YOLOv11 Cattle Classification App")

# UI Confidence threshold slider
ui_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# -----------------------
# File uploader for Prediction
# -----------------------
uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO prediction
    results = model.predict(image)
    top_result = results[0].probs
    class_id = int(top_result.top1)
    confidence = float(top_result.top1conf)
    class_name = model.names[class_id]

    if confidence < 0.90:
        st.error("⚠️ Data not available in DB for reliable classification.")
    else:
        if confidence >= ui_threshold:
            st.success(f"Predicted Class: **{class_name}** (Confidence: {confidence:.2f})")

            # Lookup in CSV
            row = cattle_df[cattle_df["class"] == class_name]
            if not row.empty:
                st.write("📌 **Mapped Result from Database:**")
                st.table(row[["12_digit_id", "cattle_name", "class"]])
            else:
                st.warning("⚠️ Predicted class not found in CSV mapping.")
        else:
            st.warning("No prediction passed the selected confidence threshold.")

# -----------------------
# Register New Cattle
# -----------------------
st.subheader("➕ Register New Cattle")

def generate_unique_id():
    """Generate a unique 12-digit ID"""
    return str(random.randint(100000000000, 999999999999))

if st.button("Register New Cattle"):
    with st.form("register_form", clear_on_submit=True):
        cattle_name = st.text_input("Enter Cattle Name")
        images = st.file_uploader(
            "Upload at least 4 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not cattle_name:
                st.error("⚠️ Please enter a cattle name.")
            elif len(images) < 4:
                st.error("⚠️ Please upload at least 4 images.")
            else:
                # Generate ID
                cattle_id = generate_unique_id()

                # Save images to MongoDB
                image_list = []
                for file in images:
                    file_bytes = file.read()
                    image_list.append(base64.b64encode(file_bytes).decode("utf-8"))

                cattle_collection.insert_one({
                    "12_digit_id": cattle_id,
                    "cattle_name": cattle_name,
                    "images": image_list
                })

                st.success(f"✅ Registered {cattle_name} with ID {cattle_id}")

                # Show uploaded images
                for img_data in image_list:
                    st.image(Image.open(io.BytesIO(base64.b64decode(img_data))), use_column_width=True)
