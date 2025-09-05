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
    df = pd.read_csv("./file/data.csv", dtype=str)

    # Clean up the 12_digit_id column
    df["12_digit_id"] = (
        df["12_digit_id"]
        .str.replace(r"\.0$", "", regex=True)   # remove trailing .0 if present
        .str.replace(",", "", regex=True)       # remove commas
    )

    # Convert scientific notation like 7.78268E+11 → 778268000000
    def fix_id(x):
        try:
            return str(int(float(x)))
        except:
            return x

    df["12_digit_id"] = df["12_digit_id"].apply(fix_id)

    return df




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

with st.form("register_form", clear_on_submit=True):
    cattle_id = st.text_input("Enter 12-digit Cattle ID (numbers only)")
    cattle_name = st.text_input("Enter Cattle Name")
    images = st.file_uploader(
        "Upload at least 4 images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    submitted = st.form_submit_button("Submit")

    if submitted:
        # --- Validation ---
        if not cattle_id:
            st.error("⚠️ Please enter a 12-digit ID.")
        elif not cattle_id.isdigit() or len(cattle_id) != 12:
            st.error("⚠️ ID must be exactly 12 numeric digits.")
        elif cattle_collection.find_one({"12_digit_id": cattle_id}):
            st.error("⚠️ This ID already exists. Please use a unique one.")
        elif not cattle_name:
            st.error("⚠️ Please enter a cattle name.")
        elif len(images) < 4:
            st.error("⚠️ Please upload at least 4 images.")
        else:
            # --- Save ---
            image_list = [base64.b64encode(file.read()).decode("utf-8") for file in images]

            cattle_collection.insert_one({
                "12_digit_id": cattle_id,
                "cattle_name": cattle_name,
                "images": image_list,
                "created_at": datetime.utcnow().isoformat()
            })

            st.success(f"✅ Registered {cattle_name} with ID {cattle_id}")

            # Show previews
            for img_data in image_list:
                st.image(Image.open(io.BytesIO(base64.b64decode(img_data))), use_container_width=True)
