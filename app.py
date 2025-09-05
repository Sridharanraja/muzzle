# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# import pandas as pd
# import io, base64, random
# from pymongo import MongoClient

# # -----------------------
# # Connect to MongoDB
# # -----------------------
# @st.cache_resource
# def get_db():
#     client = MongoClient(st.secrets["mongodb"]["uri"])
#     db = client["cattle_db"]
#     return db

# db = get_db()
# cattle_collection = db["cattle_images"]

# # -----------------------
# # Load Data.csv
# # -----------------------
# @st.cache_data
# def load_csv():
#     df = pd.read_csv("./file/data.csv", dtype=str)

#     # Clean up the 12_digit_id column
#     df["12_digit_id"] = (
#         df["12_digit_id"]
#         .str.replace(r"\.0$", "", regex=True)   # remove trailing .0 if present
#         .str.replace(",", "", regex=True)       # remove commas
#     )

#     # Convert scientific notation like 7.78268E+11 → 778268000000
#     def fix_id(x):
#         try:
#             return str(int(float(x)))
#         except:
#             return x

#     df["12_digit_id"] = df["12_digit_id"].apply(fix_id)

#     return df




# cattle_df = load_csv()

# # -----------------------
# # Load YOLOv11 classification model
# # -----------------------
# model = YOLO("./models/best_25.pt")

# st.title("🐄 YOLOv11 Cattle Classification App")

# # UI Confidence threshold slider
# ui_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# # -----------------------
# # File uploader for Prediction
# # -----------------------
# uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Run YOLO prediction
#     results = model.predict(image)
#     top_result = results[0].probs
#     class_id = int(top_result.top1)
#     confidence = float(top_result.top1conf)
#     class_name = model.names[class_id]

#     if confidence < 0.90:
#         st.error("⚠️ Data not available in DB for reliable classification.")
#     else:
#         if confidence >= ui_threshold:
#             st.success(f"Predicted Class: **{class_name}** (Confidence: {confidence:.2f})")

#             # Lookup in CSV
#             row = cattle_df[cattle_df["class"] == class_name]
#             if not row.empty:
#                 st.write("📌 **Mapped Result from Database:**")
#                 st.table(row[["12_digit_id", "cattle_name", "class"]])
#             else:
#                 st.warning("⚠️ Predicted class not found in CSV mapping.")
#         else:
#             st.warning("No prediction passed the selected confidence threshold.")

# # -----------------------
# # Register New Cattle
# # -----------------------
# st.subheader("➕ Register New Cattle")

# with st.form("register_form", clear_on_submit=True):
#     cattle_id = st.text_input("Enter 12-digit Cattle ID (numbers only)")
#     cattle_name = st.text_input("Enter Cattle Name")
#     images = st.file_uploader(
#         "Upload at least 4 images", 
#         type=["jpg", "jpeg", "png"], 
#         accept_multiple_files=True
#     )
#     submitted = st.form_submit_button("Submit")

#     if submitted:
#         # --- Validation ---
#         if not cattle_id:
#             st.error("⚠️ Please enter a 12-digit ID.")
#         elif not cattle_id.isdigit() or len(cattle_id) != 12:
#             st.error("⚠️ ID must be exactly 12 numeric digits.")
#         elif cattle_collection.find_one({"12_digit_id": cattle_id}):
#             st.error("⚠️ This ID already exists. Please use a unique one.")
#         elif not cattle_name:
#             st.error("⚠️ Please enter a cattle name.")
#         elif len(images) < 4:
#             st.error("⚠️ Please upload at least 4 images.")
#         else:
#             # --- Save ---
#             image_list = [base64.b64encode(file.read()).decode("utf-8") for file in images]

#             cattle_collection.insert_one({
#                 "12_digit_id": cattle_id,
#                 "cattle_name": cattle_name,
#                 "images": image_list,
#                 "created_at": datetime.utcnow().isoformat()
#             })

#             st.success(f"✅ Registered {cattle_name} with ID {cattle_id}")

#             # Show previews
#             for img_data in image_list:
#                 st.image(Image.open(io.BytesIO(base64.b64decode(img_data))), use_container_width=True)















import streamlit as st
from pymongo import MongoClient, ASCENDING
from datetime import datetime
import base64, io, zipfile, pandas as pd
from PIL import Image
import os

# -----------------------
# DB connection (use your st.secrets)
# -----------------------
@st.cache_resource
def get_db():
    client = MongoClient(st.secrets["mongodb"]["uri"], connectTimeoutMS=20000, serverSelectionTimeoutMS=20000)
    db = client["cattle_db"]
    # Ensure indexes for uniqueness and fast queries
    db["cattle_images"].create_index([("12_digit_id", ASCENDING)], unique=True)
    return db

db = get_db()
cattle_collection = db["cattle_images"]

# -----------------------
# Helpers: Save, Get, List
# -----------------------
def save_new_cattle(cattle_id: str, cattle_name: str, image_files):
    """
    Saves a new cattle doc to MongoDB.
    image_files: list of UploadedFile objects from Streamlit
    Stores images as list of base64 strings under 'images' and also stores image filenames
    """
    created_at = datetime.utcnow().isoformat()
    image_entries = []
    for i, f in enumerate(image_files, start=1):
        raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        # choose a filename so downloads are meaningful
        ext = os.path.splitext(f.name)[1] or ".jpg"
        filename = f"{cattle_id}_{i}{ext}"
        image_entries.append({"filename": filename, "b64": b64})

    doc = {
        "12_digit_id": cattle_id,
        "cattle_name": cattle_name,
        "images": image_entries,   # list of dicts {filename, b64}
        "created_at": created_at
    }
    # Insert (unique constraint ensured by index); if conflict, raise
    cattle_collection.insert_one(doc)
    return doc

def get_cattle_by_id(cattle_id: str):
    return cattle_collection.find_one({"12_digit_id": cattle_id})

def list_cattle(filter_id=None, filter_name=None, limit=100):
    q = {}
    if filter_id:
        q["12_digit_id"] = {"$regex": f"^{filter_id}"}  # partial or full match
    if filter_name:
        q["cattle_name"] = {"$regex": filter_name, "$options": "i"}
    cursor = cattle_collection.find(q).sort("created_at", -1).limit(limit)
    return list(cursor)

# -----------------------
# Helpers: create CSV bytes and ZIP bytes
# -----------------------
def create_metadata_csv_bytes(docs):
    """
    Given a list of cattle docs, create a CSV bytes object containing:
    12_digit_id, cattle_name, created_at, image_filename
    (one row per image)
    """
    rows = []
    for d in docs:
        cid = d.get("12_digit_id")
        name = d.get("cattle_name")
        created = d.get("created_at")
        for img in d.get("images", []):
            rows.append({
                "12_digit_id": cid,
                "cattle_name": name,
                "created_at": created,
                "image_filename": img.get("filename")
            })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.read()

def create_images_zip_bytes(docs):
    """
    Create an in-memory ZIP containing all images of provided docs.
    Filenames preserved from doc['images'][i]['filename'].
    Returns bytes of zip file.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for d in docs:
            for img in d.get("images", []):
                filename = img.get("filename")
                b64 = img.get("b64")
                raw = base64.b64decode(b64)
                z.writestr(filename, raw)
    buf.seek(0)
    return buf.read()

# -----------------------
# UI: Register New Cattle (with 12-digit numeric input only)
# -----------------------
st.header("➕ Register New Cattle")

with st.form("register_form", clear_on_submit=True):
    cattle_id = st.text_input("Enter 12-digit Cattle ID (numbers only)")
    cattle_name = st.text_input("Enter Cattle Name")
    images = st.file_uploader(
        "Upload at least 4 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Validation
        if not cattle_id:
            st.error("⚠️ Please enter a 12-digit ID.")
        elif not cattle_id.isdigit() or len(cattle_id) != 12:
            st.error("⚠️ ID must be exactly 12 numeric digits.")
        elif cattle_collection.find_one({"12_digit_id": cattle_id}):
            st.error("⚠️ This ID already exists. Please use a unique one.")
        elif not cattle_name:
            st.error("⚠️ Please enter a cattle name.")
        elif not images or len(images) < 4:
            st.error("⚠️ Please upload at least 4 images.")
        else:
            # Save and show details
            try:
                doc = save_new_cattle(cattle_id, cattle_name, images)
            except Exception as e:
                st.error(f"Error saving to DB: {e}")
            else:
                st.success(f"✅ Registered {cattle_name} with ID {cattle_id}")
                st.markdown("**Details:**")
                st.write(f"- ID: `{doc['12_digit_id']}`")
                st.write(f"- Name: {doc['cattle_name']}")
                st.write(f"- Created at (UTC): {doc['created_at']}")
                st.write("**Uploaded Images Preview:**")
                # Show all images uploaded
                for img in doc.get("images", []):
                    raw = base64.b64decode(img["b64"])
                    st.image(Image.open(io.BytesIO(raw)), caption=img["filename"], use_column_width=True)

                # Provide immediate download options for this just-registered cattle
                single_docs = [doc]
                csv_bytes = create_metadata_csv_bytes(single_docs)
                zip_bytes = create_images_zip_bytes(single_docs)

                st.download_button(
                    label="⬇️ Download metadata CSV for this cattle",
                    data=csv_bytes,
                    file_name=f"{cattle_id}_metadata.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="⬇️ Download images ZIP for this cattle",
                    data=zip_bytes,
                    file_name=f"{cattle_id}_images.zip",
                    mime="application/zip"
                )

# -----------------------
# UI: Browse / Filter / Download (Admin)
# -----------------------
st.header("📂 Browse & Download Registered Cattle")

col1, col2, col3 = st.columns([3,3,2])
with col1:
    q_id = st.text_input("Filter by ID (partial or full)")
with col2:
    q_name = st.text_input("Filter by Name (partial, case-insensitive)")
with col3:
    limit = st.number_input("Limit", min_value=1, max_value=500, value=50, step=1)

if st.button("Search"):
    docs = list_cattle(filter_id=q_id.strip() or None, filter_name=q_name.strip() or None, limit=int(limit))
    if not docs:
        st.info("No matching records found.")
    else:
        st.write(f"Found {len(docs)} records (showing up to {limit})")
        # Show a compact table of results with first-image thumbnail
        table_rows = []
        for d in docs:
            thumb = None
            if d.get("images"):
                thumb_b64 = d["images"][0]["b64"]
                thumb = f"data:image/png;base64,{thumb_b64}"  # for possible HTML embedding (not used)
            table_rows.append({
                "12_digit_id": d.get("12_digit_id"),
                "cattle_name": d.get("cattle_name"),
                "created_at": d.get("created_at"),
                "n_images": len(d.get("images", []))
            })
        st.dataframe(pd.DataFrame(table_rows))

        # Allow selecting subset for download
        st.write("Select IDs to download (or leave blank to download all shown):")
        ids = [d["12_digit_id"] for d in docs]
        selected = st.multiselect("Select specific IDs", options=ids)

        # Determine download set
        if selected:
            download_docs = [get_cattle_by_id(s) for s in selected]
        else:
            download_docs = docs

        if st.button("Download metadata CSV for selection"):
            csv_bytes = create_metadata_csv_bytes(download_docs)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_bytes,
                file_name=f"cattle_metadata_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        if st.button("Download images ZIP for selection"):
            zip_bytes = create_images_zip_bytes(download_docs)
            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_bytes,
                file_name=f"cattle_images_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )

# -----------------------
# Quick single-ID lookup panel
# -----------------------
st.header("🔎 Quick Lookup by ID")
lookup_id = st.text_input("Enter exact 12-digit ID to view", key="lookup")
if st.button("Lookup"):
    if not lookup_id:
        st.error("Enter an ID to lookup.")
    else:
        doc = get_cattle_by_id(lookup_id.strip())
        if not doc:
            st.warning("No cattle found with that ID.")
        else:
            st.write(f"**{doc['12_digit_id']} — {doc['cattle_name']}**")
            st.write(f"Created at: {doc['created_at']}")
            for img in doc.get("images", []):
                raw = base64.b64decode(img["b64"])
                st.image(Image.open(io.BytesIO(raw)), caption=img["filename"], use_column_width=True)
            # downloads for this single doc
            csv_bytes = create_metadata_csv_bytes([doc])
            zip_bytes = create_images_zip_bytes([doc])
            st.download_button("⬇️ Download CSV (this cattle)", data=csv_bytes, file_name=f"{lookup_id}_metadata.csv", mime="text/csv")
            st.download_button("⬇️ Download ZIP (this cattle)", data=zip_bytes, file_name=f"{lookup_id}_images.zip", mime="application/zip")
