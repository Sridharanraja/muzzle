# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO

# # Load YOLOv11 classification model
# model = YOLO("./models/best_25.pt")

# st.title("Muzzle Identification")

# # File uploader
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Run prediction
#     results = model.predict(image)

#     # Extract top class with highest confidence
#     top_result = results[0].probs
#     class_id = int(top_result.top1)
#     confidence = float(top_result.top1conf)
#     class_name = model.names[class_id]

#     # Display result
#     st.success(f"Predicted Class: **{class_name}**") #(Confidence: {confidence:.2f})

# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO

# # Load YOLOv11 classification model
# # Replace 'yolo11-cls.pt' with your trained weights file
# model = YOLO("./models/best_25.pt")


# st.title("YOLOv11 Image Classification App")

# # Confidence threshold slider
# conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.9, 0.01)  # default set at 0.9 (90%)

# # File uploader
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Run prediction
#     results = model.predict(image)

#     # Extract top class with highest confidence
#     top_result = results[0].probs
#     class_id = int(top_result.top1)
#     confidence = float(top_result.top1conf)
#     class_name = model.names[class_id]

#     # Check confidence against threshold
#     if confidence >= conf_threshold:
#         st.success(f"Predicted Class: **{class_name}** (Confidence: {confidence:.2f})")
#     else:
#         st.error(
#             f"⚠️ Data not available in DB for reliable classification.\n\n"
#             f"But it looks **similar to: {class_name}** (Confidence: {confidence:.2f})"
#         )



import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLOv11 classification model
# Replace 'yolo11-cls.pt' with your trained weights file
model = YOLO("./models/best_25.pt")

st.title("YOLOv11 Image Classification App")

# UI Confidence threshold slider (user adjustable, separate from fixed 90% rule)
ui_threshold = st.slider("Confidence Threshold (for filtering results)", 0.0, 1.0, 0.5, 0.01)

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

    # --- Fixed logic for <90% ---
    if confidence < 0.90:
        st.error(
            f"⚠️ Data not available in DB for reliable classification.\n\n"
            # f"But it looks **similar to: {class_name}** (Confidence: {confidence:.2f})"
        )
    else:
        # --- Normal case, but also respect UI slider ---
        if confidence >= ui_threshold:
            st.success(f"Predicted Class: **{class_name}** (Confidence: {confidence:.2f})")
        else:
            st.warning("No prediction passed the selected confidence threshold.")

