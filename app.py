
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

model = tf.keras.models.load_model("skin_cancer_cnn.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("Skin Cancer Classification")
st.write("CNN Based Skin Lesion Image Classification")

uploaded_file = st.file_uploader(
    "Upload a Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(image_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"Predicted Class: {predicted_class}")
        st.write(f"Model Confidence: {confidence:.2f}%")

st.warning(
    "Educational project only. This prediction is not a medical diagnosis."
)
