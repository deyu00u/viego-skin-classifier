import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Is This Viego?", layout="centered")

def predict_champion(image_data, model, labels):
    size = (224, 224)
    image = image_data.convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

st.title("Is This Viego?")
st.write("Upload any picture to identify Viego.")

@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

try:
    model, labels = load_my_model()
except Exception as e:
    st.error("Missing model files. Please ensure keras_model.h5 and labels.txt are in your GitHub repo.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Scanning the Black Mist..."):
        label, confidence = predict_champion(image, model, labels)
        
        clean_label = label.split(' ', 1)[1] if ' ' in label else label
        
        st.divider()
        
        # New Fixed Logic: Only succeed if "Viego" is in the name
        if "viego" in clean_label.lower() and "not viego" not in clean_label.lower() and confidence > 0.65:
            st.success(f"Skin Identified: {clean_label}")
            st.write(f"Confidence Level: {confidence:.2%}")
            st.write("Target confirmed: The Ruined King.")
        else:
            st.error("Identification: Not Viego")
            st.write(f"Detected: {clean_label}")
            st.write(f"Confidence Level: {confidence:.2%}")
            st.write("This soul does not belong to the Ruined King.")

st.divider()
st.caption("System Status: Operational")
