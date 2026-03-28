import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Page configuration
st.set_page_config(page_title="Viego Detector", layout="centered")

def predict_viego(image_data, model, labels):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

# UI Header
st.title("League of Legends Champion Detector")
st.write("Upload a champion splash art to check if it is Viego.")

# Load Model and Labels
try:
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = f.readlines()
except Exception as e:
    st.error("Missing model files. Please ensure keras_model.h5 and labels.txt are in your GitHub repo.")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        label, confidence = predict_viego(image, model, labels)
        
        # Clean label text (removes numbers like '0 ' or '1 ')
        clean_label = label[2:].strip() if len(label) > 2 else label.strip()
        
        # Display Results
        st.divider()
        if "Viego" in clean_label and confidence > 0.70:
            st.success(f"Result: {clean_label}")
            st.write(f"Confidence: {confidence:.2%}")
            st.write("The Ruined King has been identified.")
        else:
            st.error("Result: Not Viego")
            st.write(f"This looks like {clean_label} (Confidence: {confidence:.2%})")

st.divider()
st.caption("Powered by Teachable Machine and Streamlit")
