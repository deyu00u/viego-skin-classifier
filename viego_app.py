import streamlit as st
import tensorflow as tf
import tf_keras as keras
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Viego Skin Classifier", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .title-text {
        text-align: center; color: #ffffff;
        font-family: 'Inter', sans-serif; font-weight: 800;
        font-size: 3rem; margin-bottom: 0px;
    }
    .subtitle-text {
        text-align: center; color: #8b949e; margin-bottom: 2rem;
    }
    .upload-card {
        background-color: #161b22; padding: 30px;
        border-radius: 15px; border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .result-container-blue {
        background-color: #112135; border-radius: 10px;
        padding: 15px; border-left: 5px solid #58a6ff;
        margin-top: 20px;
    }
    .result-container-red {
        background-color: #2a1215; border-radius: 10px;
        padding: 15px; border-left: 5px solid #ff4b4b;
        margin-top: 20px;
    }
    .result-text-blue { color: #58a6ff; font-weight: bold; font-size: 1.2rem; margin: 0; }
    .result-text-red { color: #ff4b4b; font-weight: bold; font-size: 1.2rem; margin: 0; }
    .confidence-text { color: #ffffff; font-size: 0.9rem; opacity: 0.8; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_viego_ai():
    model = keras.models.load_model("keras_model.h5", compile=False)
    labels = [
        "0 Base Viego",
        "1 King Viego",
        "2 Revenant Reign Viego",
        "3 Viego EDG",
        "4 Viego Soul Fighter",
        "5 Viego Lunar Beast",
        "6 Viego Pentakill",
        "7 Viego Worlds"
    ]
    return model, labels

st.markdown('<p class="title-text">Viego Skin Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Identify the Ruined King or detect an impostor.</p>', unsafe_allow_html=True)

st.divider()

left_spacer, center_col, right_spacer = st.columns([1, 2, 1])

with center_col:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("### Submit Your Artwork")
    uploaded_file = st.file_uploader("Drop Image Here", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.caption("Analyzed Image Output")

        model, labels = load_viego_ai()
        
        with st.spinner("Analyzing Spectral Resonance..."):
            size = (224, 224)
            processed_image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(processed_image)
            normalized_img = (img_array.astype(np.float32) / 127.5) - 1
            data = np.expand_dims(normalized_img, axis=0)

            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = labels[index]
            confidence = prediction[0][index]

        if confidence < 0.40:
            st.markdown(f"""
                <div class="result-container-red">
                    <p class="result-text-red">Result: Not Viego</p>
                    <p class="confidence-text">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            display_name = class_name.split(" ", 1)[1] if " " in class_name else class_name
            st.markdown(f"""
                <div class="result-container-blue">
                    <p class="result-text-blue">Result: {display_name}</p>
                    <p class="confidence-text">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)