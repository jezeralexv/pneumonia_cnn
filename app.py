import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd


# Load your trained model
model = tf.keras.models.load_model("pneumonia_cnn_model.h5")

st.title("Pneumonia Detection AI")

# Upload an X-ray
uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Preprocess image
    img = tf.keras.utils.load_img(uploaded_file, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.write("Prediction: **Pneumonia**")
    else:
        st.write("Prediction: **Normal**")


import tensorflow as tf

# Load the converted model
model = tf.keras.models.load_model("pneumonia_cnn_model.keras")

# Later in your app, use model.predict(...)

