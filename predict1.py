import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("pneumonia_cnn_model.keras")
print("Model loaded successfully!")

# Path to the new X-ray image
img_path = r"C:\pneumonia project\pneumonia_cnn\chestray.jpg"

# Load and preprocess
img = tf.keras.utils.load_img(img_path, target_size=(150, 150))   # resize
img_array = tf.keras.utils.img_to_array(img)                      # convert to array
img_array = np.expand_dims(img_array, axis=0) / 255.0             # add batch dimension + normalize

# Predict
prediction = model.predict(img_array)
prob = prediction[0][0]

# Interpret result + severity
if prob > 0.5:
    if prob > 0.9:
        severity = "Severe"
    elif prob > 0.7:
        severity = "Moderate"
    else:
        severity = "Mild"
    print(f"Prediction: Pneumonia — {severity} [{prob:.2f}]")
else:
    print(f"Prediction: Normal [{1 - prob:.2f}]")

