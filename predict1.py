import tensorflow as tf
import numpy as np

# ✅ Save after training in proper Keras v3 format
model.save("pneumonia_cnn_model.keras", save_format="keras")

# ✅ Load your trained model (new format)
model = tf.keras.models.load_model("pneumonia_cnn_model.keras")

# Path to the new X-ray image
img_path = r"C:\pneumonia project\pneumonia_cnn\chestray.jpg"

# Load and preprocess
img = tf.keras.utils.load_img(img_path, target_size=(150, 150))   # resize
img_array = tf.keras.utils.img_to_array(img)                      # convert to array
img_array = np.expand_dims(img_array, axis=0) / 255.0             # add batch dimension + normalize

# Predict
prediction = model.predict(img_array)

# Interpret result
if prediction[0][0] > 0.5:
    print("Prediction: Pneumonia")
else:
    print("Prediction: Normal")
    