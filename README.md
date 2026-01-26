**# Pneumonia Detection using CNN (MobileNetV2)**



**## 📌 Project Overview**

This project applies \*\*Convolutional Neural Networks (CNNs)\*\* with transfer learning (MobileNetV2) to detect pneumonia from chest X-ray images.  

It demonstrates end-to-end workflow: dataset preparation, model training, evaluation, saving, and prediction.



---



**## 📂 Repository Structure**



pneumonia\_cnn/ │── cnn\_pneumonia.py        # Training script (builds, trains, saves model) │── predict1.py             # Prediction script (loads model, runs inference) │── requirements.txt        # Dependencies │── .gitignore              # Ignore datasets and large model files │── README.md               # Project documentation │── archive/                # (ignored) chest\_xray dataset │── models/                 # (ignored) saved models like pneumonia\_cnn\_model.keras



---



**## ⚙️ Requirements**

Install dependencies inside your Python 3.10 virtual environment:

```bash

pip install -r requirements.txt



**REQUIREMENTS**



tensorflow==2.20.0

matplotlib==3.9.2

numpy

pandas



🚀 **Usage**

**1. Train the model**

Run the training script:

python cnn\_pneumonia.py



This will:

\- Load and preprocess the dataset.

\- Train MobileNetV2-based CNN.

\- Save the trained model as pneumonia\_cnn\_model.keras.





**2. Run prediction**

Use the prediction script:

python predict1.py





This will:

\- Load the saved model.

\- Preprocess a test chest X-ray image.

\- Output whether the image is Normal or Pneumonia.



**📊 Results**

\- Model trained with data augmentation and transfer learning.

\- Evaluated on test set with accuracy printed in console.

\- Training history plotted with Matplotlib.



🛑 **Notes**

\- The dataset (archive/chest\_xray) is large and excluded from GitHub via .gitignore.

\- The trained model file (pneumonia\_cnn\_model.keras) is also excluded to keep the repo lightweight.

\- Anyone cloning the repo can retrain the model using cnn\_pneumonia.py.



**✨ Future Improvements**

\- Add support for batch predictions on multiple test images.

\- Optionally build a Streamlit interface for deployment.

\- Experiment with other architectures (ResNet, EfficientNet).



**📜 License**

Apache 2.0 — same as TensorFlow.

















