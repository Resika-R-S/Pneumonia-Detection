import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained ResNet model
MODEL_PATH = 'vgg16.h5'
model = load_model(MODEL_PATH)

# Helper function to preprocess the image
def preprocess_image(img, target_size):
    # Convert the image to RGB if it's not already in that mode
    img = img.convert('RGB')
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert the image to a numpy array
    img_array = np.asarray(img)
    
    # Check if img_array is 3D (without channels)
    if img_array.ndim == 2:  # If the image is grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)  # Stack to create a 3-channel image
    
    # Expand dimensions to fit the model input
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit interface
st.title('Pneumonia Detection')
st.write("Upload a chest X-ray image to detect if it shows signs of pneumonia.")

# Upload image using Streamlit's file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the uploaded image
    st.write("Classifying...")
    img_preprocessed = preprocess_image(img, target_size=(256, 256))

    # Make a prediction
    preds = model.predict(img_preprocessed)
    result = "Pneumonia" if preds[0] > 0 else "Normal"

    # Display prediction result
    st.write(f"Prediction: {result}")
