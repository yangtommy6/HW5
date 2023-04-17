import os
import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_file = "model.pkl"
    model = load_learner(model_file)
    return model

# Predict the class of the input image
def predict(model, img):
    pred, idx, probs = model.predict(img)
    return pred, probs[idx].item()

def main():
    st.title("Your Image Classification App")
    st.write("Upload an image and the model will predict its class.")

    model = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Predicting..."):
            pred, prob = predict(model, img)
        
        st.success(f"Prediction: {pred}, Confidence: {prob:.2%}")

if __name__ == "__main__":
    main()
