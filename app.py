import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
from PIL import Image

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "model.pkl"
    learn = load_learner(model_path)
    return learn

def main():
    st.title("Cat or Dog Classifier")
    st.write("Upload an image, and the classifier will tell you if it's a cat or a dog.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        with st.spinner("Predicting..."):
            learn = load_model()
            pred, _, _ = learn.predict(PILImage.create(uploaded_file))

        # Display the result
        st.header(f"Prediction: {pred.capitalize()}")

if __name__ == "__main__":
    main()
