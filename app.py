import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'model.pkl'
    model = load_learner(model_path)
    return model

def classify_image(model, img):
    return model.predict(img)[0]

# Set up the Streamlit app
def main():
    st.title("Cat or Dog Classifier")
    st.write("Upload an image, and the classifier will tell you if it's a cat or a dog!")

    # Load the trained model
    model = load_model()

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        img = PIL.Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Classify the image using the model
        label = classify_image(model, img)
        st.write(f"Prediction: {label.capitalize()}")

if __name__ == "__main__":
    main()
