import streamlit as st
from fastai.vision.all import *
import PIL.Image
website_url = "https://colab.research.google.com/drive/1REmEqpWZw3nTDEYCmuUNsUwMglNdbwnG"

# Function to load the model
def load_model(model_path):
    model = load_learner(model_path)
    return model

# Function to predict the image
def predict_image(img, model):
    img_fastai = PILImage.create(img)
    pred, pred_idx, probs = model.predict(img_fastai)
    return f'Prediction: {pred}, Probability: {probs[pred_idx]:.04f}'

# Main function
def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Cat or Dog Classifier")
    st.write("Upload an image, and this app will tell you if it's a cat or a dog. ")
    st.write("To see how to train the model using fast.ai library, please go to:")
    st.markdown(f"{website_url}")

    model_file = 'model.pkl'
    model = load_model(model_file)

    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        img = PIL.Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        with st.spinner('Predicting...'):
            prediction = predict_image(img, model)

        st.write(prediction)

if __name__ == '__main__':
    main()
