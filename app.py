import streamlit as st
from fastai.vision.all import *

# Load the model
model_path = 'model.pkl'
learner = load_learner(model_path)

# Define the predict function
def predict(image):
    # Open the image file
    img = PIL.Image.open(image)

    # Make the prediction
    pred, _, _ = learner.predict(img)
    return pred

# Define the Streamlit app
def app():
    st.title('Photo Classification App')

    # Add a file uploader widget
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make a prediction and display the result
        pred = predict(uploaded_file)
        st.write(f'Prediction: {pred}')

# Run the app
if __name__ == '__main__':
    app()
