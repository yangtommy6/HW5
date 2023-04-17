import streamlit as st
import fastai.vision.all  as fv

# Load your fast.ai model
model = fv.load_learner('model.pkl')

# Create a Streamlit interface
st.title('Image Classification')
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

# Make predictions
if uploaded_file is not None:
    image = fv.open_image(uploaded_file)
    prediction = model.predict(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Prediction:', prediction[0])
