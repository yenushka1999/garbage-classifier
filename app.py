import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Garbage Classifier", layout="centered")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("garbage_classifier.h5")
        return model
    except:
        st.error("Model file not found. Please upload garbage_classifier.h5")
        return None

def predict_image(image, model):
    IMG_SIZE = (128, 128)
    class_names = ['metal', 'organic', 'plastic']
    
    # Preprocess
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

# Main app
st.title("Garbage Classifier")
st.write("Upload an image to classify it as Metal, Organic, or Plastic")

model = load_model()

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)
        
        with st.spinner("Classifying..."):
            predicted_class, confidence = predict_image(image, model)
        
        # Color-coded results
        colors = {"plastic": "#007bff", "organic": "#ffc107", "metal": "#6c757d"}
        color = colors.get(predicted_class, "#333")
        
        st.markdown(f'<h3 style="color: {color}">Prediction: {predicted_class.upper()}</h3>', unsafe_allow_html=True)
        st.write(f"Confidence: {confidence:.2%}")
else:
    st.error("Cannot load model. Please check if garbage_classifier.h5 is in the repository.")
