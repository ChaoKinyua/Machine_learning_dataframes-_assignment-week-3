import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os

# Page setup
st.set_page_config(page_title="MNIST Classifier", layout="wide")
st.title("🎨 MNIST Digit Classifier")
st.write("Draw a digit (0-9) and see AI prediction!")

# Check if model exists
if not os.path.exists('mnist_cnn_model.h5'):
    st.error("❌ Model file not found! Please train the model first.")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    st.success("✅ Model loaded successfully!")
    return model

model = load_model()

# Main app
col1, col2 = st.columns(2)

with col1:
    st.header("✏️ Draw Here")
    try:
        from streamlit_drawable_canvas import st_canvas
        canvas = st_canvas(
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    except ImportError:
        st.error("Please install: pip install streamlit-drawable-canvas")
        st.stop()

with col2:
    st.header("🤖 Prediction")
    if canvas.image_data is not None and model is not None:
        if np.any(canvas.image_data[:, :, :3] != 0):
            # Process image
            img = Image.fromarray(canvas.image_data.astype('uint8'), 'RGBA')
            img = img.convert('L')
            img = img.resize((28, 28))
            
            img_array = np.array(img) / 255.0
            img_array = 1 - img_array  # Invert colors
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # Predict
            pred = model.predict(img_array, verbose=0)
            digit = np.argmax(pred)
            confidence = np.max(pred)
            
            st.success(f"**Predicted: {digit}**")
            st.info(f"**Confidence: {confidence*100:.1f}%**")
            
            # Show probabilities
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(range(10), pred[0], color='lightblue')
            bars[digit].set_color('red')
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            ax.set_xticks(range(10))
            st.pyplot(fig)
        else:
            st.info("👆 Draw a digit first!")
    else:
        st.info("👆 Draw a digit first!")

st.markdown("---")
st.write("**Model Accuracy: 99.16%** | Built with TensorFlow & Streamlit")