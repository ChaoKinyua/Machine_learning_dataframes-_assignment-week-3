
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
import io
from streamlit_drawable_canvas import st_canvas
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #2ca02c;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    try:
        # Try .keras first
        try:
            model = keras.models.load_model('mnist_cnn_model.keras')
            return model
        except:
            # Fall back to .h5
            model = keras.models.load_model('mnist_cnn_model.h5')
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first using: `python main_model.py`")
        return None

@st.cache_resource
def load_metrics():
    """Load model metrics"""
    try:
        with open('model_metrics.json', 'r') as f:
            return json.load(f)
    except:
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_image(img_array):
    """Preprocess image for model prediction"""
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # If grayscale, keep as is; if RGB, convert to grayscale
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Reshape to (28, 28, 1) if needed
    if len(img_array.shape) == 2:
        img_array = img_array.reshape(28, 28, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_digit(model, img_array):
    """Make prediction on input image"""
    predictions = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return predicted_digit, confidence, predictions[0]

def plot_confidence(predictions):
    """Plot confidence scores for all digits"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    digits = np.arange(10)
    colors = ['#2ca02c' if i == np.argmax(predictions) else '#d62728' for i in digits]
    
    bars = ax.bar(digits, predictions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, pred in zip(bars, predictions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pred:.2%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Digit', fontsize=12, fontweight='bold')
    ax.set_ylabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Model Confidence for Each Digit', fontsize=14, fontweight='bold')
    ax.set_xticks(digits)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🔢 MNIST Classifier")
    st.markdown("---")
    
    # Model info
    metrics = load_metrics()
    if metrics:
        st.subheader("📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
        with col2:
            st.metric("Test Loss", f"{metrics['loss']:.4f}")
    
    st.markdown("---")
    
    # Instructions
    st.subheader("📝 How to Use")
    st.write("""
    1. **Draw or Upload** a handwritten digit
    2. Click **Predict** button
    3. View the model's prediction and confidence
    4. Try different digits!
    """)
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About This App")
    st.write("""
    - **Model**: CNN (Convolutional Neural Network)
    - **Dataset**: MNIST (70,000 digit images)
    - **Accuracy**: >95% on test set
    - **Framework**: TensorFlow/Keras
    """)

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("✍️ Handwritten Digit Classification")
st.markdown("### Using Deep Learning CNN")

# Load model
model = load_model()

if model is None:
    st.error("❌ Model not found. Please train the model first.")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎨 Draw Digit", "📤 Upload Image", "📊 Model Info"])

# ============================================================================
# TAB 1: DRAW DIGIT
# ============================================================================

with tab1:
    st.subheader("Draw a Digit (0-9)")
    st.write("Use your mouse to draw a digit in the canvas below:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=3,
            stroke_color="black",
            background_color="white",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas"
        )
    
    with col2:
        st.write("")
        if st.button("🔮 Predict Drawing", key="draw_predict", use_container_width=True):
            if canvas_result.image_data is not None:
                # Get the drawn image
                drawn_image = canvas_result.image_data
                
                # Resize to 28x28
                img_pil = Image.fromarray((drawn_image).astype('uint8'), 'RGBA')
                img_pil = img_pil.convert('L')
                img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img_pil)
                
                # Preprocess
                img_preprocessed = preprocess_image(img_array)
                
                # Predict
                predicted_digit, confidence, predictions = predict_digit(model, img_preprocessed)
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.metric("Predicted Digit", predicted_digit, f"({confidence:.2%} confidence)")
                    
                    if confidence >= 0.9:
                        st.success(f"✅ High confidence prediction!")
                    elif confidence >= 0.7:
                        st.info(f"⚠️ Medium confidence prediction")
                    else:
                        st.warning(f"❓ Low confidence - unclear drawing")
                
                with result_col2:
                    st.write("**Processed Image (28×28):**")
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(img_preprocessed[0].reshape(28, 28), cmap='gray')
                    ax.set_axis_off()
                    st.pyplot(fig, use_container_width=True)
                
                # Confidence chart
                st.markdown("---")
                st.subheader("📊 Confidence Scores for All Digits")
                fig = plot_confidence(predictions)
                st.pyplot(fig, use_container_width=True)

# ============================================================================
# TAB 2: UPLOAD IMAGE
# ============================================================================

with tab2:
    st.subheader("Upload a Handwritten Digit Image")
    st.write("Upload an image of a handwritten digit (JPG, PNG, etc.)")
    
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'gif', 'bmp'])
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Image:**")
            img = Image.open(uploaded_file)
            st.image(img, use_container_width=True)
        
        # Process image
        img_array = np.array(img.convert('L'))
        
        # Resize to 28x28
        img_resized = Image.fromarray(img_array).resize((28, 28), Image.Resampling.LANCZOS)
        img_array_resized = np.array(img_resized)
        
        with col2:
            st.write("**Resized to 28×28:**")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_array_resized, cmap='gray')
            ax.set_axis_off()
            st.pyplot(fig, use_container_width=True)
        
        # Predict button
        if st.button("🔮 Predict Uploaded Image", use_container_width=True):
            # Preprocess
            img_preprocessed = preprocess_image(img_array_resized)
            
            # Predict
            predicted_digit, confidence, predictions = predict_digit(model, img_preprocessed)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.metric("Predicted Digit", predicted_digit, f"({confidence:.2%} confidence)")
                
                if confidence >= 0.9:
                    st.success(f"✅ High confidence prediction!")
                elif confidence >= 0.7:
                    st.info(f"⚠️ Medium confidence prediction")
                else:
                    st.warning(f"❓ Low confidence - unclear image")
            
            # Confidence chart
            st.markdown("---")
            st.subheader("📊 Confidence Scores for All Digits")
            fig = plot_confidence(predictions)
            st.pyplot(fig, use_container_width=True)

# ============================================================================
# TAB 3: MODEL INFORMATION
# ============================================================================

with tab3:
    st.subheader("📊 Model Architecture & Performance")
    
    # Model summary
    st.write("**Model Architecture:**")
    st.code("""
    Input Layer: 28×28×1
    │
    ├─ Conv Block 1: 32 filters (3×3)
    │  └─ MaxPool, Dropout
    │
    ├─ Conv Block 2: 64 filters (3×3)
    │  └─ MaxPool, Dropout
    │
    ├─ Conv Block 3: 128 filters (3×3)
    │  └─ MaxPool, Dropout
    │
    ├─ Dense Layer: 256 neurons
    │  └─ Dropout
    │
    ├─ Dense Layer: 128 neurons
    │  └─ Dropout
    │
    └─ Output Layer: 10 neurons (Softmax)
    """, language="")
    
    st.markdown("---")
    
    # Performance metrics
    metrics = load_metrics()
    if metrics:
        st.subheader("📈 Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%", 
                     delta="✅ >95%" if metrics['accuracy'] >= 0.95 else "⚠️ <95%")
        
        with col2:
            st.metric("Test Loss", f"{metrics['loss']:.4f}")
        
        with col3:
            accuracy_pct = (metrics['predictions_correct'] / metrics['total_predictions']) * 100
            st.metric("Correct Predictions", f"{metrics['predictions_correct']}/{metrics['total_predictions']}")
    
    st.markdown("---")
    
    # Dataset info
    st.subheader("📚 Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Training Set:**
        - 54,000 images
        - 28×28 pixels
        - Handwritten digits 0-9
        """)
    
    with col2:
        st.info("""
        **Test Set:**
        - 10,000 images
        - 28×28 pixels
        - Used for validation
        """)
    
    st.markdown("---")
    
    # Model details
    st.subheader("🔧 Model Details")
    
    details = {
        "Framework": "TensorFlow/Keras",
        "Model Type": "Convolutional Neural Network (CNN)",
        "Input Shape": "28×28×1 (grayscale images)",
        "Output": "10 classes (digits 0-9)",
        "Activation": "ReLU (hidden), Softmax (output)",
        "Optimizer": "Adam (lr=0.001)",
        "Loss Function": "Categorical Crossentropy",
        "Regularization": "Dropout, Batch Normalization",
        "Total Parameters": "~200,000+",
        "Training Epochs": "20 (early stopping)"
    }
    
    for key, value in details.items():
        st.write(f"**{key}:** {value}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔢 MNIST Handwritten Digit Classifier | Built with Streamlit & TensorFlow</p>
    <p style='font-size: 0.8rem; color: gray;'>Accuracy: >95% | Dataset: MNIST (70,000 images)</p>
</div>
""", unsafe_allow_html=True)