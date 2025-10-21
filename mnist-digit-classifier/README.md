# MNIST Handwritten Digit Classifier

A deep learning application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model achieves over 99% accuracy and is deployed as an interactive web app using Streamlit.

## Features

- **High Accuracy**: 99%+ test accuracy on MNIST dataset
- **Interactive Web Interface**: Draw digits directly in the browser and get instant predictions
- **Real-time Visualization**: See prediction confidence for all 10 digits
- **Pre-trained Model**: Ready-to-use trained model included
- **Easy Deployment**: Simple Streamlit app for quick setup

## Project Structure

```
mnist-digit-classifier/
├── train_model.py              # Model training script
├── streamlit_app.py            # Web app interface
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── mnist_cnn_model.h5          # Trained model (auto-generated)
├── model_metrics.json          # Model performance metrics (auto-generated)
└── venv/                       # Virtual environment (ignored in git)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mnist-digit-classifier.git
cd mnist-digit-classifier
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

If you want to retrain the model from scratch:

```bash
python train_model.py
```

This will:
- Download the MNIST dataset (70,000 training images)
- Build and train the CNN model
- Achieve ~99% accuracy
- Save the trained model as `mnist_cnn_model.h5`
- Generate performance metrics

Training takes approximately 5-10 minutes depending on your system.

### Run the Web App

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## How to Use the App

1. **Draw a Digit**: Use your mouse to draw a handwritten digit (0-9) in the canvas
2. **Get Prediction**: The model will instantly predict the digit
3. **View Confidence**: See the probability distribution across all digits
4. **Try Again**: Clear and draw another digit

## Model Architecture

The CNN consists of:

- **Input Layer**: 28×28 grayscale images
- **Convolutional Block 1**: 32 filters → MaxPooling → Dropout
- **Convolutional Block 2**: 64 filters → MaxPooling → Dropout
- **Convolutional Block 3**: 128 filters → MaxPooling → Dropout
- **Dense Layers**: 256 neurons → 128 neurons → Dropout
- **Output Layer**: 10 neurons (softmax for classification)

**Total Parameters**: ~200,000+

## Performance

- **Test Accuracy**: 99.16%
- **Test Loss**: 0.027
- **Training Time**: ~8 minutes
- **Inference Time**: <100ms per prediction
- **Model Size**: ~5.7 MB

## Dataset

- **MNIST (Modified National Institute of Standards and Technology)**
- **70,000 total images**: 60,000 training + 10,000 test
- **Image size**: 28×28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)

## Technology Stack

- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Image Processing**: Pillow

## Installation of Dependencies

The `requirements.txt` includes:
```
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
streamlit>=1.0.0
streamlit-drawable-canvas>=0.2.0
pillow>=9.0.0
```

## Troubleshooting

### Model file not found
Make sure you've trained the model first:
```bash
python train_model.py
```

### Drawing canvas not working
Install the drawable canvas component:
```bash
pip install streamlit-drawable-canvas
```

### Import errors
Reinstall all dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Slow predictions
- Ensure you're using a modern browser (Chrome, Firefox, Edge)
- Close other CPU-intensive applications
- If available, use a GPU-enabled TensorFlow installation

## Future Improvements

- Add real-time webcam support
- Implement model ensemble for higher accuracy
- Add batch prediction capability
- Deploy to cloud (Streamlit Cloud, AWS, GCP)
- Create mobile app version
- Add explainability features (saliency maps)

## Files Description

### train_model.py
- Loads MNIST dataset
- Builds CNN architecture
- Trains model with early stopping
- Evaluates on test set
- Saves trained model and metrics

### streamlit_app.py
- Interactive web interface
- Canvas for drawing digits
- Real-time predictions
- Confidence visualization
- Model information display

### requirements.txt
- Lists all required Python packages
- Specifies minimum versions for compatibility

## Model Training Details

**Hyperparameters:**
- Optimizer: Adam (learning rate: 0.001)
- Loss Function: Categorical Crossentropy
- Batch Size: 128
- Epochs: 20 (with early stopping)
- Validation Split: 10%

**Regularization:**
- Dropout (0.25-0.5)
- Batch Normalization
- Early stopping (patience: 5 epochs)

## Deployment Options

### Local Deployment
```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect your repository
4. App deploys automatically

### Docker
```bash
docker build -t mnist-classifier .
docker run -p 8501:8501 mnist-classifier
```

## Performance Metrics

Per-digit accuracy on test set:
- Digit 0: 99%+
- Digit 1: 99%+
- Digit 2: 98%+
- Digit 3: 98%+
- Digit 4: 99%+
- Digit 5: 98%+
- Digit 6: 99%+
- Digit 7: 98%+
- Digit 8: 98%+
- Digit 9: 98%+

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your work, please cite:

```bibtex
@project{mnist_classifier_2025,
  title={MNIST Handwritten Digit Classifier},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/mnist-digit-classifier}
}
```

## Acknowledgments

- MNIST dataset provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- TensorFlow/Keras for deep learning framework
- Streamlit for web application framework

## FAQ

**Q: How accurate is the model?**
A: The model achieves 99.16% accuracy on the MNIST test set.

**Q: Can I use my own handwritten digits?**
A: Yes! Draw or upload any handwritten digit image, and the model will predict it.

**Q: How long does training take?**
A: Approximately 5-10 minutes on a standard CPU, faster on GPU.

**Q: Can I retrain the model with different data?**
A: Yes, modify `train_model.py` to use your own dataset.

**Q: Is GPU required?**
A: No, but it significantly speeds up training and predictions.

**Q: Can I deploy this online?**
A: Yes, using Streamlit Cloud, Heroku, AWS, or any cloud platform that supports Python.

---

**Last Updated**: October 2025
**Status**: Active and Maintained
