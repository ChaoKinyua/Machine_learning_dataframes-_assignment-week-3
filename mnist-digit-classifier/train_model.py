import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("MNIST HANDWRITTEN DIGIT CLASSIFICATION WITH CNN")
print("=" * 70)

print("\n📊 Loading MNIST Dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"✓ Training: {X_train.shape}, Test: {X_test.shape}")

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train_encoded = keras.utils.to_categorical(y_train, 10)
y_test_encoded = keras.utils.to_categorical(y_test, 10)

print("\n🏗️ Building Model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n🚀 Training Model...")
history = model.fit(X_train, y_train_encoded,
                    batch_size=128,
                    epochs=20,
                    validation_split=0.1,
                    verbose=1)

print("\n📈 Evaluating...")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
test_accuracy = accuracy_score(y_test, y_pred)
test_loss, _ = model.evaluate(X_test, y_test_encoded, verbose=0)

print(f"\n✓ Test Accuracy: {test_accuracy*100:.2f}%")
if test_accuracy >= 0.95:
    print(f"✅ SUCCESS! >95% accuracy achieved!")

print("\n💾 Saving Model...")
model.save('mnist_cnn_model.h5')
print("✓ Model saved as 'mnist_cnn_model.h5'")

metrics = {
    'accuracy': float(test_accuracy),
    'loss': float(test_loss),
    'predictions_correct': int((y_test == y_pred).sum()),
    'total_predictions': len(y_test)
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f)

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print("\nRun Streamlit with: streamlit run streamlit_app.py")