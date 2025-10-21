import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("IRIS SPECIES CLASSIFICATION WITH SCIKIT-LEARN")
print("=" * 70)

print("\n📊 SECTION 1: Loading and Exploring Data")
print("-" * 70)

# Load the Iris dataset from Scikit-learn
iris = load_iris()

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

print(f"\n✓ Dataset shape: {df.shape}")
print(f"✓ Features: {list(iris.feature_names)}")
print(f"✓ Target classes: {iris.target_names}")
print(f"✓ Samples per class:\n{df['species'].value_counts()}")

# Display first few rows
print("\n✓ First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Get basic statistics
print(f"\nBasic statistics:\n{df.describe()}")


print("\n" + "=" * 70)
print("📋 SECTION 2: Data Preprocessing")
print("=" * 70)

# Check for missing values
missing_values = df.isnull().sum().sum()
print(f"\n✓ Total missing values: {missing_values}")

if missing_values > 0:
    print("✓ Handling missing values using mean imputation")
    # Fill missing values with the mean of each column
    df_numeric = df.select_dtypes(include=[np.number])
    for col in df_numeric.columns:
        df[col].fillna(df[col].mean(), inplace=True)
    print("✓ Missing values handled")
else:
    print("✓ No missing values found - data is clean")

# Separate features (X) and target (y)
X = df[iris.feature_names]  # Features: sepal length, sepal width, petal length, petal width
y = df['species']  # Target: iris species

print(f"\n✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")

# Encode target labels (convert species names to numerical values)
# In this case, Scikit-learn will handle encoding automatically, but let's be explicit
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n✓ Label encoding mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label}: {i}")

print(f"✓ Encoded target (first 10 values): {y_encoded[:10]}")


print("\n" + "=" * 70)
print("🔀 SECTION 3: Train-Test Split")
print("=" * 70)

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,  # 20% for testing
    random_state=42,  # For reproducibility
    stratify=y_encoded  # Maintain class distribution
)

print(f"\n✓ Training set size: {X_train.shape[0]} samples")
print(f"✓ Testing set size: {X_test.shape[0]} samples")
print(f"✓ Training set class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} samples")



print("\n" + "=" * 70)
print("🌳 SECTION 4: Training Decision Tree Classifier")
print("=" * 70)

# Create Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(
    max_depth=5,  # Limit tree depth to prevent overfitting
    random_state=42,
    min_samples_split=2,  # Minimum samples to split a node
    min_samples_leaf=1  # Minimum samples required at leaf node
)

print("\n✓ Decision Tree parameters:")
print(f"  Max depth: {dt_classifier.max_depth}")
print(f"  Min samples split: {dt_classifier.min_samples_split}")
print(f"  Min samples leaf: {dt_classifier.min_samples_leaf}")

# Train the model on training data
print("\n✓ Training the model...")
dt_classifier.fit(X_train, y_train)
print("✓ Model training completed!")

# Get feature importance
feature_importance = dt_classifier.feature_importances_
print("\n✓ Feature importance:")
for feature, importance in zip(iris.feature_names, feature_importance):
    print(f"  {feature}: {importance:.4f}")


print("\n" + "=" * 70)
print("🔮 SECTION 5: Making Predictions")
print("=" * 70)

# Predict on training data
y_train_pred = dt_classifier.predict(X_train)

# Predict on testing data
y_test_pred = dt_classifier.predict(X_test)

print("\n✓ Predictions completed on both training and test sets")

# Decode predictions back to original labels for interpretation
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

print("\n✓ Sample predictions on test set (first 10):")
for i in range(10):
    actual = y_test_labels[i]
    predicted = y_test_pred_labels[i]
    match = "✓" if actual == predicted else "✗"
    print(f"  {match} Sample {i+1}: Actual={actual}, Predicted={predicted}")



print("\n" + "=" * 70)
print("📈 SECTION 6: Model Evaluation")
print("=" * 70)

# Calculate evaluation metrics for training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\n✓ TRAINING SET METRICS:")
print(f"  Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# Calculate evaluation metrics for testing data
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

print(f"\n✓ TEST SET METRICS:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"  Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")

# Per-class metrics
print(f"\n✓ PER-CLASS METRICS:")
print(classification_report(y_test, y_test_pred, 
                          target_names=label_encoder.classes_,
                          digits=4))

# Check for overfitting
if train_accuracy - test_accuracy > 0.1:
    print(f"\n⚠️ Warning: Possible overfitting detected!")
    print(f"   Training accuracy ({train_accuracy:.4f}) is significantly higher than test accuracy ({test_accuracy:.4f})")
else:
    print(f"\n✓ Model shows good generalization (no obvious overfitting)")



print("\n" + "=" * 70)
print("🔢 SECTION 7: Confusion Matrix Analysis")
print("=" * 70)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\n✓ Confusion Matrix:")
print(cm)

print("\nInterpretation:")
for i, class_name in enumerate(label_encoder.classes_):
    correct = cm[i, i]
    total = cm[i].sum()
    accuracy_per_class = correct / total if total > 0 else 0
    print(f"  {class_name}: {correct}/{total} correct ({accuracy_per_class*100:.2f}%)")



print("\n" + "=" * 70)
print("📊 SECTION 8: Creating Visualizations")
print("=" * 70)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Decision Tree Classifier - Iris Dataset Analysis', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix Heatmap
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax1)
ax1.set_title('Confusion Matrix')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

# Plot 2: Feature Importance
ax2 = axes[0, 1]
importance_sorted = sorted(zip(iris.feature_names, feature_importance), key=lambda x: x[1], reverse=True)
features_sorted = [x[0] for x in importance_sorted]
importances_sorted = [x[1] for x in importance_sorted]
ax2.barh(features_sorted, importances_sorted, color='skyblue')
ax2.set_title('Feature Importance')
ax2.set_xlabel('Importance')

# Plot 3: Model Performance Comparison
ax3 = axes[1, 0]
metrics = ['Accuracy', 'Precision', 'Recall']
scores = [test_accuracy, test_precision, test_recall]
colors = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax3.bar(metrics, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylim([0, 1])
ax3.set_ylabel('Score')
ax3.set_title('Model Performance Metrics')
# Add value labels on bars
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Train vs Test Accuracy
ax4 = axes[1, 1]
datasets = ['Training', 'Testing']
accuracies = [train_accuracy, test_accuracy]
colors_acc = ['#3498db', '#e74c3c']
bars = ax4.bar(datasets, accuracies, color=colors_acc, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_ylim([0, 1])
ax4.set_ylabel('Accuracy')
ax4.set_title('Train vs Test Accuracy')
# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('iris_classifier_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'iris_classifier_analysis.png'")
plt.show()



print("\n" + "=" * 70)
print("🆕 SECTION 9: Predictions on New Data")
print("=" * 70)

# Create a new sample for prediction
new_sample = pd.DataFrame({
    'sepal length (cm)': [5.1],
    'sepal width (cm)': [3.5],
    'petal length (cm)': [1.4],
    'petal width (cm)': [0.2]
})

print(f"\n✓ New sample for prediction:")
print(new_sample)

# Make prediction
prediction_encoded = dt_classifier.predict(new_sample)[0]
prediction_probability = dt_classifier.predict_proba(new_sample)[0]
prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

print(f"\n✓ Prediction result:")
print(f"  Predicted species: {prediction_label}")
print(f"  Prediction probabilities:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"    {class_name}: {prediction_probability[i]:.4f} ({prediction_probability[i]*100:.2f}%)")



print("\n" + "=" * 70)
print("✅ CLASSIFICATION TASK COMPLETED")
print("=" * 70)

print("\n📋 SUMMARY:")
print(f"  ✓ Dataset: Iris Species (150 samples, 4 features, 3 classes)")
print(f"  ✓ Model: Decision Tree Classifier")
print(f"  ✓ Train-Test Split: 80-20")
print(f"  ✓ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  ✓ Test Precision: {test_precision*100:.2f}%")
print(f"  ✓ Test Recall: {test_recall*100:.2f}%")
print(f"  ✓ Feature Importance Top: {features_sorted[0]}")
print(f"  ✓ Status: Model successfully trained and evaluated")
print("\n" + "=" * 70)