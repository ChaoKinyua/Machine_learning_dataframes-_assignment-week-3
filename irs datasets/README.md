# Iris Species Classification with Scikit-learn

A comprehensive machine learning project demonstrating classical ML techniques using the Iris dataset. This project implements a Decision Tree classifier to predict iris flower species based on physical measurements.

## Project Overview

This project showcases a complete machine learning pipeline including data loading, preprocessing, model training, evaluation, and visualization. It serves as an educational tool for understanding supervised learning with Scikit-learn.

**Dataset:** Iris Species Dataset (150 samples, 4 features, 3 classes)
**Algorithm:** Decision Tree Classifier
**Target Accuracy:** >95%

## Features

- Complete data preprocessing pipeline
- Decision Tree classifier implementation
- Comprehensive model evaluation metrics
- Confusion matrix analysis
- Feature importance analysis
- Professional visualizations
- Predictions on new data
- Well-commented code for learning

## Dataset Description

The Iris dataset contains measurements of iris flowers from three different species:

**Features (4):**
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

**Target Classes (3):**
- Setosa
- Versicolor
- Virginica

**Total Samples:** 150 (50 per species)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

### Run the Script

```bash
python iris_classifier.py
```

### Expected Output

The script will display:
- Dataset exploration and statistics
- Data preprocessing steps
- Model training progress
- Evaluation metrics (Accuracy, Precision, Recall)
- Classification report
- Confusion matrix
- Feature importance rankings
- Visualization saved as `iris_classifier_analysis.png`

### Using in Jupyter Notebook

1. Copy the code into a Jupyter cell
2. Run the cell
3. View outputs and visualizations inline

## Project Structure

```
iris-classifier/
├── iris_classifier.py          # Main script
├── iris_classifier_analysis.png # Generated visualization
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## Code Sections Explained

### Section 1: Libraries Import
Imports all necessary libraries for data handling, modeling, and visualization.

### Section 2: Data Loading
Loads the Iris dataset from Scikit-learn and converts it to a Pandas DataFrame for easier manipulation.

### Section 3: Data Preprocessing
- Checks for missing values
- Separates features and target variables
- Encodes categorical labels to numerical format

### Section 4: Train-Test Split
Divides data into 80% training and 20% testing sets with stratification to maintain class distribution.

### Section 5: Model Training
Creates and trains a Decision Tree Classifier with optimized hyperparameters.

### Section 6: Predictions
Makes predictions on both training and test datasets.

### Section 7: Model Evaluation
Calculates and displays accuracy, precision, and recall metrics.

### Section 8: Confusion Matrix
Generates and analyzes the confusion matrix to understand prediction patterns.

### Section 9: Visualizations
Creates four-panel visualization including:
- Confusion matrix heatmap
- Feature importance bar chart
- Performance metrics comparison
- Train vs test accuracy

### Section 10: New Data Predictions
Demonstrates how to make predictions on completely new iris measurements.

## Model Performance

Typical results when running the script:

- **Test Accuracy:** ~93-96%
- **Precision:** ~93-96%
- **Recall:** ~93-96%
- **Training Accuracy:** ~97-100%

Note: Exact values may vary due to random train-test split.

## Evaluation Metrics Explained

**Accuracy:** Percentage of correct predictions overall
- Formula: (True Positives + True Negatives) / Total Predictions
- Use: General performance indicator

**Precision:** Ratio of correct positive predictions to all positive predictions
- Formula: True Positives / (True Positives + False Positives)
- Use: Importance of false positives is high

**Recall:** Ratio of correct positive predictions to all actual positives
- Formula: True Positives / (True Positives + False Negatives)
- Use: Importance of false negatives is high

## Confusion Matrix Interpretation

A 3x3 matrix showing:
- Rows represent actual species
- Columns represent predicted species
- Diagonal values are correct predictions
- Off-diagonal values are misclassifications

## Feature Importance

The script identifies which features contribute most to classification decisions:
- Petal length typically has highest importance
- Petal width also contributes significantly
- Sepal measurements have lower importance

## Hyperparameters

Current Decision Tree configuration:
- `max_depth=5` - Prevents overfitting
- `random_state=42` - Ensures reproducibility
- `min_samples_split=2` - Minimum samples to split a node
- `min_samples_leaf=1` - Minimum samples at leaf node

Adjust these parameters to experiment with model complexity and performance.

## Output Files

**Console Output:**
- Step-by-step execution logs
- Performance metrics
- Detailed classification report
- Feature importance rankings

**iris_classifier_analysis.png:**
- Confusion matrix heatmap
- Feature importance chart
- Performance metrics bar chart
- Train vs test accuracy comparison

## Common Issues and Solutions

**Issue: ImportError: No module named 'sklearn'**
Solution: Install Scikit-learn
```bash
pip install scikit-learn
```

**Issue: ImportError: No module named 'seaborn'**
Solution: Install Seaborn
```bash
pip install seaborn
```

**Issue: ModuleNotFoundError: No module named 'pandas'**
Solution: Install Pandas
```bash
pip install pandas
```

**Issue: Visualization not displaying**
Solution: Ensure matplotlib backend is configured or save the PNG file.

## Modifications and Experiments

Try these modifications to deepen your understanding:

1. **Different Train-Test Split:**
   - Change `test_size=0.3` for 70-30 split
   - Change `test_size=0.1` for 90-10 split

2. **Hyperparameter Tuning:**
   - Adjust `max_depth` to test underfitting/overfitting
   - Change `min_samples_split` to control tree complexity

3. **Alternative Algorithms:**
   - Replace DecisionTreeClassifier with RandomForestClassifier
   - Try LogisticRegression or SVM

4. **Cross-Validation:**
   - Use `cross_val_score` for more robust evaluation
   - Implement k-fold cross-validation

## Learning Outcomes

After completing this project, you will understand:
- How to load and explore datasets
- Data preprocessing and label encoding
- Train-test split strategies
- Decision Tree classifier implementation
- Model evaluation metrics
- Confusion matrix interpretation
- Feature importance analysis
- Creating professional visualizations

## Theoretical Background

**Decision Tree Classifier:**
A supervised learning algorithm that builds a tree-like model of decisions. It works by repeatedly splitting the data based on feature values to create regions that minimize classification error.

**Advantages:**
- Easy to understand and interpret
- Works with non-linear data
- Requires minimal data preprocessing
- Handles both numerical and categorical data

**Disadvantages:**
- Prone to overfitting
- Can be unstable with small data changes
- Biased toward features with many levels

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- Iris Dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set
- Decision Trees: https://scikit-learn.org/stable/modules/tree.html

## Requirements File

Create `requirements.txt` with:
```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Tips for Best Results

1. Ensure all libraries are properly installed
2. Run the script from the project directory
3. Check that sufficient disk space exists for visualization output
4. Use Python 3.7 or higher for compatibility
5. Experiment with different random_state values to observe stability

## Contributing

Feel free to modify and experiment with this code for learning purposes.

## License

This project is created for educational purposes.

## Author

Created as part of a machine learning curriculum demonstrating classical ML techniques with Scikit-learn.

---

**Last Updated:** October 2025
**Status:** Complete and functional
