# 🚢 Titanic Survival Prediction: A Comparative Deep Learning Approach

## 📌 Overview
This repository contains an end-to-end machine learning and deep learning pipeline to predict passenger survival on the Titanic. The core focus of this project is to implement identical multi-layer neural network architectures across the two most prominent deep learning frameworks—**TensorFlow/Keras** and **PyTorch**—to compare their workflow implementations, early stopping mechanisms, and final evaluation metrics.

## 📊 Dataset & Preprocessing
The project utilizes the classic `Titanic-Dataset.csv`. Before training the neural networks, the raw data underwent a rigorous preprocessing pipeline:

* **Handling Missing Values:** Cleaned data dynamically by imputing the `median` for missing continuous values (like `Age`) and the `mode` for missing categorical values (like `Embarked`).
* **Categorical Encoding:** Converted categorical features (`Sex` and `Embarked`) into numerical format using One-Hot Encoding (`pd.get_dummies`), ensuring high-cardinality flags like `drop_first=True` were maintained to avoid multi-collinearity.
* **Feature Dropping:** Excluded identifiers and text columns that do not contribute to predictive patterns, such as `PassengerId`, `Name`, `Ticket`, `Cabin`, and the unencoded categorical columns.
* **Feature Scaling:** Applied standard scaling (`StandardScaler`) to normalize continuous variations and guarantee optimal gradient descent convergence across both neural network frameworks.
* **Data Splitting:** Partitioned data into an 80% training set and a 20% test set using a locked random state for reproducibility.

## 🧠 Model Architecture
To guarantee a completely fair framework comparison, both models share the exact same multi-layer perceptron (MLP) architecture:

1. **Input Layer:** Fits the standardized feature dimension (8 input nodes).
2. **Hidden Layer 1:** 64 dense neurons with a `ReLU` activation function.
3. **Regularization Layer 1:** 40% Dropout (`0.4`) to prevent overfitting.
4. **Hidden Layer 2:** 32 dense neurons with a `ReLU` activation function.
5. **Regularization Layer 2:** 30% Dropout (`0.3`) to further secure generalization.
6. **Output Layer:** 1 node with a `Sigmoid` activation function, transforming the logits into a binary class probability.

---

## 🚀 Framework Implementations

### 1. TensorFlow / Keras
* Modeled cleanly using Keras' `Sequential` API.
* Optimized using the **Adam** optimizer against a **Binary Cross-Entropy** loss function.
* Utilized an `EarlyStopping` callback monitoring `val_loss` with a patience threshold of 10 epochs, successfully rolling back to the absolute best weights upon completion.

### 2. PyTorch
* Structured explicitly via a custom class inheriting from `nn.Module`.
* Wrapped features into custom PyTorch `TensorDataset` and `DataLoader` batches.
* Structured a manual, transparent training loop tracking epoch losses, featuring a custom early-stopping check that automatically caches the optimal model weights file (`best.pth`).

---

## 📈 Evaluation Results
Both models display strong predictive classification thresholds on unseen test data. The detailed evaluation break downs from the classification reports indicate:

### TensorFlow / Keras Model
* **Train Set Accuracy:** 83%
* **Test Set Accuracy:** 80%
* **Test Metrics:** * Class 0 (Did not survive): Precision: 0.81 | Recall: 0.88 | F1: 0.84
  * Class 1 (Survived): Precision: 0.80 | Recall: 0.70 | F1: 0.75

### PyTorch Model
* **Train Set Accuracy:** 85%
* **Test Set Accuracy:** 82%
* **Test Metrics:**
  * Class 0 (Did not survive): Precision: 0.81 | Recall: 0.90 | F1: 0.86
  * Class 1 (Survived): Precision: 0.84 | Recall: 0.70 | F1: 0.76

IMPORTANT :- USE COLAB FOR BEST RUNTIME.

*Note: Minor accuracy discrepancies between the two setups reflect differences in baseline initialization states and framework-specific internal dropout tracking.*

---