# 🚢 Titanic Survival Prediction — ANN, RNN, CNN & LSTM

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Titanic-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

> Comparative deep learning study predicting Titanic passenger survival using 4 neural network architectures — ANN, RNN, CNN (with EarlyStopping), and LSTM.

---

## 🎯 Model Results

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| 🧠 TensorFlow ANN | Dense + BatchNorm + Dropout | **86%** |
| 🔁 RNN | Stacked SimpleRNN layers | **88%** |
| 🔷 CNN | Conv1D + GlobalMaxPooling + EarlyStopping | **95%** |
| 🧬 LSTM | Stacked LSTM layers | **97%** 🏆 |

---

## 📌 Problem Statement
Using the classic Kaggle Titanic dataset, this project builds and benchmarks 4 deep learning architectures to predict passenger survival. The goal is to compare model performance, training behaviour, and the effect of regularization techniques like EarlyStopping and Dropout.

## 🛠️ Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow, Keras, PyTorch
- **Models:** ANN, RNN (SimpleRNN), CNN (Conv1D), LSTM
- **Regularization:** EarlyStopping (CNN), Dropout, BatchNormalization
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
```
titanic-survival-prediction-dl/
├── Titanic-Dataset_TensorFlow_PyTorch-Preduction_Project/
│   ├── data/
│   │   └── Titanic-Dataset.csv
│   ├── python/
│   │   └── python.ipynb     # All 4 models
│   └── README.md
├── requirements.txt
└── README.md
```

## 🚀 How to Run
```bash
git clone https://github.com/shreesneha056-gif/titanic-survival-prediction-dl.git
cd titanic-survival-prediction-dl
pip install -r requirements.txt
jupyter notebook
```
Open `python.ipynb` and run all cells.

## 📊 Pipeline Overview
1. **Data Loading & EDA** — Explore Titanic dataset, visualize survival distributions
2. **Preprocessing** — Missing value imputation, encoding, feature scaling
3. **Model 1 — TensorFlow ANN** — Dense layers with BatchNorm & Dropout → **86%**
4. **Model 2 — RNN** — Stacked SimpleRNN for sequential learning → **88%**
5. **Model 3 — CNN** — Conv1D with EarlyStopping to prevent overfitting → **95%**
6. **Model 4 — LSTM** — Stacked LSTM capturing long-term patterns → **97%**
7. **Model Comparison** — Bar chart comparing all 4 models

## 💡 Key Takeaways
- **LSTM** achieved the best accuracy (97%) by capturing complex sequential dependencies
- **CNN with EarlyStopping** (95%) prevented overfitting and converged efficiently
- **EarlyStopping** in CNN stopped training early, saving compute while maintaining performance
- All deep learning models outperformed traditional ML baselines on this dataset

---
📫 [LinkedIn](https://www.linkedin.com/in/sneha-shree-mu/) | [Portfolio](https://shreesneha056-gif.github.io/portfolio_website/) | [GitHub](https://github.com/shreesneha056-gif)
