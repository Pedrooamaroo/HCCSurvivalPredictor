# 🏥 HCCSurvivalPredictor: 1-Year Survival Prognosis with Clinical ML

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Sklearn-Machine_Learning-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imblearn-SMOTE_Balancing-green.svg)](https://imbalanced-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-red.svg)](https://pandas.pydata.org/)

## 🎯 Project Overview
This project tackles the critical medical challenge of predicting **1-year survival rates** for patients diagnosed with **Hepatocellular Carcinoma (HCC)**. Using clinical data from the *Hospital and University Centre of Coimbra (Portugal)*, I engineered a supervised learning pipeline to identify high-risk patients based on physiological, demographic, and laboratory factors.

The goal is to provide a robust algorithmic second opinion to support clinical decision-making in oncology.

---

## 📈 Executive Performance Summary
The project compared multiple classifiers, finding that **Ensemble Methods (Random Forest / Gradient Boosting)** consistently outperformed single estimators and Neural Networks on this tabular dataset.

| Architecture | Handling Imbalance | Accuracy | AUC-ROC | Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **SMOTE + GridSearch** | **~78%** | **0.82** | **🏆 Best** |
| Random Forest | SMOTE | ~75% | 0.79 | Strong |
| MLP (Neural Net) | Standard Scaling | ~68% | 0.71 | Baseline |

> *Note: Final accuracy metrics depend on the specific test split in the notebook execution.*

**Key Takeaway:** The application of **SMOTE (Synthetic Minority Over-sampling Technique)** was critical. By synthetically balancing the "Lives" vs. "Dies" classes, the model's ability to recall "High Risk" (Dies) patients improved significantly compared to the baseline.

---

## 🔬 Technical Deep Dive

### 1. Advanced Preprocessing & Imputation
Real-world clinical data is often messy. I implemented a robust cleaning strategy:
* **KNN Imputation:** Instead of dropping rows with missing data (which would lose 40%+ of the dataset), I used `fancyimpute.KNN(k=3)` to mathematically infer missing clinical values based on nearest patient neighbors.
* **Categorical Encoding:** Manual mapping of ordinal features (e.g., "Child-Pugh Score") to preserve hierarchy, and One-Hot Encoding for nominal features like Gender.

### 2. Handling Class Imbalance
The dataset suffers from a natural survival imbalance (more patients live than die). To prevent the model from ignoring the critical minority class:
* **SMOTE (Synthetic Minority Over-sampling):** Generated synthetic examples of the minority class in the training set to force the model to learn the decision boundary more effectively.

### 3. Model Architecture & Optimization
* **Ensemble Learning:** Leveraged `RandomForestClassifier` and `GradientBoostingClassifier` to capture non-linear relationships between features like Alpha-Fetoprotein (AFP) levels and survival rates.
* **Hyperparameter Tuning:** Utilized `GridSearchCV` to optimize decision tree depth, learning rates, and n-estimators, preventing overfitting on the small patient cohort (n=165).

---

## 📊 Visual Insights

| Feature Importance & Confusion Matrix | Analysis |
| :--- | :--- |
| <img src="https://placehold.co/400x300?text=Confusion+Matrix" width="450"> | **Clinical Insight:** The Confusion Matrix reveals the trade-off between Precision and Recall. For medical prognosis, maximizing **Recall** (catching all potential non-survivors) is prioritized over Precision to ensure high-risk patients receive aggressive care. |

---

## 🛠️ Installation & Usage

### 1. Clone & Setup
```bash
git clone [https://github.com/pedrooamaroo/HCCSurvivalPredictor.git](https://github.com/pedrooamaroo/HCCSurvivalPredictor.git)
cd HCCSurvivalPredictor
pip install -r requirements.txt
