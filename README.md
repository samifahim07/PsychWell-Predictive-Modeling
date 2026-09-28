# Mental Health Classification

A comprehensive machine learning study for binary classification of mental health issues using structured survey data. Nine algorithms — ranging from classical statistical models to gradient boosting ensembles and deep neural networks — are trained, tuned, and benchmarked against one another to identify the most effective approach for this domain.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

Mental health conditions affect a significant portion of the global population, yet remain underdiagnosed due to social stigma, limited access to professionals, and absence of scalable screening tools. This project investigates whether a machine learning classifier, trained on demographic, behavioral, and self-reported symptom data, can reliably predict the presence of a mental health issue.

The pipeline covers end-to-end model development: exploratory analysis, class imbalance handling via SMOTE, hyperparameter optimization using `RandomizedSearchCV`, and a structured comparison of nine classifiers across accuracy, F1, precision, and recall.

---

## Dataset

**File:** `mental_health.csv`

The dataset contains individual records with demographic attributes, lifestyle indicators, and self-reported psychological symptoms. The binary target variable is `Has_Mental_Health_Issue`.

**Feature categories include:**

- **Demographic:** Age, Gender, Education
- **Lifestyle:** Sleep_Hours_Night, Screen_Time_Hours_Day, Social_Media_Hours_Day
- **Stressors:** Work_Stress_Level, Financial_Stress, Loneliness
- **Symptoms (binary flags):** Feeling_Sad_Down, Loss_Of_Interest, Sleep_Trouble, Fatigue, Poor_Appetite_Or_Overeating, Feeling_Worthless, Concentration_Difficulty, Anxious_Nervous, Panic_Attacks, Mood_Swings, Irritability, Obsessive_Thoughts, Compulsive_Behavior, Self_Harm_Thoughts, Suicidal_Thoughts

**Target:** `Has_Mental_Health_Issue` (binary: 0 = No, 1 = Yes)

---

## Project Structure

```
mental-health-classification/
├── mental_health.ipynb       # Main notebook: EDA, preprocessing, modeling, evaluation
├── mental_health.csv         # Source dataset
├── best_model_rf.pkl         # Serialized best-performing model (Random Forest)
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis

Distributions of the target variable across demographic groups (gender, education, age) and lifestyle factors (sleep, screen time, social media) are examined via count plots, histograms, and box plots. Symptom prevalence is ranked, and a correlation heatmap reveals the relationship between individual symptoms and the target.

### 2. Preprocessing

- **Label Encoding:** All categorical columns are numerically encoded using `LabelEncoder`.
- **Feature/Target Split:** Features (`X`) are separated from the binary target (`y`).
- **Train-Test Split:** 25% of the data is used for training, 75% for testing (`train_size=0.25`, `random_state=42`).

### 3. Class Imbalance Handling

**SMOTE (Synthetic Minority Oversampling Technique)** is applied exclusively to the training set to balance the class distribution without leaking synthetic samples into the evaluation set.

### 4. Hyperparameter Optimization

With the exception of Logistic Regression (baseline), all models undergo `RandomizedSearchCV` with 3-fold cross-validation over broad parameter grids (`n_iter` between 20 and 40 depending on model complexity). This ensures a fair, systematic search rather than manual tuning.

### 5. Evaluation

Each model is assessed using:
- Overall accuracy
- Macro F1 score
- Per-class precision, recall, and F1
- Confusion matrix heatmap

---

## Models Evaluated

| # | Model | Tuning |
|---|-------|--------|
| 1 | Logistic Regression | Default (baseline) |
| 2 | Decision Tree | RandomizedSearchCV |
| 3 | Random Forest | RandomizedSearchCV |
| 4 | XGBoost | RandomizedSearchCV |
| 5 | CatBoost | RandomizedSearchCV |
| 6 | AdaBoost | RandomizedSearchCV |
| 7 | LightGBM | RandomizedSearchCV |
| 8 | ANN (Dense layers) | RandomizedSearchCV via scikeras |
| 9 | CNN (1D Conv) | RandomizedSearchCV via scikeras |

---

## Results

| Model | Accuracy (%) | Macro F1 | Class 1 Recall |
|-------|-------------|----------|----------------|
| Random Forest | **91.43** | 0.50 | 0.99 |
| XGBoost | 90.71 | 0.52 | 0.98 |
| LightGBM | 90.68 | 0.51 | 0.98 |
| AdaBoost | 90.05 | 0.52 | 0.97 |
| CatBoost | 89.77 | 0.51 | 0.97 |
| ANN | 86.11 | 0.52 | 0.93 |
| CNN | 83.81 | 0.52 | 0.90 |
| Logistic Regression | 80.28 | 0.53 | 0.85 |
| Decision Tree | 78.45 | 0.50 | 0.84 |

**Random Forest** achieves the highest overall accuracy (91.43%) and is saved as the production model. However, the macro F1 scores across all models remain low (0.50-0.53), reflecting a persistent challenge in identifying Class 0 (non-issue) cases — a consequence of the underlying class imbalance even after SMOTE, and a meaningful consideration for real-world deployment.

> The best model is serialized to `best_model_rf.pkl` using Python's `pickle` module for downstream inference.

---

## Installation

**Python 3.8 or higher is required.**

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/mental-health-classification.git
cd mental-health-classification
pip install -r requirements.txt
```

---

## Usage

Open and run the notebook sequentially:

```bash
jupyter notebook mental_health.ipynb
```

To load and use the saved model directly:

```python
import pickle

with open("best_model_rf.pkl", "rb") as f:
    model = pickle.load(f)

# Pass a preprocessed feature array
predictions = model.predict(X_new)
```

Ensure that any new data undergoes the same label encoding and feature ordering used during training before passing it to the model.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
catboost
tensorflow
scikeras
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost tensorflow scikeras
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.
