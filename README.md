# Stroke Prediction using Machine Learning

This project aims to predict the likelihood of a person having a stroke using multiple machine learning models. The prediction is based on health and lifestyle parameters, using a real-world dataset.

##  Problem Statement

Stroke is one of the leading causes of death worldwide and accounts for approximately 11% of all deaths. Early prediction can save lives by enabling timely medical interventions. This project uses pattern recognition and machine learning techniques to identify individuals at high risk of stroke.

---

- **Features**:
  - Age
  - Gender
  - Hypertension
  - Heart Disease
  - Marital Status
  - Work Type
  - Residence Type
  - Average Glucose Level
  - BMI
  - Smoking Status
  - Stroke (Target: 0 or 1)

---

##  Preprocessing

- Handling missing values
- Encoding categorical variables
- Feature scaling using `StandardScaler`
- Dimensionality Reduction using:
  - **PCA (Principal Component Analysis)**
  - **LDA (Linear Discriminant Analysis)**

---

##  Machine Learning Models Implemented

| Model | With Dim. Reduction | Accuracy |
|-------|---------------------|----------|
| Naive Bayes | None | 71.6% |
| Naive Bayes | LDA | 61.6% |
| Logistic Regression | None | 67.9% |
| Logistic Regression | PCA | 69.7% |
| Logistic Regression | LDA | 72.7% |
| K-Nearest Neighbors (k=7) | None | 75.2% |
| Decision Tree | Varies with depth | Up to 96% |
| Neural Network (MLP) | None | ~95.1% (but biased due to imbalance) |

>  Neural Network and Decision Tree show artificially high accuracy due to class imbalance.

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

All models were evaluated using an 80-20 train-test split.

---

## Insights

- **KNN (k=7)** gave the best balanced accuracy (75%) after accounting for data imbalance.
- **Decision Tree** showed high accuracy but was biased due to overfitting on imbalanced data.
- **Logistic Regression with LDA** performed reasonably well with ~72% accuracy.
- **Neural Network** predicted only the majority class due to imbalance, despite high accuracy.

---

---

##  Team Contributions

| Name                        | Contribution |
|-----------------------------|--------------|
| Yash Singholiya             | Logistic Regression, PCA, Project Page |
| Sunny Kumar                 | Logistic + LDA, Final Report |
| Gaurav Subhash Vishwe      | Naive Bayes + PCA, Frontend |
| Bhatt Nidhish Hiteshbhai    | Decision Tree, Neural Network, Slides |
| Banavath Chandra Mouli Naik | KNN, Meeting Minutes |

---
