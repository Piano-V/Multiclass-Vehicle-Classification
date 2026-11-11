# Multiclass Vehicle Type Classification Pipeline

An end-to-end, statistically grounded machine learning pipeline classifying vehicles into three categories (**Sedan**, **SUV**, and **Sports Car**) using continuous physical, dimensional, and performance attributes.

---

## Key Results & Architecture
* **Validation Strategy:** Stratified 5-fold cross-validation evaluated via out-of-fold (OOF) predictions.
* **Final Model:** Calibrated soft-voting probability ensemble blending multinomial **Logistic Regression (92%)** and **XGBoost (8%)**.
* **OOF Cross-Validation Accuracy:** `89.44%`
* **Holdout Test Set Accuracy:** `90.40%`
* **Generalization Gap:** 0.18 percentage points on the primary Logistic Regression baseline.

---

## Dataset & Preprocessing
* **Scale:** 10,000 raw samples with 20 numerical features and 1 target class (`vehicle_class`).
* **Class Balance:** Approximately equal distribution across classes (~33.89% Sedan, ~33.26% SUV, ~32.85% Sports Car).
* **Data Quality Audit:** Identified and eliminated **2,500 duplicate feature rows** to prevent data leakage across validation folds, using **7,500 unique records** for modeling.
* **Missing Values:** 0 missing entries present in the raw data.

---

## Model Benchmarking (Stratified 5-Fold CV)

| Model Architecture | Training Acc | 5-Fold CV Acc | Generalization Gap | Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression ($C=0.1$)** | **89.60%** | **89.41%** | **0.18%** | Best individual generalization; minimal variance |
| **XGBoost** (depth=3, lr=0.04) | 92.09% | 88.49% | 3.60% | Captures residual non-linear interactions |
| **QDA** | 89.42% | 88.67% | 0.75% | Non-linear quadratic decision boundary |
| **Random Forest** (depth=10) | 94.13% | 88.13% | 6.00% | Overfits to training bootstrap samples |
| **Decision Tree** (Unrestricted) | 94.15% | 84.00% | 10.15% | Severe overfitting as tree depth increases |
| **Gaussian Naive Bayes** | 84.70% | 84.72% | -0.02% | Limited by feature independence assumption |
| **Final Weighted Ensemble** | — | **89.44%** | — | **92% LR + 8% XGBoost probability blend** |

---

## Feature Engineering & Selection Analysis
Engineered 5 physical domain ratios (Power-to-weight, Torque-to-weight, Height-to-width, Ground-clearance ratio, and Estimated volume):
* **Original 20 Features:** 89.44% Ensemble CV accuracy
* **Engineered 25 Features:** 89.40% Ensemble CV accuracy
* **Correlation-Filtered (9 Features, $|r| > 0.85$):** 86.48% CV accuracy

*Outcome:* Following the principle of parsimony, the final ensemble retained the raw 20-feature space to avoid unnecessary complexity without performance sacrifice.

---

## Class-Wise Performance (OOF)
* **Sports Car:** 92.76% F1-score (high physical separability)
* **SUV:** 90.85% F1-score (distinct height, volume, and clearance signatures)
* **Sedan:** 84.80% F1-score (boundary overlap with crossovers and performance trims)
* **Macro F1-Score:** **89.47%**
