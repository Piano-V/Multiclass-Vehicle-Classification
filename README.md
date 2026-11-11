# Multiclass Vehicle Type Classification Pipeline

An end-to-end, statistically grounded machine learning pipeline classifying vehicles into three categories (**Sedan**, **SUV**, and **Sports Car**) using continuous physical, dimensional, and performance attributes[cite: 2, 4].

---

## Key Results & Architecture
* **Validation Strategy:** Stratified 5-fold cross-validation evaluated via out-of-fold (OOF) predictions[cite: 7].
* **Final Model:** Calibrated soft-voting probability ensemble blending multinomial **Logistic Regression (92%)** and **XGBoost (8%)**[cite: 1, 10, 11].
* **OOF Cross-Validation Accuracy:** `89.44%`[cite: 1, 10, 12]
* **Holdout Test Set Accuracy:** `90.40%`[cite: 1, 12]
* **Generalization Gap:** 0.18 percentage points on the primary Logistic Regression baseline[cite: 7, 8].

---

## Dataset & Preprocessing
* **Scale:** 10,000 raw samples with 20 numerical features and 1 target class (`vehicle_class`)[cite: 2, 3, 4].
* **Class Balance:** Approximately equal distribution across classes (~33.89% Sedan, ~33.26% SUV, ~32.85% Sports Car)[cite: 3, 5].
* **Data Quality Audit:** Identified and eliminated **2,500 duplicate feature rows** to prevent data leakage across validation folds, using **7,500 unique records** for modeling[cite: 3, 4, 5].
* **Missing Values:** 0 missing entries present in the raw data[cite: 4, 5].

---

## Model Benchmarking (Stratified 5-Fold CV)

| Model Architecture | Training Acc | 5-Fold CV Acc | Generalization Gap | Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression ($C=0.1$)** | **89.60%**[cite: 8, 9] | **89.41%**[cite: 8, 9] | **0.18%**[cite: 7, 8] | Best individual generalization; minimal variance[cite: 8, 9] |
| **XGBoost** (depth=3, lr=0.04)[cite: 9] | 92.09%[cite: 9] | 88.49%[cite: 9, 10] | 3.60%[cite: 9] | Captures residual non-linear interactions[cite: 11] |
| **QDA** | 89.42%[cite: 7] | 88.67%[cite: 7] | 0.75%[cite: 7] | Non-linear quadratic decision boundary[cite: 7] |
| **Random Forest** (depth=10)[cite: 9] | 94.13%[cite: 9] | 88.13%[cite: 9] | 6.00%[cite: 9] | Overfits to training bootstrap samples[cite: 9] |
| **Decision Tree** (Unrestricted)[cite: 8] | 94.15%[cite: 8] | 84.00%[cite: 8] | 10.15%[cite: 8] | Severe overfitting as tree depth increases[cite: 8] |
| **Gaussian Naive Bayes** | 84.70%[cite: 7] | 84.72%[cite: 7] | -0.02%[cite: 7] | Limited by feature independence assumption[cite: 7] |
| **Final Weighted Ensemble** | — | **89.44%**[cite: 10, 12] | — | **92% LR + 8% XGBoost probability blend**[cite: 1, 10, 11] |

---

## Feature Engineering & Selection Analysis
Engineered 5 physical domain ratios (Power-to-weight, Torque-to-weight, Height-to-width, Ground-clearance ratio, and Estimated volume)[cite: 6]:
* **Original 20 Features:** 89.44% Ensemble CV accuracy[cite: 6, 9]
* **Engineered 25 Features:** 89.40% Ensemble CV accuracy[cite: 6, 9]
* **Correlation-Filtered (9 Features, $|r| > 0.85$):** 86.48% CV accuracy[cite: 6, 9]

*Outcome:* Following the principle of parsimony, the final ensemble retained the raw 20-feature space to avoid unnecessary complexity without performance sacrifice[cite: 6, 9, 11].

---

## Class-Wise Performance (OOF)
* **Sports Car:** 92.76% F1-score (high physical separability)[cite: 10]
* **SUV:** 90.85% F1-score (distinct height, volume, and clearance signatures)[cite: 4, 10]
* **Sedan:** 84.80% F1-score (boundary overlap with crossovers and performance trims)[cite: 10]
* **Macro F1-Score:** **89.47%**[cite: 10]
