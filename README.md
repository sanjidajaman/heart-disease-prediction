# Heart Disease Prediction using Machine Learning

##  Project Overview
This project aims to predict the presence of heart disease in patients using the **UCI Heart Disease (Cleveland) dataset**.  
We implemented multiple machine learning algorithms, compared their performance, and identified the most suitable model for this classification task.

---

## Dataset
- **Source:** [UCI Machine Learning Repository – Heart Disease Dataset]
- **Subset Used:** Cleveland dataset
- **Size:** 303 patient records
- **Features:** 13 clinical attributes + target (heart disease presence)
- **Target Variable:**
  - `0` → No heart disease
  - `1–4` → Presence of heart disease (converted to binary: `1`)

---

## Algorithms Used

### Pre-Midterm
1. **Logistic Regression** – Interpretable baseline model
2. **Decision Tree (Classification)** – Rule-based classification
3. **k-Nearest Neighbors (k-NN)** – Similarity-based classification

### Post-Midterm
- Replaced **k-NN** with **Random Forest** for better performance and reduced overfitting.

---

## Data Preprocessing
- Handled missing values (`ca`, `thal`) with median/mode imputation
- One-hot encoding for categorical variables
- Feature scaling (StandardScaler) for k-NN
- Train/test split: **80/20**
- Cross-validation: **5-fold**

---

## Results

| Algorithm          | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 85%     | 83%       | 86%    | 84%      |
| Decision Tree       | 81%     | 80%       | 79%    | 79%      |
| Random Forest       | **89%** | **87%**   | **90%**| **88%**  |

**Best Model:** Random Forest

---

## Visualizations
- Accuracy comparison bar chart
- Feature importance plot (Random Forest)
- ROC curves for model evaluation

---

## Requirements
To run the project, install the required dependencies:
```bash

