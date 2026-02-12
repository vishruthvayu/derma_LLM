# derma_LLM
# Skin Lesion Metadata Analysis and Feature Selection  
**Hospital Italiano de Buenos Aires (2019–2022)**

---

## 1. Introduction

### Problem Statement  
Skin cancer, particularly melanoma, is a serious and potentially fatal condition if not detected early. While image-based diagnosis is widely studied, **clinical metadata** (such as age, anatomical site, and medical history) also plays a critical role in diagnosis. The problem addressed in this project is to determine **which metadata features are truly informative** for distinguishing **benign vs malignant skin lesions**.

### Why This Problem Is Important  
Many machine learning projects jump directly into model training without deeply understanding the data or validating feature relevance. In medical applications, this can lead to **unreliable models**, hidden biases, and poor generalization. Feature selection is especially important to:
- Improve interpretability
- Reduce overfitting
- Ensure clinical plausibility
- Build trust in ML-assisted decision systems

### Real-World Relevance  
Hospitals and diagnostic systems often have access to structured patient metadata even when high-quality images are unavailable. Understanding which metadata features matter can:
- Support clinicians in triage
- Improve risk assessment
- Complement image-based diagnostic systems

### Goal of the Project  
The primary goal is **not just to build a classifier**, but to:
- Clean and understand real-world medical metadata
- Perform statistically sound feature selection
- Validate feature importance using multiple methods
- Build an interpretable and defensible ML pipeline

---

## 2. Dataset Description

### Dataset Source  
The dataset is derived from the **Hospital Italiano de Buenos Aires Skin Lesions Dataset (2019–2022)**, a clinically curated dataset containing metadata associated with dermoscopic images.

### Why This Dataset Was Chosen  
This dataset was chosen because:
- It is **real-world medical data**, not synthetic or toy data
- It contains **rich clinical metadata**
- It reflects realistic challenges such as missing values, imbalance, and noise
- It is suitable for learning proper ML workflows, not just model fitting

### Data Size, Classes, and Features  
- **Total samples:** 1616 lesions  
- **Target classes:**  
  - Benign  
  - Malignant  
- **Key features:**  
  - Age (`age_approx`)  
  - Sex (`sex`)  
  - Anatomical site (`anatom_site_general`)  
  - Fitzpatrick skin type  
  - Family history of melanoma  
  - Personal history of melanoma  

### Observations About Data Quality  
- The dataset is **imbalanced**, with benign lesions significantly outnumbering malignant ones
- Several features contain missing values
- Some categorical variables have rare categories
- Boolean medical history features have unreported (NaN) values

These characteristics closely mirror real clinical data and require careful handling.

---

## 3. Data Preprocessing

### Steps Performed  
- Removed identifier and non-informative columns (e.g., `isic_id`, `patient_id`)
- Handled missing values:
  - Numerical features filled using **median**
  - Boolean medical history treated as `False` when not reported
  - Categorical features filled using **mode**
- Encoded categorical variables:
  - Binary features mapped to integers
  - Multi-class features one-hot encoded

### Why Each Step Is Necessary  
- Removing identifiers prevents **data leakage**
- Median imputation is robust to outliers in medical data
- Treating missing medical history as “not reported” preserves semantic meaning
- One-hot encoding avoids imposing false ordinal relationships

### Observed Improvements  
- No remaining missing values in the modeling dataset
- Reduced noise and redundancy
- Features became compatible with statistical tests and ML algorithms

---

## 4. Exploratory Data Analysis (EDA)

### Techniques Used  
- Class distribution analysis
- Age distribution plots
- Crosstabs between features and malignancy
- Summary statistics

### Why EDA Is Important  
EDA helps:
- Identify imbalance and bias
- Detect potential data leakage
- Form hypotheses about feature relevance
- Avoid blind reliance on models

### Key Insights  
- Malignant lesions are more common at higher ages
- Certain anatomical locations (e.g., head/neck) show higher malignant proportions
- Sex and skin type show moderate trends
- Some anatomical sites appear to have minimal signal

These insights guided later feature selection but were not used as final decisions.

---

## 5. Feature Engineering / Feature Selection

### Methods Used  
- One-hot encoding for categorical variables
- **Chi-Square test** for statistical dependency
- **Mutual Information** for non-linear relevance
- **Logistic Regression coefficients** for directional interpretability
- **Random Forest impurity-based importance**
- **Permutation Importance** to remove RF bias

### Why These Methods Were Chosen  
- Chi-square identifies statistical dependence
- Mutual Information captures non-linear relationships
- Logistic Regression provides interpretability
- Random Forest captures interactions
- Permutation Importance provides **unbiased importance estimates**

Using multiple methods avoids relying on a single, potentially misleading metric.

### Impact  
- Weak and noisy features were removed
- Strong, consistent features were retained
- Final feature set was statistically and empirically justified

---

## 6. Model Architecture / Algorithms Used

### Models Implemented  
- Logistic Regression (baseline interpretability model)
- Random Forest Classifier

### Why These Models Were Selected  
- Logistic Regression:
  - Simple
  - Interpretable
  - Good baseline for medical data
- Random Forest:
  - Handles non-linearity
  - Captures feature interactions
  - Robust to noise

### Strengths and Limitations  
- Logistic Regression:
  - Strength: interpretability
  - Limitation: linear assumptions
- Random Forest:
  - Strength: expressive power
  - Limitation: biased impurity-based importance (addressed via permutation)

---

## 7. Training Strategy

### Choices Made  
- Stratified train-test split to preserve class distribution
- Class-weight balancing for Random Forest
- Default hyperparameters with sufficient estimators (300 trees)

### Why These Choices  
- Stratification prevents misleading evaluation
- Class weighting addresses imbalance
- Avoiding heavy tuning keeps focus on feature analysis

### Observations During Training  
- Models converged stably
- No severe overfitting observed
- Performance indicated meaningful signal in metadata

---

## 8. Evaluation Metrics

### Metrics Used  
- ROC-AUC
- Precision
- Recall
- F1-score

### Why These Metrics Are Suitable  
- Accuracy alone is misleading due to imbalance
- ROC-AUC measures ranking quality
- Recall is critical for malignant cases
- Precision helps control false positives

### Interpretation  
ROC-AUC values above 0.7 indicate that metadata alone provides substantial predictive power.

---

## 9. Results and Analysis

### Final Observations  
- Age is the strongest predictor across all methods
- Personal medical history and anatomical site are strong secondary features
- Sex and skin type provide moderate refinement
- Some anatomical sites contribute negligible signal

### Comparison Across Methods  
Feature relevance was consistent across:
- Statistical tests
- Linear models
- Non-linear models
- Bias-corrected importance

This consistency strengthens confidence in the results.

---

## 10. Challenges Faced

### Key Challenges  
- Handling missing boolean values correctly
- Understanding why RF importance was biased
- Avoiding premature feature dropping
- Interpreting conflicting importance signals

### How They Were Addressed  
- Careful semantic treatment of missing data
- Use of permutation importance
- Multi-method validation
- Step-by-step pipeline design

### Lessons Learned  
- Feature importance is not a single number
- Order of preprocessing steps matters
- Interpretability and performance must be balanced

---

## 11. Conclusion

### Summary  
This project demonstrated a **complete, defensible ML workflow** for medical metadata analysis:
- Careful preprocessing
- Thorough EDA
- Multi-stage feature selection
- Model-based validation

### Final Outcome  
A compact, interpretable feature set with strong predictive signal and clinical plausibility.

### Why This Method Worked  
Because decisions were driven by:
- Statistics
- Domain logic
- Multiple validation layers  
—not by blind optimization.

---

## 12. Future Work

### Limitations  
- Metadata-only analysis
- No temporal or longitudinal data
- Limited sample size

### Possible Improvements  
- Combine image embeddings with metadata
- Explore gradient boosting models
- Perform external validation

### Advanced Directions  
- Multi-class diagnosis prediction
- Explainability with SHAP
- Deployment as a clinical decision-support tool

---

## 13. Tech Stack

- **Python** – core language
- **NumPy** – numerical operations
- **Pandas** – data manipulation
- **Matplotlib** – visualization
- **scikit-learn** – ML models and feature selection

Each tool was chosen for reliability, transparency, and industry relevance.

---

## 14. How to Run the Project

### Requirements  
- Python 3.9+
- pip or conda

### Setup  
```bash
pip install numpy pandas matplotlib scikit-learn
Execution

Place metadata.csv in the project directory

Run the notebook or script step by step:

Data preprocessing

EDA

Feature selection

Model training and evaluation

Notes

No GPU is required. The project is lightweight and fully reproducible.
