# Credit Risk Scoring System

### Mid-Sem Machine Learning Project

---

## Introduction

This is my first complete end-to-end Machine Learning project.

In this project, I built a system that predicts whether a borrower is likely to default on a loan. The goal was to understand how ML models work in real-world financial problems and how to deploy them using a simple web interface.

This submission is only for **Milestone 1 (Mid-Sem)** and focuses completely on classical Machine Learning (no agent-based or LLM components).

---

## Problem Statement

Credit risk is a major concern for banks and financial institutions because loan defaults directly impact revenue and stability.

The objective of this project is:

* To predict whether a borrower will default or not.
* To calculate default probability.
* To understand which features influence credit risk.
* To display results using a Streamlit web app.

---

## What I Built

This project includes:

* Data preprocessing pipeline
* Logistic Regression model
* Proper train-test split
* Evaluation using multiple metrics
* Interactive Streamlit dashboard
* Manual single-borrower prediction feature

---

## Machine Learning Approach

I used **Logistic Regression** as the main model because:

* It is easy to understand.
* It provides probability output.
* It is commonly used in credit risk and financial prediction problems.
* It helps explain how features affect default likelihood.

---

## Data Preprocessing

Before training the model, I handled:

* Missing numerical values → Filled with median
* Missing categorical values → Filled with most frequent value
* Categorical features → OneHotEncoding
* Numerical features → Standard Scaling
* Class imbalance → Used `class_weight="balanced"`

This ensures the model does not favor only the majority class.

---

## System Flow

```
Upload Dataset (CSV)
        ↓
Data Cleaning
        ↓
Preprocessing Pipeline
        ↓
Train-Test Split
        ↓
Logistic Regression Model
        ↓
Evaluation Metrics
        ↓
Streamlit Web Interface
```

---

## Model Evaluation

I used the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* ROC Curve

### Why ROC-AUC?

Since credit risk datasets are usually imbalanced, accuracy alone is not enough. ROC-AUC shows how well the model separates defaulters from non-defaulters.

---

## Model Performance

* Accuracy: ~0.83
* ROC-AUC: ~0.86

These results show that the model performs well without overfitting.

---

## Streamlit Application Features

The web application allows:

* Uploading a borrower dataset
* Viewing dataset preview and shape
* Viewing target distribution
* Training the Logistic Regression model
* Saving the trained model to `/models`
* Viewing evaluation metrics (Accuracy, ROC-AUC)
* Viewing classification report
* Viewing confusion matrix
* Viewing ROC curve
* Predicting default risk for a single borrower manually
* Viewing default probability and risk category (Low / Medium / High)

---

## Risk Category Logic

Based on predicted probability:

* 0.0 – 0.3 → **Low Risk**
* 0.3 – 0.6 → **Medium Risk**
* 0.6 – 1.0 → **High Risk**

This makes the output easier to understand for non-technical users.

---

## Technologies Used

* Python
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## Feature Engineering

Before feeding the data into the model, I dropped columns that are not useful for prediction:

* **`year`** — just a time identifier, not a predictive feature
* **`ID`** — unique row identifier, carries no information
* **`rate_of_interest`** — removed to avoid data leakage, as interest rates are often determined after risk is assessed

This helps the model focus only on meaningful borrower characteristics.

---

## Project Structure

```
Credit-Risk-Scoring/
│
├── app/
│   └── streamlit_app.py          # Streamlit web application
│
├── data/
│   └── raw/
│       └── Loan_Default.csv      # Raw dataset
│
├── models/
│   ├── logistic_model.pkl        # Trained Logistic Regression pipeline
│   ├── feature_columns.pkl       # Saved feature column names
│   └── scaler.pkl                # Saved scaler
│
├── notebooks/
│   └── EDA.ipynb                 # Exploratory Data Analysis
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run Locally

1. Create virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## What I Learned

* How to build a complete ML pipeline.
* How to handle missing values properly.
* Why evaluation metrics matter beyond accuracy.
* How to deploy an ML model using Streamlit.
* How to save and reuse trained models with Joblib.

This project helped me understand how machine learning moves from theory to a working application.

---

## Conclusion

This project successfully demonstrates how classical Machine Learning can be used to solve a real-world financial problem like credit risk scoring.

It is my first full ML system that includes:

* Data preprocessing
* Model training
* Evaluation
* Deployment via web interface

This forms the foundation for future expansion into more advanced AI-based systems.
