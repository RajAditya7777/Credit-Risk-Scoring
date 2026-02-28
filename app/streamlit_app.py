import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Credit Risk Scoring", layout="wide")
st.title("Intelligent Credit Risk Scoring System")

MODEL_PATH = "models/logistic_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Train Model", "Predict Single Borrower"])


# =====================================================
# TRAIN MODEL
# =====================================================
if page == "Train Model":

    uploaded_file = st.file_uploader("Upload Borrower Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        if "ID" in df.columns:
            df = df.drop("ID", axis=1)

        if "Status" not in df.columns:
            st.error("Column 'Status' not found.")
            st.stop()

        X = df.drop("Status", axis=1)
        y = df["Status"]

        st.subheader("Target Distribution")
        st.write(y.value_counts())
        st.write("Default %:", round((y.sum()/len(y))*100, 2), "%")

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])

        if st.button("Train Logistic Regression"):

            pipeline.fit(X_train, y_train)

            # Save inside models folder
            joblib.dump(pipeline, MODEL_PATH)
            joblib.dump(X.columns.tolist(), FEATURE_PATH)

            st.success("Model trained and saved in /models folder.")

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            st.write("ROC-AUC:", round(roc_auc, 4))

            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig_cm)

            st.subheader("ROC Curve")
            fig_roc, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax2.plot([0, 1], [0, 1], linestyle="--")
            ax2.legend()
            st.pyplot(fig_roc)


# =====================================================
# MANUAL PREDICTION
# =====================================================
elif page == "Predict Single Borrower":

    st.header("Single Borrower Risk Prediction")

    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please train model first.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    # NUMERIC INPUTS
    income = st.number_input("Income", min_value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    credit_score = st.number_input("Credit Score", min_value=0)
    ltv = st.number_input("Loan-to-Value (LTV)", min_value=0.0)
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.0)

    # IMPORTANT CATEGORICAL INPUTS
    credit_worthiness = st.selectbox("Credit Worthiness", ["l1", "l2"])
    open_credit = st.selectbox("Open Credit", ["opc", "nopc"])
    interest_only = st.selectbox("Interest Only", ["int_only", "not_int"])
    loan_limit = st.selectbox("Loan Limit Type", ["cf", "ncf"])
    business_flag = st.selectbox("Business or Commercial", ["ob/c", "nob/c"])
    occupancy = st.selectbox("Occupancy Type", ["pr", "sr", "ir"])

    if st.button("Predict Risk"):

        # Create base row with all features defaulted
        input_dict = {col: 0 for col in feature_columns}

        # Fill numeric
        input_dict["income"] = income
        input_dict["loan_amount"] = loan_amount
        input_dict["Credit_Score"] = credit_score
        input_dict["LTV"] = ltv
        input_dict["dtir1"] = dti

        # Fill categorical safely
        if "Credit_Worthiness" in input_dict:
            input_dict["Credit_Worthiness"] = credit_worthiness
        if "open_credit" in input_dict:
            input_dict["open_credit"] = open_credit
        if "interest_only" in input_dict:
            input_dict["interest_only"] = interest_only
        if "loan_limit" in input_dict:
            input_dict["loan_limit"] = loan_limit
        if "business_or_commercial" in input_dict:
            input_dict["business_or_commercial"] = business_flag
        if "occupancy_type" in input_dict:
            input_dict["occupancy_type"] = occupancy

        input_df = pd.DataFrame([input_dict])

        probability = model.predict_proba(input_df)[0][1]

        st.write("Default Probability:", round(probability, 3))

        if probability < 0.3:
            st.success("Low Risk")
        elif probability < 0.6:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")