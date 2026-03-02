"""
Credit Risk Scoring — Professional Analytics Dashboard
=======================================================
Author  : Senior Data Analyst / Dashboard Engineer
Sections:
  1. Executive Summary   – KPI metric cards
  2. Data Overview       – Dataset stats, charts, heatmap
  3. Feature Insights    – Logistic-coefficient importance
  4. Model Performance   – Confusion matrix, ROC, report
  5. Customer Risk Pred. – Input form + Plotly gauge
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  — injected CSS for premium look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global font & background ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp, .main { background: #0d0f1a !important; }

    /* ── KPI cards ── */
    .kpi-card {
        background: #13162a;
        border: 1px solid #1e2240;
        border-radius: 14px;
        padding: 22px 24px 18px;
        text-align: left;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5);
        transition: transform .2s ease, box-shadow .2s ease;
        height: 130px;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 32px rgba(0,0,0,0.6);
    }
    .kpi-title {
        font-size: 11px; color: #6b7090;
        letter-spacing: 1.2px; text-transform: uppercase;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 38px; font-weight: 700;
        margin: 0 0 6px; line-height: 1;
    }
    .kpi-sub { font-size: 12px; color: #555a78; }

    /* Per-card accent colors (match reference) */
    .kpi-cyan   { color: #00d4ff; }
    .kpi-red    { color: #e63946; }
    .kpi-gold   { color: #f4a92a; }
    .kpi-pink   { color: #ff6b9d; }
    .kpi-purple { color: #a78bfa; }

    /* ── Section headers with red dot ── */
    .section-header {
        font-size: 18px; font-weight: 600;
        color: #d0d4f0;
        display: flex; align-items: center;
        gap: 8px;
        margin: 24px 0 14px;
    }
    .section-header::before {
        content: '';
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: #e63946;
        flex-shrink: 0;
    }

    /* ── Risk badge ── */
    .risk-low    { background:rgba(0,255,150,0.15); color:#00ff96; border:1px solid #00c97a;
                   padding:8px 24px; border-radius:24px; font-weight:700; font-size:18px; display:inline-block; }
    .risk-medium { background:rgba(244,169,42,0.15); color:#f4a92a; border:1px solid #c98900;
                   padding:8px 24px; border-radius:24px; font-weight:700; font-size:18px; display:inline-block; }
    .risk-high   { background:rgba(230,57,70,0.18); color:#e63946; border:1px solid #b02030;
                   padding:8px 24px; border-radius:24px; font-weight:700; font-size:18px; display:inline-block; }

    /* ── Divider ── */
    .fancy-divider { border:none; border-top:1px solid #1a1e35; margin:24px 0; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #0a0c18 !important; border-right:1px solid #1a1e35; }
    [data-testid="stSidebar"] * { color: #c0c4e0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH     = "models/logistic_model.pkl"
FEATURE_PATH   = "models/feature_columns.pkl"
LOCAL_CSV_PATH = "data/raw/Loan_Default.csv"   # auto-detected at startup

# Plotly dark template shared across all charts
PLOTLY_TEMPLATE = "plotly_dark"
# Primary accent = red; secondary = teal/mint (matches reference image)
ACCENT          = "#e63946"   # bold red  (risk / churn)
ACCENT2         = "#00d4ff"   # cyan      (safe / model)
ACCENT3         = "#f4a92a"   # gold      (neutral metrics)
ACCENT_PURPLE   = "#a78bfa"   # purple    (AUC / advanced)
ACCENT_MINT     = "#00ffb3"   # mint      (retained / low-risk)
# Chart sequence: red → gold → cyan → purple → pink → ...
COLOR_SEQ = ["#e63946", "#f4a92a", "#00d4ff", "#a78bfa", "#ff6b9d", "#00ffb3", "#ff9f43", "#48dbfb"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — load model gracefully
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
        model    = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURE_PATH)
        return model, features
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — auto-initialise from local CSV + saved model (no upload needed)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def auto_initialize(csv_path: str):
    """
    Reads the local CSV and evaluates the already-saved model on a held-out
    test split.  Returns a metrics dict identical to train_and_cache_model()
    so all dashboard tabs render without the user needing to re-upload.
    """
    df = pd.read_csv(csv_path)
    if "ID" in df.columns:
        df = df.drop("ID", axis=1)
    if "Status" not in df.columns:
        return None

    model, _ = load_model()
    if model is None:
        return None

    X = df.drop("Status", axis=1)
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc      = auc(fpr, tpr)
    acc          = accuracy_score(y_test, y_pred)
    cm           = confusion_matrix(y_test, y_pred)
    cr           = classification_report(y_test, y_pred, output_dict=True)

    return {
        "pipeline": model,
        "X": X, "y": y,
        "X_test": X_test, "y_test": y_test,
        "y_pred": y_pred, "y_prob": y_prob,
        "fpr": fpr, "tpr": tpr,
        "roc_auc": roc_auc, "acc": acc,
        "cm": cm, "cr": cr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — train and cache model from uploaded CSV
# ─────────────────────────────────────────────────────────────────────────────
def train_and_cache_model(df: pd.DataFrame):
    """Train logistic regression pipeline, save to disk, return metrics dict."""
    if "ID" in df.columns:
        df = df.drop("ID", axis=1)
    if "Status" not in df.columns:
        st.error("Column 'Status' not found in the dataset.")
        st.stop()

    X = df.drop("Status", axis=1)
    y = df["Status"]

    numeric_features     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline,     numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with st.spinner("🔄 Training Logistic Regression model…"):
        pipeline.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURE_PATH)
    st.cache_resource.clear()

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc      = auc(fpr, tpr)
    acc          = accuracy_score(y_test, y_pred)
    cm           = confusion_matrix(y_test, y_pred)
    cr           = classification_report(y_test, y_pred, output_dict=True)

    return {
        "pipeline": pipeline,
        "X": X, "y": y,
        "X_test": X_test, "y_test": y_test,
        "y_pred": y_pred, "y_prob": y_prob,
        "fpr": fpr, "tpr": tpr,
        "roc_auc": roc_auc, "acc": acc,
        "cm": cm, "cr": cr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Executive Summary KPI cards
# ─────────────────────────────────────────────────────────────────────────────
def render_executive_summary(total_customers, churn_rate, accuracy, roc_auc):
    st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    # Each card gets its own vivid accent color (matching reference image)
    def kpi(col, title, value, accent_class, sub_text):
        col.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">{title}</div>
              <div class="kpi-value {accent_class}">{value}</div>
              <div class="kpi-sub">{sub_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    kpi(c1, "Total Customers",   f"{total_customers:,}",
        "kpi-cyan",   "full dataset")

    kpi(c2, "Default Rate",      f"{churn_rate:.1f}%",
        "kpi-red",    f"{'▲ high risk' if churn_rate > 30 else '▼ manageable'}")

    kpi(c3, "Model Accuracy",    f"{accuracy:.1%}",
        "kpi-gold",   f"{'▲ good fit' if accuracy >= 0.75 else '▼ needs review'}")

    kpi(c4, "ROC-AUC Score",     f"{roc_auc:.3f}",
        "kpi-purple", f"{'▲ strong discriminator' if roc_auc >= 0.80 else '▼ moderate'}")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Data Overview tab
# ─────────────────────────────────────────────────────────────────────────────
def render_data_overview(df: pd.DataFrame, y: pd.Series):
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    # ── Dataset preview (collapsible) ──────────────────────────────────────
    with st.expander("📋 Dataset Preview (first 200 rows)", expanded=False):
        st.dataframe(df.head(200), width='stretch', height=280)

    col_shape1, col_shape2, col_shape3 = st.columns(3)
    col_shape1.metric("Rows",    f"{df.shape[0]:,}")
    col_shape2.metric("Columns", df.shape[1])
    col_shape3.metric("Memory",  f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Missing values ──────────────────────────────────────────────────────
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)

    col_miss, col_churn = st.columns([1.4, 1])

    with col_miss:
        st.markdown("**Missing Values by Feature**")
        if miss.empty:
            st.success("✅ No missing values found in the dataset.")
        else:
            fig_miss = px.bar(
                x=miss.values, y=miss.index,
                orientation="h",
                labels={"x": "Missing Count", "y": "Feature"},
                color=miss.values,
                color_continuous_scale="Reds",
                template=PLOTLY_TEMPLATE,
                title="Missing Value Distribution",
            )
            fig_miss.update_layout(
                showlegend=False, coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=40, b=10), height=350,
            )
            st.plotly_chart(fig_miss, width='stretch')

    with col_churn:
        st.markdown("**Default / Churn Distribution**")
        churn_counts = y.value_counts().reset_index()
        churn_counts.columns = ["Status", "Count"]
        churn_counts["Status"] = churn_counts["Status"].map({0: "No Default", 1: "Default"})
        fig_pie = px.pie(
            churn_counts, values="Count", names="Status",
            color_discrete_sequence=[ACCENT_MINT, ACCENT],
            hole=0.45, template=PLOTLY_TEMPLATE,
            title="Default Distribution",
        )
        fig_pie.update_traces(textinfo="percent+label", pull=[0.04, 0.04])
        fig_pie.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=350)
        st.plotly_chart(fig_pie, width='stretch')

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Numeric feature histograms ──────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    st.markdown("**Numeric Feature Distributions**")
    if numeric_cols:
        cols_per_row = 3
        rows_needed  = (min(len(numeric_cols), 9) + cols_per_row - 1) // cols_per_row
        fig_hist = make_subplots(
            rows=rows_needed, cols=cols_per_row,
            subplot_titles=numeric_cols[:9],
        )
        for i, col in enumerate(numeric_cols[:9]):
            r, c = divmod(i, cols_per_row)
            fig_hist.add_trace(
                go.Histogram(x=df[col], name=col,
                             marker_color=COLOR_SEQ[i % len(COLOR_SEQ)],
                             showlegend=False, nbinsx=30),
                row=r + 1, col=c + 1,
            )
        fig_hist.update_layout(
            template=PLOTLY_TEMPLATE, height=340 * rows_needed,
            title_text="Feature Histograms",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_hist, width='stretch')

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Correlation heatmap ─────────────────────────────────────────────────
    st.markdown("**Correlation Heatmap (Numeric Features)**")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            template=PLOTLY_TEMPLATE,
            title="Pearson Correlation Matrix",
        )
        fig_corr.update_layout(
            margin=dict(l=10, r=10, t=50, b=10), height=480,
        )
        st.plotly_chart(fig_corr, width='stretch')
        st.caption("🔍 Values near ±1 indicate strong correlation; near 0 indicates independence.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Feature Insights tab
# ─────────────────────────────────────────────────────────────────────────────
def render_feature_insights(pipeline, X: pd.DataFrame, y: pd.Series):
    st.markdown('<div class="section-header">Feature Insights</div>', unsafe_allow_html=True)

    # Extract feature names from ColumnTransformer
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier   = pipeline.named_steps["classifier"]

        num_features = preprocessor.transformers_[0][2]
        try:
            cat_encoder  = preprocessor.transformers_[1][1].named_steps["encoder"]
            cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]).tolist()
        except Exception:
            cat_features = []

        all_features = list(num_features) + cat_features
        coefs        = classifier.coef_[0]

        if len(coefs) == len(all_features):
            feat_df = pd.DataFrame({"Feature": all_features, "Coefficient": coefs})
            feat_df["Abs_Coef"] = feat_df["Coefficient"].abs()
            feat_df = feat_df.sort_values("Abs_Coef", ascending=False).head(20)

            # ── Top-20 feature importance bar chart ────────────────────────
            feat_df["Direction"] = feat_df["Coefficient"].apply(
                lambda v: "↑ Increases Risk" if v > 0 else "↓ Decreases Risk"
            )
            fig_feat = px.bar(
                feat_df.sort_values("Coefficient"),
                x="Coefficient", y="Feature",
                color="Direction",
                color_discrete_map={
                    "↑ Increases Risk": ACCENT,
                    "↓ Decreases Risk": ACCENT_MINT,
                },
                orientation="h",
                template=PLOTLY_TEMPLATE,
                title="Top Feature Coefficients (Logistic Regression)",
                labels={"Coefficient": "Log-Odds Coefficient"},
            )
            fig_feat.update_layout(
                height=max(400, len(feat_df) * 24 + 80),
                margin=dict(l=10, r=10, t=50, b=10),
                legend_title="Impact Direction",
            )
            st.plotly_chart(fig_feat, width='stretch')
            st.caption(
                "🔴 Red bars → feature pushes model toward **Default**. "
                "🟢 Green bars → feature pushes toward **No Default**. "
                "Magnitude = strength of influence."
            )
        else:
            st.info("Feature names could not be aligned with coefficients (OHE mismatch).")

    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Boxplots of numeric features grouped by churn ──────────────────────
    st.markdown("**Key Numeric Features — Distribution by Default Status**")
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if numeric_cols:
        plot_df = X[numeric_cols[:6]].copy()
        plot_df["Status"] = y.values
        plot_df["Status"] = plot_df["Status"].map({0: "No Default", 1: "Default"})

        cols_box = st.columns(2)
        for i, col in enumerate(numeric_cols[:6]):
            fig_box = px.box(
                plot_df, x="Status", y=col,
                color="Status",
                color_discrete_map={"No Default": ACCENT_MINT, "Default": ACCENT},
                template=PLOTLY_TEMPLATE,
                title=f"{col} by Default Status",
            )
            fig_box.update_layout(
                showlegend=False,
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            cols_box[i % 2].plotly_chart(fig_box, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Model Performance tab
# ─────────────────────────────────────────────────────────────────────────────
def render_model_performance(metrics: dict):
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    acc    = metrics["acc"]
    roc    = metrics["roc_auc"]
    cr     = metrics["cr"]
    fpr    = metrics["fpr"]
    tpr    = metrics["tpr"]
    cm     = metrics["cm"]
    y_test = metrics["y_test"]
    y_pred = metrics["y_pred"]

    # ── Headline metrics ────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.1%}")
    m2.metric("ROC-AUC",  f"{roc:.4f}")
    m3.metric("Precision (Class 1)", f"{cr.get('1', cr.get(1, {})).get('precision', 0):.3f}")
    m4.metric("Recall (Class 1)",    f"{cr.get('1', cr.get(1, {})).get('recall',    0):.3f}")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    col_cm, col_roc = st.columns([1, 1.3])

    # ── Confusion Matrix ────────────────────────────────────────────────────
    with col_cm:
        st.markdown("**Confusion Matrix**")
        labels = ["No Default (0)", "Default (1)"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels, y=labels,
            text_auto=True,
            color_continuous_scale=[[0,"#13162a"],[0.5,"#7b0d1e"],[1,"#e63946"]],
            template=PLOTLY_TEMPLATE,
            title="Confusion Matrix",
        )
        fig_cm.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_showscale=False,
        )
        fig_cm.update_traces(textfont_size=18)
        st.plotly_chart(fig_cm, width='stretch')
        st.caption(
            "**TP / TN** = correct predictions. "
            "**FP** = false alarms. "
            "**FN** = missed defaults (most costly in credit risk)."
        )

    # ── ROC Curve ───────────────────────────────────────────────────────────
    with col_roc:
        st.markdown("**ROC Curve**")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Logistic Regression (AUC = {roc:.4f})",
            line=dict(color=ACCENT, width=3),
            fill="tozeroy", fillcolor="rgba(230,57,70,0.12)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Classifier",
            line=dict(color="#607d8b", width=1.5, dash="dash"),
        ))
        fig_roc.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"ROC Curve — AUC = {roc:.4f}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1),
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_roc, width='stretch')
        st.caption(
            "📌 **ROC-AUC** measures the model's ability to distinguish defaults from non-defaults "
            "across all thresholds. An AUC close to 1.0 is excellent; 0.5 = random guessing."
        )

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Classification Report ───────────────────────────────────────────────
    st.markdown("**Full Classification Report**")
    report_df = (
        pd.DataFrame(cr)
        .transpose()
        .round(4)
        .drop(index=["accuracy"], errors="ignore")
    )
    st.dataframe(report_df.style.background_gradient(cmap="Blues", axis=0), width='stretch')
    st.caption(
        "⚠️ **Why accuracy alone is insufficient**: In imbalanced credit datasets, "
        "a model predicting 'No Default' 100% of the time achieves high accuracy but zero recall on defaults. "
        "Prioritise **Recall** and **ROC-AUC** for credit risk models."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Customer Risk Prediction tab
# ─────────────────────────────────────────────────────────────────────────────
def render_prediction_panel(model, feature_columns):
    st.markdown('<div class="section-header">Single Customer Risk Assessment</div>', unsafe_allow_html=True)
    st.caption("Fill in the borrower details below and click **Assess Risk** to get a real-time prediction.")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Numeric Inputs ──────────────────────────────────────────────────────
    st.markdown("**📐 Numeric Attributes**")
    num_col1, num_col2, num_col3 = st.columns(3)

    income       = num_col1.number_input("Annual Income ($)", min_value=0.0, step=1000.0, format="%.2f")
    loan_amount  = num_col2.number_input("Loan Amount ($)",   min_value=0.0, step=1000.0, format="%.2f")
    credit_score = num_col3.slider("Credit Score", min_value=300, max_value=850, value=650, step=1)

    num_col4, num_col5 = st.columns(2)
    ltv = num_col4.slider("Loan-to-Value Ratio (LTV %)", min_value=0.0, max_value=200.0, value=80.0, step=0.5)
    dti = num_col5.slider("Debt-to-Income Ratio (DTI %)", min_value=0.0, max_value=100.0, value=35.0, step=0.5)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Categorical Inputs ──────────────────────────────────────────────────
    st.markdown("**📑 Categorical Attributes**")
    cat_col1, cat_col2, cat_col3 = st.columns(3)

    credit_worthiness = cat_col1.selectbox(
        "Credit Worthiness",
        options=["l1", "l2"],
        help="l1 = Prime / l2 = Sub-prime",
    )
    open_credit = cat_col2.selectbox(
        "Open Credit Lines",
        options=["opc", "nopc"],
        help="opc = Open Credit / nopc = No Open Credit",
    )
    interest_only = cat_col3.selectbox(
        "Interest Only Loan",
        options=["int_only", "not_int"],
        help="Whether the loan is interest-only",
    )

    cat_col4, cat_col5, cat_col6 = st.columns(3)
    loan_limit   = cat_col4.selectbox("Loan Limit Type",      ["cf", "ncf"], help="cf = Conforming / ncf = Non-conforming")
    business_flag= cat_col5.selectbox("Business / Commercial",["ob/c", "nob/c"], help="ob/c = Business / nob/c = Personal")
    occupancy    = cat_col6.selectbox("Occupancy Type",        ["pr", "sr", "ir"], help="pr = Primary / sr = Secondary / ir = Investment")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Predict Button ──────────────────────────────────────────────────────
    if st.button("⚡ Assess Risk Now", type="primary", width='stretch'):
        input_dict = {col: 0 for col in feature_columns}

        # Fill numeric
        for key, val in [
            ("income", income), ("loan_amount", loan_amount),
            ("Credit_Score", credit_score), ("LTV", ltv), ("dtir1", dti),
        ]:
            if key in input_dict:
                input_dict[key] = val

        # Fill categorical
        for key, val in [
            ("Credit_Worthiness", credit_worthiness),
            ("open_credit", open_credit),
            ("interest_only", interest_only),
            ("loan_limit", loan_limit),
            ("business_or_commercial", business_flag),
            ("occupancy_type", occupancy),
        ]:
            if key in input_dict:
                input_dict[key] = val

        input_df    = pd.DataFrame([input_dict])
        probability = model.predict_proba(input_df)[0][1]
        pct         = probability * 100

        # ── Results layout ──────────────────────────────────────────────────
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            # Plotly Gauge
            if pct < 30:
                gauge_color = ACCENT_MINT    # green for low risk
            elif pct < 60:
                gauge_color = ACCENT3         # gold for medium risk
            else:
                gauge_color = ACCENT          # red for high risk

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pct,
                number={"suffix": "%", "font": {"size": 38, "color": "#e8eaf6"}},
                delta={"reference": 50, "increasing": {"color": ACCENT}, "decreasing": {"color": ACCENT_MINT}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
                    "bar": {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  30], "color": "rgba(76,175,80,0.15)"},
                        {"range": [30, 60], "color": "rgba(255,152,0,0.15)"},
                        {"range": [60,100], "color": "rgba(239,83,80,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.8,
                        "value": pct,
                    },
                },
                title={"text": "Default Probability", "font": {"size": 16, "color": "#8b8fa8"}},
            ))
            fig_gauge.update_layout(
                template=PLOTLY_TEMPLATE,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_gauge, width='stretch')

        with res_col2:
            st.markdown("### Risk Assessment Result")
            st.progress(int(pct))
            st.markdown(f"**Default Probability:** `{probability:.3f}` ({pct:.1f}%)")

            if pct < 30:
                st.markdown('<div class="risk-low">✅ LOW RISK</div>', unsafe_allow_html=True)
                st.success("This borrower presents a **low default risk**. Loan approval recommended subject to standard checks.")
            elif pct < 60:
                st.markdown('<div class="risk-medium">⚠️ MEDIUM RISK</div>', unsafe_allow_html=True)
                st.warning("This borrower presents a **moderate default risk**. Additional due diligence is advised before approval.")
            else:
                st.markdown('<div class="risk-high">🚨 HIGH RISK</div>', unsafe_allow_html=True)
                st.error("This borrower presents a **high default risk**. Loan approval is not recommended without significant collateral.")

            st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
            st.markdown("**Risk Level Guide**")
            st.markdown("""
            | Range | Category | Action |
            |-------|----------|--------|
            | 0–30% | 🟢 Low | Approve |
            | 30–60%| 🟠 Medium | Review |
            | 60–100%| 🔴 High | Decline |
            """)

        st.toast(f"Prediction complete — {pct:.1f}% default probability", icon="💳")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## \U0001f4b3 Credit Risk Scoring")
    st.markdown("*Professional Analytics Dashboard*")
    st.markdown("---")

    # ── Option A: upload a custom CSV ──────────────────────────────────────
    uploaded_file = st.file_uploader(
        "\U0001f4c1 Upload Dataset (CSV)",
        type=["csv"],
        help="Upload your borrower dataset to train or analyse the model.",
    )

    # ── Option B: use the local default dataset ─────────────────────────────
    local_csv_exists = os.path.exists(LOCAL_CSV_PATH)
    if local_csv_exists:
        st.markdown(
            "<div style='text-align:center; color:#555a78; font-size:12px; margin:4px 0'>or</div>",
            unsafe_allow_html=True,
        )
        use_default = st.button(
            "\U0001f5c4 Use Default Dataset",
            help=f"Load the built-in dataset at {LOCAL_CSV_PATH}",
            width='stretch',
        )
    else:
        use_default = False

    st.markdown("---")

    model_loaded, features_loaded = load_model()
    if model_loaded:
        st.success("\u2705 Model loaded from disk")
    else:
        st.info("\u2139\ufe0f No model found \u2014 upload a CSV and train.")

    # ── Train / Retrain from uploaded file ──────────────────────────────────
    if uploaded_file and st.button("\U0001f680 Train / Retrain Model", type="primary", width='stretch'):
        raw_df  = pd.read_csv(uploaded_file)
        metrics_new = train_and_cache_model(raw_df)
        st.session_state["metrics"] = metrics_new
        st.session_state["df"]      = raw_df
        auto_initialize.clear()
        st.success("\u2705 Model trained successfully!")
        st.rerun()

    # ── Load default dataset (train if no model, evaluate if model exists) ──
    if use_default:
        with st.spinner("Loading default dataset\u2026"):
            default_df = pd.read_csv(LOCAL_CSV_PATH)
        if model_loaded:
            # Model already on disk \u2014 just evaluate
            auto_initialize.clear()
            st.session_state["df"] = default_df
            st.session_state.pop("metrics", None)   # let auto_initialize re-run
            st.rerun()
        else:
            # No model yet \u2014 train on the default CSV
            metrics_new = train_and_cache_model(default_df)
            st.session_state["metrics"] = metrics_new
            st.session_state["df"]      = default_df
            auto_initialize.clear()
            st.success("\u2705 Trained on default dataset!")
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <small>
        Built with Streamlit · Scikit-learn · Plotly<br>
        Logistic Regression Pipeline<br><br>
        © 2025 Credit Analytics Group
        </small>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='color:#c7c9e8; margin-bottom:2px; font-size:32px;'>
      💳 Credit Risk Scoring Dashboard
    </h1>
    <p style='color:#8b8fa8; margin-top:0; font-size:15px;'>
      Intelligent Borrower Default Prediction · Powered by Logistic Regression
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ── Determine data source ───────────────────────────────────────────────────
if uploaded_file is not None and "df" not in st.session_state:
    st.session_state["df"] = pd.read_csv(uploaded_file)

df_main  = st.session_state.get("df", None)
metrics  = st.session_state.get("metrics", None)

# ── AUTO-INIT: if model+local CSV exist, compute metrics without uploading ───
if metrics is None and model_loaded is not None and os.path.exists(LOCAL_CSV_PATH):
    metrics = auto_initialize(LOCAL_CSV_PATH)
    if metrics is not None and df_main is None:
        df_main = pd.read_csv(LOCAL_CSV_PATH)

# ── Active model / features ──────────────────────────────────────────────────
active_model    = metrics["pipeline"] if metrics else model_loaded
active_features = (
    list(metrics["X"].columns) if metrics else
    (features_loaded if features_loaded is not None else [])
)

# ─────────────────────────────────────────────────────────────────────────────
# RENDER based on state
# ─────────────────────────────────────────────────────────────────────────────
if df_main is not None and metrics is not None:
    # ── Full dashboard (model + data available) ─────────────────────────────
    X_main = metrics["X"]
    y_main = metrics["y"]

    total_customers = len(df_main)
    churn_rate      = float(y_main.mean() * 100)

    # ── Executive Summary ───────────────────────────────────────────────────
    render_executive_summary(total_customers, churn_rate, metrics["acc"], metrics["roc_auc"])

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_overview, tab_features, tab_perf, tab_predict = st.tabs([
        "\U0001f5c3 Data Overview",
        "\U0001f50d Feature Insights",
        "\U0001f3c6 Model Performance",
        "\U0001f3af Customer Risk Prediction",
    ])

    with tab_overview:
        render_data_overview(df_main.drop(columns=["Status"], errors="ignore"), y_main)

    with tab_features:
        render_feature_insights(metrics["pipeline"], X_main, y_main)

    with tab_perf:
        render_model_performance(metrics)

    with tab_predict:
        render_prediction_panel(active_model, active_features)

elif active_model is not None:
    # ── Prediction-only (model on disk but local CSV missing) ───────────────
    st.warning(
        "\U0001f4cc Model loaded. The local dataset was not found at "
        f"`{LOCAL_CSV_PATH}`. Upload a CSV via the sidebar and click "
        "**\U0001f680 Train / Retrain Model** to unlock all analytics tabs."
    )
    render_prediction_panel(active_model, active_features)

else:
    # ── Onboarding ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding:60px 0;'>
          <div style='font-size:64px;'>\U0001f4b3</div>
          <h2 style='color:#c7c9e8;'>Welcome to the Credit Risk Dashboard</h2>
          <p style='color:#8b8fa8; max-width:500px; margin:0 auto;'>
            Upload your borrower dataset using the sidebar to begin analysis,
            train the model, and explore interactive insights.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )