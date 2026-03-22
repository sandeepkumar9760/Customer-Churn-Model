"""
dashboard/app.py
----------------
Streamlit Dashboard for Customer Churn Prediction
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import build_features, prepare_single_record

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1E40AF; }
    .metric-label { font-size: 0.85rem; color: #64748B; margin-top: 4px; }
    .risk-high   { background: #FEF2F2; border: 1px solid #FECACA; border-radius: 10px; padding: 16px; text-align: center; }
    .risk-medium { background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px; padding: 16px; text-align: center; }
    .risk-low    { background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 16px; text-align: center; }
    .risk-value  { font-size: 2.5rem; font-weight: 800; }
    .risk-label  { font-size: 0.9rem; margin-top: 4px; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1E293B; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Load model & data ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), '..', 'model', 'churn_model.pkl')
    return joblib.load(path)

@st.cache_data
def load_predictions():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'test_predictions.csv')
    return pd.read_csv(path)

@st.cache_data
def load_raw():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

model = load_model()
preds = load_predictions()
raw   = load_raw()

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
st.sidebar.title("📉 Churn Predictor")
st.sidebar.markdown("**ShyftLabs Portfolio Project**")
st.sidebar.divider()
st.sidebar.markdown("**Stack:** XGBoost · SHAP · FastAPI · Streamlit")
st.sidebar.markdown("**Author:** Sandeep Kumar · LPU")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Live Predictor", "📊 Data Insights", "📈 Model Performance"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1: LIVE PREDICTOR
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.title("🎯 Real-Time Churn Risk Predictor")
    st.markdown("Fill in the customer details below and click **Predict** to get an instant churn risk score.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Customer Profile**")
        gender         = st.selectbox("Gender", ["Male", "Female"])
        senior         = st.selectbox("Senior Citizen", [0, 1])
        partner        = st.selectbox("Partner", ["Yes", "No"])
        dependents     = st.selectbox("Dependents", ["Yes", "No"])
        tenure         = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("**📱 Services**")
        phone          = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec     = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_bkp     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_prot    = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support   = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv   = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_mv   = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("**💳 Billing**")
        contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment        = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly        = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
        total          = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly * tenure), step=1.0)

    st.divider()
    predict_btn = st.button("⚡ Predict Churn Risk", type="primary", use_container_width=True)

    if predict_btn:
        customer = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
            "DeviceProtection": device_prot, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_mv,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        X = prepare_single_record(customer)
        churn_prob = float(model.predict_proba(X)[0][1])
        churn_pred = int(churn_prob >= 0.5)

        if churn_prob >= 0.6:
            risk = "High"; css = "risk-high"; color = "#DC2626"
            rec = "🚨 Immediate intervention required. Offer a personalised retention deal."
        elif churn_prob >= 0.3:
            risk = "Medium"; css = "risk-medium"; color = "#D97706"
            rec = "⚠️ Proactively offer loyalty discount or contract upgrade."
        else:
            risk = "Low"; css = "risk-low"; color = "#16A34A"
            rec = "✅ Customer is stable. Standard retention programme sufficient."

        st.divider()
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.markdown(f"""
            <div class='{css}'>
                <div class='risk-value' style='color:{color}'>{churn_prob*100:.1f}%</div>
                <div class='risk-label'>Churn Probability</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class='{css}'>
                <div class='risk-value' style='color:{color}'>{risk}</div>
                <div class='risk-label'>Risk Level</div>
            </div>""", unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{"Churn" if churn_pred else "Retain"}</div>
                <div class='metric-label'>Prediction</div>
            </div>""", unsafe_allow_html=True)

        with r4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{(1-churn_prob)*100:.1f}%</div>
                <div class='metric-label'>Retention Probability</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"### 💡 Recommendation\n> {rec}")

        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            title={'text': "Churn Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30],  'color': '#F0FDF4'},
                    {'range': [30, 60], 'color': '#FFFBEB'},
                    {'range': [60, 100],'color': '#FEF2F2'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': churn_prob * 100
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2: DATA INSIGHTS
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title("📊 Data Insights — Telco Churn Dataset")

    # KPIs
    churn_rate = (raw['Churn'] == 'Yes').mean() * 100
    avg_tenure = raw['tenure'].mean()
    avg_monthly = raw['MonthlyCharges'].mean()
    total_customers = len(raw)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{total_customers:,}</div>
            <div class='metric-label'>Total Customers</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{churn_rate:.1f}%</div>
            <div class='metric-label'>Overall Churn Rate</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{avg_tenure:.0f}mo</div>
            <div class='metric-label'>Avg Tenure</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>${avg_monthly:.0f}</div>
            <div class='metric-label'>Avg Monthly Charges</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Churn Rate by Contract Type</div>", unsafe_allow_html=True)
        contract_churn = raw.groupby('Contract').apply(
            lambda x: (x['Churn'] == 'Yes').mean() * 100
        ).reset_index()
        contract_churn.columns = ['Contract', 'Churn Rate (%)']
        fig = px.bar(contract_churn, x='Contract', y='Churn Rate (%)',
                     color='Contract', color_discrete_sequence=['#DC2626','#D97706','#16A34A'])
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Churn Rate by Internet Service</div>", unsafe_allow_html=True)
        internet_churn = raw.groupby('InternetService').apply(
            lambda x: (x['Churn'] == 'Yes').mean() * 100
        ).reset_index()
        internet_churn.columns = ['InternetService', 'Churn Rate (%)']
        fig = px.bar(internet_churn, x='InternetService', y='Churn Rate (%)',
                     color='InternetService', color_discrete_sequence=['#2563EB','#DC2626','#16A34A'])
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Tenure distribution by churn
    st.markdown("<div class='section-header'>Tenure Distribution by Churn Status</div>", unsafe_allow_html=True)
    fig = px.histogram(raw, x='tenure', color='Churn', nbins=40,
                       barmode='overlay', opacity=0.7,
                       color_discrete_map={'No': '#2563EB', 'Yes': '#DC2626'})
    fig.update_layout(height=300, margin=dict(t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-header'>Monthly Charges vs Tenure</div>", unsafe_allow_html=True)
        sample = raw.sample(min(1000, len(raw)), random_state=42)
        fig = px.scatter(sample, x='tenure', y='MonthlyCharges', color='Churn',
                         opacity=0.5, color_discrete_map={'No': '#2563EB', 'Yes': '#DC2626'})
        fig.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("<div class='section-header'>Churn Rate by Payment Method</div>", unsafe_allow_html=True)
        pay_churn = raw.groupby('PaymentMethod').apply(
            lambda x: (x['Churn'] == 'Yes').mean() * 100
        ).reset_index()
        pay_churn.columns = ['PaymentMethod', 'Churn Rate (%)']
        pay_churn['PaymentMethod'] = pay_churn['PaymentMethod'].str.replace(' (automatic)', '', regex=False)
        fig = px.bar(pay_churn, x='Churn Rate (%)', y='PaymentMethod',
                     orientation='h', color_discrete_sequence=['#2563EB'])
        fig.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL PERFORMANCE
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.title("📈 Model Performance")

    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
    import sklearn

    y_true = preds['actual']
    y_pred = preds['predicted']
    y_prob = preds['churn_probability']

    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    # KPIs
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{roc_auc:.3f}</div>
            <div class='metric-label'>ROC-AUC Score</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{precision:.3f}</div>
            <div class='metric-label'>Precision</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{recall:.3f}</div>
            <div class='metric-label'>Recall</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{f1:.3f}</div>
            <div class='metric-label'>F1 Score</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>ROC Curve</div>", unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f'AUC = {roc_auc:.3f}',
                                 line=dict(color='#2563EB', width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                 name='Random', line=dict(color='gray', dash='dash')))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=350, margin=dict(t=10,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x='Predicted', y='Actual'),
            x=['No Churn', 'Churned'],
            y=['No Churn', 'Churned'],
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=350, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Churn probability distribution
    st.markdown("<div class='section-header'>Predicted Churn Probability Distribution</div>", unsafe_allow_html=True)
    fig = px.histogram(preds, x='churn_probability', color='actual',
                       nbins=40, barmode='overlay', opacity=0.7,
                       color_discrete_map={0: '#2563EB', 1: '#DC2626'},
                       labels={'actual': 'Actual Churn'})
    fig.update_layout(height=300, margin=dict(t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
