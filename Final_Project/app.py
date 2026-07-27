"""
Streamlit application for the Telco Customer Churn Prediction project.

Loads the fitted preprocessor (preprocessor.pkl) and the tuned final model
pipeline (saved_model.pkl) produced by the companion Jupyter notebook, and
serves live churn predictions for a single customer entered through a form.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from model_utils import ColumnSelector  # noqa: F401  (required for unpickling saved_model.pkl)

st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📉",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("saved_model.pkl")
    return preprocessor, model


preprocessor, model = load_artifacts()

st.title("📉 Telco Customer Churn Predictor")
st.write(
    "Enter a customer's account details below to predict whether they are "
    "likely to churn. This app uses a model trained on the IBM Telco Customer "
    "Churn dataset."
)

with st.form("customer_form"):
    st.subheader("Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col3:
        partner = st.selectbox("Has Partner", ["No", "Yes"])

    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.subheader("Account Information")
    col4, col5, col6 = st.columns(3)
    with col4:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
    with col5:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
    with col6:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=840.0, step=1.0)

    col7, col8 = st.columns(2)
    with col7:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col8:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.subheader("Services")
    col9, col10, col11 = st.columns(3)
    with col9:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with col10:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    with col11:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    col12, col13, col14 = st.columns(3)
    with col12:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    with col13:
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    with col14:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    col15, col16, col17 = st.columns(3)
    with col15:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    with col16:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    with col17:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    # Apply preprocessing
    X_transformed = preprocessor.transform(input_df)

    # Make prediction
    prediction = model.predict(X_transformed)[0]

    # Convert NumPy float to native Python float
    probability = float(model.predict_proba(X_transformed)[0, 1])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.metric(
        label="Churn Probability",
        value=f"{probability * 100:.1f}%"
    )

    # Streamlit progress bar requires a native Python float
    progress_value = float(max(0.0, min(probability, 1.0)))
    st.progress(progress_value)

    with st.expander("View submitted customer data"):
        st.dataframe(
            input_df.T.rename(columns={0: "Value"}),
            use_container_width=True,
        )

st.markdown("---")

st.caption(
    "Model trained on the IBM Telco Customer Churn dataset. "
    "For educational / graduation-project purposes only."
)