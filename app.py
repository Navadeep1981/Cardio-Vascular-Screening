import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import base64
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Cardiovascular Risk Prediction", layout="centered")

# -------------------------
# Load Model
# -------------------------
saved_xgb = joblib.load("models/xgb_model.pkl")
model = saved_xgb["model"]
feature_list = saved_xgb["features"]

# -------------------------
# Auto Generate Patient ID
# -------------------------
def generate_patient_id():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    rand = random.randint(100, 999)
    return f"PT-{timestamp}-{rand}"

if "patient_id" not in st.session_state:
    st.session_state.patient_id = generate_patient_id()

patient_id = st.session_state.patient_id

# -------------------------
# Watermark Image
# -------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

heart_base64 = get_base64_image("assets/heart.png")

# -------------------------
# Random Quote
# -------------------------
quotes = [
    "Prevention is better than cure.",
    "A healthy heart beats longer.",
    "Care today, live tomorrow.",
    "Early detection saves lives.",
    "Your heart deserves attention."
]

selected_quote = random.choice(quotes)

# -------------------------
# Background Styling
# -------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{heart_base64}");
        background-size: 250px;
        background-repeat: no-repeat;
        background-position: bottom right;
        opacity: 0.98;
    }}

    .quote {{
        position: fixed;
        top: 20px;
        right: 30px;
        font-size: 16px;
        font-style: italic;
        color: #E63946;
        font-weight: 500;
    }}
    </style>

    <div class="quote">
        "{selected_quote}"
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Health Score Logic
# -------------------------
def calculate_health_scores(bp_sys, heart_rate, oxygen, temp):

    score = 0

    if bp_sys > 140 or bp_sys < 90:
        score += 2

    if heart_rate > 100 or heart_rate < 55:
        score += 2

    if oxygen < 94:
        score += 3

    if temp > 38:
        score += 1

    symptom_severity = min(score * 2, 10)
    med_adherence = max(0.2, 1 - (score * 0.12))

    return round(med_adherence, 2), symptom_severity

# -------------------------
# Title
# -------------------------
st.title("🫀 Cardiovascular Critical Risk Prediction System")
st.write("Enter patient clinical details below.")

st.text_input("Patient ID", value=patient_id, disabled=True)

# -------------------------
# Clinical Inputs
# -------------------------
age = st.number_input("Age", 1, 100, 45)
day = st.number_input("Hospital Day", 1, 30, 1)

bp_systolic = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
bp_diastolic = st.number_input("Diastolic BP (mmHg)", 50, 150, 80)

heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 75)
respiratory_rate = st.number_input("Respiratory Rate", 10, 40, 18)

temperature = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0)
oxygen_saturation = st.number_input("Oxygen Saturation (%)", 70, 100, 98)

# Auto Scores
med_adherence, symptom_severity = calculate_health_scores(
    bp_systolic,
    heart_rate,
    oxygen_saturation,
    temperature
)

col1, col2 = st.columns(2)
col1.metric("Medication Adherence (Auto)", med_adherence)
col2.metric("Symptom Severity (Auto)", symptom_severity)

# -------------------------
# Predict Button
# -------------------------
if st.button("Predict Risk"):

    patient_data = pd.DataFrame([{
        "patient_id": patient_id,
        "age": age,
        "day": day,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "temperature": temperature,
        "oxygen_saturation": oxygen_saturation,
        "med_adherence": med_adherence,
        "symptom_severity": symptom_severity
    }])

    # Feature Engineering
    patient_data["pulse_pressure"] = patient_data["bp_systolic"] - patient_data["bp_diastolic"]
    patient_data["shock_index"] = patient_data["heart_rate"] / patient_data["bp_systolic"]

    model_input = patient_data.drop("patient_id", axis=1)
    model_input = model_input[feature_list]

    prediction = model.predict(model_input)[0]
    probability = model.predict_proba(model_input)[0][1]

    st.session_state.prediction = prediction
    st.session_state.probability = probability
    st.session_state.patient_data = patient_data

# -------------------------
# Show Prediction
# -------------------------
if "prediction" in st.session_state:

    prediction = st.session_state.prediction
    probability = st.session_state.probability

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"⚠ HIGH RISK\nProbability: {probability*100:.2f}%")
    else:
        st.success(f"✔ LOW RISK\nProbability: {(1-probability)*100:.2f}%")

    st.divider()

    # Lifestyle Section
    st.subheader("🩺 Lifestyle & Health Assessment")

    smoke = st.radio("Do you smoke?", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol consumption", ["Never", "Occasionally", "Weekly", "Daily"])
    exercise = st.radio("Do you exercise regularly?", ["Yes", "No"])
    diabetes = st.radio("Do you have diabetes?", ["No", "Yes"])
    bp_history = st.radio("History of high BP?", ["No", "Yes"])
    family_history = st.radio("Family history of heart disease?", ["No", "Yes"])
    stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])

    if st.button("Get Personalized Recommendations"):

        st.subheader("📋 Recommendations")

        if smoke == "Yes":
            st.warning("🚭 Smoking increases cardiovascular risk.")
        if alcohol in ["Weekly", "Daily"]:
            st.warning("🍺 Reduce alcohol intake.")
        if exercise == "No":
            st.info("🏃 Exercise at least 30 minutes daily.")
        if diabetes == "Yes":
            st.warning("🩸 Monitor blood glucose regularly.")
        if bp_history == "Yes":
            st.warning("📈 Maintain blood pressure control.")
        if family_history == "Yes":
            st.info("🧬 Regular cardiac checkups recommended.")
        if stress == "High":
            st.info("🧘 Practice stress management techniques.")

        if prediction == 1:
            st.error("⚠ Immediate medical consultation advised.")
        else:
            st.success("✔ Maintain healthy lifestyle.")

        # Save Record
        patient_data = st.session_state.patient_data
        patient_data["prediction"] = prediction
        patient_data["critical_probability"] = probability
        patient_data["date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_path = "data/patient_predictions.csv"

        if os.path.exists(file_path):
            old = pd.read_csv(file_path)
            patient_data = pd.concat([old, patient_data], ignore_index=True)

        patient_data.to_csv(file_path, index=False)

# -------------------------
# Model Dashboard
# -------------------------
st.divider()
st.header("📊 Model Comparison Dashboard")

if st.button("Show Model Comparison"):

    df = pd.read_csv("data/synthetic_ehr_with_age.csv")

    df["pulse_pressure"] = df["bp_systolic"] - df["bp_diastolic"]
    df["shock_index"] = df["heart_rate"] / df["bp_systolic"]

    if "patient_id" in df.columns:
        df = df.drop("patient_id", axis=1)

    X = df.drop("progressed_to_critical", axis=1)
    y = df["progressed_to_critical"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "XGBoost": joblib.load("models/xgb_model.pkl")["model"],
        "Random Forest": joblib.load("models/random_forest.pkl")["model"],
        "Decision Tree": joblib.load("models/decision_tree.pkl")["model"],
        "Logistic Regression": joblib.load("models/logistic_regression.pkl")["model"],
        "SVM": joblib.load("models/svm.pkl")["model"]
    }

    accuracies = {}
    auc_scores = {}

    plt.figure()

    for name, model in models.items():

        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        auc_scores[name] = roc_auc

        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    st.pyplot(plt)

    acc_df = pd.DataFrame({
        "Model": list(accuracies.keys()),
        "Accuracy": list(accuracies.values())
    }).sort_values(by="Accuracy", ascending=False)

    st.subheader("📈 Accuracy Comparison")
    st.bar_chart(acc_df.set_index("Model"))

    auc_df = pd.DataFrame({
        "Model": list(auc_scores.keys()),
        "AUC Score": list(auc_scores.values())
    })

    st.subheader("📊 AUC Scores")
    st.table(auc_df)