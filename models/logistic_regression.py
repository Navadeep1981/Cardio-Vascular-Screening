import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------
# 1️⃣ Load Dataset
# ----------------------------------

df = pd.read_csv("C:/Users/Navadeep/OneDrive/Desktop/CardiVascular Screening/data/synthetic_ehr_with_age.csv")

print("Synthetic Dataset Loaded")
print("Shape:", df.shape)

# ----------------------------------
# 2️⃣ Remove patient_id (IMPORTANT)
# ----------------------------------

if "patient_id" in df.columns:
    df = df.drop("patient_id", axis=1)

# ----------------------------------
# 3️⃣ Feature Engineering
# ----------------------------------

df["pulse_pressure"] = df["bp_systolic"] - df["bp_diastolic"]
df["shock_index"] = df["heart_rate"] / df["bp_systolic"]

# ----------------------------------
# 4️⃣ Define Features and Target
# ----------------------------------

X = df.drop("progressed_to_critical", axis=1)
y = df["progressed_to_critical"]

feature_columns = X.columns.tolist()

# ----------------------------------
# 5️⃣ Train-Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------
# 6️⃣ Scaling
# ----------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------
# 7️⃣ Train Model
# ----------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ----------------------------------
# 8️⃣ Evaluation
# ----------------------------------

y_pred = model.predict(X_test_scaled)

print("\n===== Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------
# 9️⃣ Save EVERYTHING Properly
# ----------------------------------

joblib.dump({
    "model": model,
    "scaler": scaler,
    "features": feature_columns
}, "logistic_regression.pkl")

print("Logistic model saved successfully!")
