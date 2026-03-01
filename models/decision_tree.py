import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------
# 1️⃣ Load Dataset
# ----------------------------------

df = pd.read_csv("C:/Users/Navadeep/OneDrive/Desktop/CardiVascular Screening/data/synthetic_ehr_with_age.csv")

print("Dataset Loaded")
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
# 6️⃣ Train Decision Tree
# ----------------------------------

model = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------
# 7️⃣ Evaluation
# ----------------------------------

y_pred = model.predict(X_test)

print("\n===== Decision Tree Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------
# 8️⃣ Save Model + Features
# ----------------------------------

joblib.dump({
    "model": model,
    "features": feature_columns
}, "decision_tree.pkl")

print("Decision Tree model saved successfully!")
