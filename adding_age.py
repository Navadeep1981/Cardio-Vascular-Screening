import pandas as pd
import numpy as np

# -------------------------------
# 1️⃣ Load Original Dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\Navadeep\OneDrive\Desktop\CardiVascular Screening\data\ehr_records.csv")

# -------------------------------
# 2️⃣ Remove Identifier Column
# -------------------------------
if "patient_id" in df.columns:
    df = df.drop(columns=["patient_id"])

# -------------------------------
# 3️⃣ Generate Realistic Age Distribution
# -------------------------------

n = len(df)

# Demographic split:
# 25% Pediatric (1–18)
# 50% Adult (19–60)
# 25% Elderly (61–85)

ages = np.concatenate([
    np.random.randint(1, 19, int(0.25 * n)),     # Pediatric
    np.random.randint(19, 61, int(0.50 * n)),    # Adult
    np.random.randint(61, 86, int(0.25 * n))     # Elderly
])

np.random.shuffle(ages)

# -------------------------------
# 4️⃣ Insert Age as FIRST Column
# -------------------------------

df.insert(0, "age", ages[:n])

# -------------------------------
# 5️⃣ Introduce Age-Based Risk Correlation
# -------------------------------

for i in range(len(df)):
    if df.loc[i, "age"] > 60:
        if np.random.rand() < 0.20:   # Elderly higher risk
            df.loc[i, "progressed_to_critical"] = 1

    elif df.loc[i, "age"] <= 18:
        if np.random.rand() < 0.05:   # Pediatric lower risk
            df.loc[i, "progressed_to_critical"] = 1

# -------------------------------
# 6️⃣ Verify Correlation
# -------------------------------

age_groups = pd.cut(df["age"], bins=[0, 18, 60, 100], labels=["Pediatric", "Adult", "Elderly"])

print(df.groupby(age_groups)["progressed_to_critical"].mean())

df.head()

print("\nColumns after processing:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

df.to_csv("data/ehr_records_with_age.csv", index=False)
print("Updated dataset saved successfully!")

