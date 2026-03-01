import pandas as pd
from ctgan import CTGAN

# ----------------------------------
# Load Processed Dataset
# ----------------------------------

df = pd.read_csv("C:/Users/Navadeep/OneDrive/Desktop/CardiVascular Screening/data/ehr_records_with_age.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ----------------------------------
# Train CTGAN (Stable Version)
# ----------------------------------

ctgan = CTGAN(
    epochs=100,        # start with 100 (not 300)
    batch_size=500,    # helps stability
    verbose=True
)

ctgan.fit(
    df,
    discrete_columns=["progressed_to_critical"]
)

print("CTGAN Training Completed")

# ----------------------------------
# Generate 60,000 Synthetic Rows
# ----------------------------------

synthetic_data = ctgan.sample(60000)

print("Synthetic Data Generated")
print(synthetic_data.head())

# ----------------------------------
# Save Synthetic Dataset
# ----------------------------------

synthetic_data.to_csv("C:/Users/Navadeep/OneDrive/Desktop/CardiVascular Screening/data/synthetic_ehr_with_age.csv", index=False)

print("Synthetic dataset saved successfully!")
