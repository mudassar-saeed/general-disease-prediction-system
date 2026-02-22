import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# Paths
# --------------------------------------------------

# Define source and destination file paths
RAW_DATA_PATH = "data/raw/dataset.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

print("Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)

# --------------------------------------------------
# Strip Whitespace from Column Names
# --------------------------------------------------

# Remove leading/trailing spaces from CSV headers
df.columns = df.columns.str.strip()

# --------------------------------------------------
# Clean Disease Names (strip spaces + fix typos)
# --------------------------------------------------

# Dictionary to standardize inconsistent disease names
disease_name_fixes = {
    "Diabetes ":                             "Diabetes",
    "Hypertension ":                          "Hypertension",
    "Peptic ulcer diseae":                    "Peptic ulcer disease",
    "Dimorphic hemmorhoids(piles)":           "Dimorphic hemorrhoids (piles)",
    "(vertigo) Paroymsal  Positional Vertigo":"(vertigo) Paroxysmal Positional Vertigo",
}

# Clean whitespace and apply name corrections to the target column
df["Disease"] = df["Disease"].str.strip()
df["Disease"] = df["Disease"].replace(disease_name_fixes)

print(f"Unique diseases after cleaning: {df['Disease'].nunique()}")

# --------------------------------------------------
# Clean Symptom Values (strip spaces)
# --------------------------------------------------

# Identify all symptom columns (excluding 'Disease')
symptom_columns = df.columns[1:]

# Remove extra spaces from every symptom entry in the dataframe
for col in symptom_columns:
    df[col] = df[col].str.strip()

# --------------------------------------------------
# Fill Missing Values
# --------------------------------------------------

# Replace empty/NaN cells with a "None" string placeholder
df.fillna("None", inplace=True)

# --------------------------------------------------
# Encode Symptoms (One-Hot Style) — Vectorized
# --------------------------------------------------

print("Encoding symptoms...")

# Create a unique, sorted list of all symptoms present in the data
all_symptoms = set()
for col in symptom_columns:
    all_symptoms.update(df[col].unique())

all_symptoms.discard("None")
all_symptoms = sorted(all_symptoms)

# Initialize a new DataFrame with 0s for every possible symptom
encoded_df = pd.DataFrame(0, index=df.index, columns=all_symptoms)

# Set value to 1 if a symptom appears in any of the original columns
for col in symptom_columns:
    for symptom in all_symptoms:
        mask = df[col] == symptom
        encoded_df.loc[mask, symptom] = 1

# --------------------------------------------------
# Encode Target
# --------------------------------------------------

print("Encoding target (Disease)...")

# Convert text disease names into numerical integers for the model
label_encoder = LabelEncoder()
disease_encoded = label_encoder.fit_transform(df["Disease"])
disease_series = pd.Series(disease_encoded, index=df.index, name="Disease")

print(f"Classes: {list(label_encoder.classes_)}")

# --------------------------------------------------
# Combine Features + Target
# --------------------------------------------------

# Merge the binary symptom features with the encoded disease labels
final_df = pd.concat([encoded_df, disease_series], axis=1)

# --------------------------------------------------
# Save Processed Data
# --------------------------------------------------

# Ensure directory exists and export the cleaned CSV
os.makedirs("data/processed", exist_ok=True)
final_df.to_csv(PROCESSED_DATA_PATH, index=False)

# --------------------------------------------------
# Save Encoder + Feature Columns
# --------------------------------------------------

# Save metadata objects for use during real-time user inference
os.makedirs("models", exist_ok=True)

joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(list(encoded_df.columns), "models/feature_columns.pkl")

print("\nPreprocessing complete!")
print(f"Processed data shape: {final_df.shape}")
print("Processed data saved at:", PROCESSED_DATA_PATH)
print("Label encoder saved at: models/label_encoder.pkl")
print("Feature columns saved at: models/feature_columns.pkl")