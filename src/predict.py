import joblib
import pandas as pd
import numpy as np

# --------------------------------------------------
# Load Saved Model and Files
# --------------------------------------------------

print("Loading model and encoders...")

# Load the trained model, label encoder, and original feature names
model = joblib.load("models/disease_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# --------------------------------------------------
# Build Clean Feature Map for Robust Matching
# --------------------------------------------------

# Normalize feature names for case-insensitive matching
clean_feature_columns = [col.strip().lower() for col in feature_columns]
# Map clean names back to original column names
feature_map = dict(zip(clean_feature_columns, feature_columns))

# --------------------------------------------------
# Display Available Symptoms (optional helper)
# --------------------------------------------------

def list_available_symptoms():
    """Prints all valid symptoms in alphabetical order"""
    symptoms = sorted(clean_feature_columns)
    print("\nAvailable symptoms:")
    for i, s in enumerate(symptoms, 1):
        print(f"  {i:>3}. {s}")

# --------------------------------------------------
# Take User Input
# --------------------------------------------------

print("\n" + "=" * 50)
print("      DISEASE PREDICTION SYSTEM")
print("=" * 50)
print("Enter your symptoms separated by commas.")
print('Example: itching, skin_rash, fever\n')
print('Type "list" to see all available symptoms.')
print("=" * 50 + "\n")

user_input = input("Symptoms: ").strip()

# Show symptom list if requested
if user_input.lower() == "list":
    list_available_symptoms()
    user_input = input("\nSymptoms: ").strip()

# --------------------------------------------------
# Parse and Normalize User Input
# --------------------------------------------------

# Split, clean, and format user input to match feature naming conventions
input_symptoms = [
    sym.strip().lower().replace(" ", "_")
    for sym in user_input.split(",")
    if sym.strip()
]

# Error handling for empty input
if not input_symptoms:
    print("\n❌ No symptoms entered. Exiting...")
    exit()

# --------------------------------------------------
# Create Input Vector
# --------------------------------------------------

# Initialize empty row for the model (One-Hot Encoding format)
input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

recognized_symptoms = []
unrecognized_symptoms = []

# Fill the vector with 1s where the symptom matches the training features
for symptom in input_symptoms:
    if symptom in feature_map:
        original_column = feature_map[symptom]
        input_data.loc[0, original_column] = 1
        recognized_symptoms.append(symptom)
    else:
        unrecognized_symptoms.append(symptom)

# Alert user to unrecognized terms
if unrecognized_symptoms:
    print(f"\n⚠  Unrecognized symptom(s): {', '.join(unrecognized_symptoms)}")
    print("   Tip: type 'list' when prompted to see valid symptom names.")

# Stop if no valid symptoms were provided
if not recognized_symptoms:
    print("\n❌ No valid symptoms recognized. Exiting...")
    exit()

# --------------------------------------------------
# Make Prediction
# --------------------------------------------------

# Run model inference and get probability distribution
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)[0]

# Decode numerical label to disease name and get top confidence
predicted_disease = label_encoder.inverse_transform(prediction)[0]
confidence = np.max(prediction_proba) * 100

# --------------------------------------------------
# Display Results
# --------------------------------------------------

print("\n" + "=" * 50)
print("           PREDICTION RESULT")
print("=" * 50)
print(f"  Recognized Symptoms : {', '.join(recognized_symptoms)}")
print(f"  Predicted Disease   : {predicted_disease}")
print(f"  Confidence Level    : {confidence:.2f}%")

# Display warnings based on probability thresholds
if confidence < 50:
    print("\n  ⚠  Low confidence — symptoms may match multiple diseases.")
elif confidence < 75:
    print("\n  ℹ  Moderate confidence — consider reviewing top 3 results.")

# --------------------------------------------------
# Top 5 Most Probable Diseases
# --------------------------------------------------

# Get indices of the 5 highest probability scores
top5_indices = np.argsort(prediction_proba)[::-1][:5]

print("\n  Top 5 Most Probable Diseases:")
print(f"  {'Rank':<6} {'Disease':<45} {'Probability':>12}")
print("  " + "-" * 65)

# Loop through top 5 to display name, probability, and visual bar
for rank, idx in enumerate(top5_indices, 1):
    disease_name = label_encoder.inverse_transform([idx])[0]
    prob = prediction_proba[idx] * 100
    bar = "█" * int(prob / 5)
    print(f"  {rank:<6} {disease_name:<45} {prob:>10.2f}%  {bar}")

print("\n" + "=" * 50)
print("  ⚕  DISCLAIMER: This tool is for educational")
print("     purposes only. Always consult a qualified")
print("     medical professional for diagnosis.")
print("=" * 50)
print("\nPrediction completed successfully!")