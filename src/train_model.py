import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

# Define location for processed training data and final model output
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/disease_model.pkl"

# --------------------------------------------------
# Load Processed Dataset
# --------------------------------------------------

print("Loading processed dataset...")
df = pd.read_csv(PROCESSED_DATA_PATH)

# --------------------------------------------------
# Split Features and Target
# --------------------------------------------------

# Separate symptom binary features (X) from disease labels (y)
X = df.drop("Disease", axis=1)
y = df["Disease"]

print(f"Dataset shape       : {df.shape}")
print(f"Number of features  : {X.shape[1]}")
print(f"Number of classes   : {len(np.unique(y))}")
print(f"Class balance       : {y.value_counts().describe()[['min','max','mean']].to_dict()}")

# --------------------------------------------------
# Train-Test Split (Stratified)
# --------------------------------------------------

# Reserve 20% of data for testing, ensuring class ratios are preserved
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTrain size : {len(X_train)}")
print(f"Test size  : {len(X_test)}")

# --------------------------------------------------
# Model Training
# --------------------------------------------------

print("\nTraining Random Forest model...")

# Initialize Random Forest with 200 trees and balanced class weights
model = RandomForestClassifier(
    n_estimators=200,       # More trees = more stable predictions
    max_depth=None,         # Let trees grow fully (data is clean/structured)
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,              # Use all CPU cores for faster training
    class_weight="balanced",# Handles any class imbalance automatically
)

# Fit the model to the training data
model.fit(X_train, y_train)

# --------------------------------------------------
# Model Evaluation on Hold-Out Test Set
# --------------------------------------------------

# Generate predictions and calculate accuracy on unseen data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n================ MODEL PERFORMANCE ================")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

# Warning for 1.0 accuracy (common in synthetic medical datasets)
if accuracy == 1.0:
    print(
        "\n⚠  WARNING: 100% accuracy detected. This dataset is synthetic and "
        "perfectly balanced, so the model has memorized symptom-disease mappings. "
        "Do not interpret this as real-world performance."
    )

print("\nClassification Report:\n")
# Load original labels to provide readable names in the report
label_encoder = joblib.load("models/label_encoder.pkl")
target_names = label_encoder.classes_
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
)

# --------------------------------------------------
# Stratified Cross-Validation (more reliable estimate)
# --------------------------------------------------



print("Performing 5-Fold Stratified Cross-Validation...")
# Use 5-fold CV to check model stability across different data slices
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"CV Scores      : {np.round(cv_scores, 4)}")
print(f"Mean CV Acc    : {cv_scores.mean() * 100:.2f}%")
print(f"Std Dev        : {cv_scores.std() * 100:.2f}%")

# --------------------------------------------------
# Feature Importance (Top 15)
# --------------------------------------------------



print("\nTop 15 Most Important Symptoms:")

# Calculate and display which symptoms contribute most to the model's decisions
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:15]

for rank, i in enumerate(indices, 1):
    print(f"  {rank:>2}. {X.columns[i]:<40} {importances[i]:.4f}")

# --------------------------------------------------
# Save Model
# --------------------------------------------------

# Export the final trained model for use in the prediction script
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved at: {MODEL_PATH}")
print("Training completed successfully!")