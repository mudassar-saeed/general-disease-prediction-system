# 🩺 Disease Prediction System

A machine learning web application that predicts diseases based on user-selected symptoms. Built with Python, scikit-learn, and Flask, it features an interactive multi-select symptom UI and returns a ranked differential diagnosis with confidence scores.

---

## 📸 Features

- **Smart Symptom Search** — Searchable, tag-based multi-select input with live filtering and highlight matching
- **Instant Prediction** — Random Forest model predicts the most likely disease from your symptom selection
- **Confidence Score** — Visual progress bar shows how confident the model is in its prediction
- **Top 5 Differential Diagnosis** — Ranked list of the 5 most probable diseases with probability bars
- **REST API** — Clean JSON API endpoints for integration with other apps
- **CLI Mode** — Standalone command-line prediction script for terminal usage

---

## 🗂️ Project Structure

```
general-disease-prediction-system/
│
├── app.py                         # Flask web application & HTML UI
│
├── src/
│   ├── preprocess.py              # Data cleaning, encoding, and feature extraction
│   ├── train_model.py             # Model training, evaluation & cross-validation
│   └── predict.py                 # CLI-based prediction script
│
├── data/
│   ├── raw/
│   │   └── dataset.csv            # Original raw dataset
│   └── processed/
│       └── processed_data.csv     # One-hot encoded, cleaned dataset
│
├── models/
│   ├── disease_model.pkl          # Trained Random Forest model
│   ├── label_encoder.pkl          # LabelEncoder for disease names
│   └── feature_columns.pkl        # Ordered list of symptom feature columns
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| ML Framework | scikit-learn |
| Web Framework | Flask |
| Data Processing | pandas, NumPy |
| Model Persistence | joblib |
| Frontend | Vanilla HTML/CSS/JS (no frameworks) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mudassar-saeed/general-disease-prediction-system.git
cd general-disease-prediction-system
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web App

```bash
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

---

## 🔁 Retrain the Model (Optional)

If you want to retrain from scratch using your own data:

```bash
# Step 1 — Preprocess the raw dataset
python src/preprocess.py

# Step 2 — Train and evaluate the model
python src/train_model.py
```

---

## 💻 CLI Prediction (No Browser Needed)

```bash
python src/predict.py
```

When prompted, enter your symptoms separated by commas:

```
Symptoms: itching, skin_rash, nodal_skin_eruptions
```

Type `list` to see all available symptom names.

---

## 🌐 API Reference

### `GET /symptoms`

Returns all valid symptom names the model understands.

**Response:**
```json
{
  "symptoms": ["abdominal_pain", "acidity", "anxiety", ...]
}
```

---

### `POST /predict`

Predicts a disease from a comma-separated symptom string.

**Request Body:**
```json
{
  "symptoms": "itching, skin_rash, nodal_skin_eruptions"
}
```

**Response:**
```json
{
  "predicted_disease": "Fungal infection",
  "confidence": 97.5,
  "recognized_symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
  "unrecognized_symptoms": [],
  "top5": [
    { "disease": "Fungal infection", "probability": 97.5 },
    { "disease": "Chicken pox",      "probability": 1.2 },
    ...
  ]
}
```

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Trees | 200 |
| Class Weighting | Balanced (handles class imbalance) |
| Train/Test Split | 80% / 20% (Stratified) |
| Cross-Validation | 5-Fold Stratified CV |
| Feature Type | Binary one-hot encoded symptoms |

### Data Preprocessing Steps

1. Strip whitespace from column names and values
2. Fix known disease name typos (e.g. "Peptic ulcer diseae" → "Peptic ulcer disease")
3. Fill missing symptom values with `"None"` placeholder
4. One-hot encode all symptoms into binary feature columns
5. Label-encode disease names for classification

> ⚠️ **Note:** This dataset is synthetic and perfectly balanced by design, which results in near-perfect accuracy. This does **not** reflect real-world clinical performance. This project is for educational and portfolio purposes only.

---

## ⚕️ Disclaimer

This application is built for **educational and demonstration purposes only**. It is not a medical diagnostic tool. Always consult a qualified healthcare professional for any medical concerns.

---

## Authors
Muhammad Mudassar Saeed
Rimsha Kiran
