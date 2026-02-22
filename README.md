# 🩺 Disease Prediction System

A machine learning web application that predicts diseases based on user-selected symptoms. Built with Python, scikit-learn, and Flask... featuring an interactive symptom picker UI with live search, confidence scores, and a Top 5 differential diagnosis.

> ⚠️ **Model files are not included in this repository.** You must run the preprocessing and training scripts once before launching the app. Instructions are below.

---

## ✨ Features

- **Smart Symptom Search** — Searchable, tag-based multi-select input with live filtering and keyword highlighting
- **Instant Prediction** — Random Forest model predicts the most likely disease from your symptom selection
- **Confidence Score** — Visual progress bar shows model confidence in the prediction
- **Top 5 Differential Diagnosis** — Ranked list of the 5 most probable diseases with probability bars
- **REST API** — Clean JSON endpoints for integration with other tools
- **CLI Mode** — Standalone terminal-based prediction script

---

## 🗂️ Project Structure

```
general-disease-prediction-system/
│
├── app.py                         # Flask web app & embedded HTML/CSS/JS UI
│
├── src/
│   ├── preprocess.py              # Data cleaning, encoding, feature extraction
│   ├── train_model.py             # Model training, evaluation & cross-validation
│   └── predict.py                 # CLI-based prediction script
│
├── data/
│   └── raw/
│       └── dataset.csv            # ← Place your raw dataset here
│
├── models/                        # ← Auto-generated after training (not in repo)
│   ├── disease_model.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
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
| Frontend | Vanilla HTML/CSS/JS |

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mudassar-saeed/general-disease-prediction-system.git
cd general-disease-prediction-system
```

### 2. Create a Virtual Environment

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

### 4. Add the Dataset

Place your raw dataset file at:

```
data/raw/dataset.csv
```

The dataset should have a `Disease` column and multiple `Symptom_1`, `Symptom_2`, ... columns.
You can download a compatible dataset from [Kaggle — Disease Symptom Prediction](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset).

### 5. Generate the Model ⚠️ Required — do this once before running the app

```bash
# Step 1 — Clean and encode the raw data
python src/preprocess.py

# Step 2 — Train the Random Forest model and save artifacts
python src/train_model.py
```

This will automatically create:

```
data/processed/processed_data.csv
models/disease_model.pkl
models/label_encoder.pkl
models/feature_columns.pkl
```

### 6. Run the Web App

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## 💻 CLI Prediction (No Browser Needed)

```bash
python src/predict.py
```

Enter symptoms separated by commas when prompted:

```
Symptoms: itching, skin_rash, nodal_skin_eruptions
```

Type `list` to see all available symptom names.

---

## 🌐 API Reference

### `GET /symptoms`
Returns all valid symptom names recognized by the model.

**Response:**
```json
{
  "symptoms": ["abdominal_pain", "acidity", "anxiety", "..."]
}
```

### `POST /predict`
Predicts disease from a comma-separated symptom string.

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
    { "disease": "Chicken pox",      "probability": 1.2 }
  ]
}
```

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Trees | 200 |
| Class Weighting | Balanced |
| Train/Test Split | 80% / 20% (Stratified) |
| Cross-Validation | 5-Fold Stratified CV |
| Features | 130+ binary one-hot encoded symptoms |

> ⚠️ **Note:** The training dataset is synthetic and perfectly balanced, which produces near-perfect accuracy. This does **not** reflect real-world clinical performance. This project is for educational and portfolio purposes only.

---

## ⚕️ Disclaimer

This application is for **educational and demonstration purposes only**. It is not a medical diagnostic tool. Always consult a qualified healthcare professional for any health concerns.

---

## 👨‍💻 Authors

| Name | GitHub |
|---|---|
| Muhammad Mudassar Saeed | [@mudassar-saeed](https://github.com/mudassar-saeed) |
| Rimsha Kiran | [@RIMSHA-KIRAN](https://github.com/RIMSHA-KIRAN) |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
