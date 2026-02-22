import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# --------------------------------------------------
# Load Model Artifacts
# --------------------------------------------------

model = joblib.load("models/disease_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

clean_feature_columns = [col.strip().lower() for col in feature_columns]
feature_map = dict(zip(clean_feature_columns, feature_columns))

# --------------------------------------------------
# Flask App
# --------------------------------------------------

app = Flask(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Disease Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --ink:       #0f1923;
      --ink-soft:  #4a5568;
      --ink-muted: #94a3b8;
      --bg:        #f7f5f0;
      --surface:   #ffffff;
      --rule:      #e8e3db;
      --accent:    #1a6b4a;
      --accent-lt: #e8f5ee;
      --accent-dk: #0d4a32;
      --warn:      #c0392b;
      --warn-lt:   #fdf0ee;
      --tag-bg:    #eef6f2;
      --tag-text:  #1a6b4a;
      --radius:    10px;
      --shadow:    0 2px 24px rgba(15,25,35,0.08);
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'DM Sans', sans-serif;
      background: var(--bg);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 32px 16px;
      color: var(--ink);
    }

    /* ── Card ── */
    .card {
      background: var(--surface);
      border-radius: 18px;
      box-shadow: var(--shadow);
      width: 100%;
      max-width: 680px;
      overflow: hidden;
    }

    .card-header {
      padding: 32px 36px 28px;
      border-bottom: 1px solid var(--rule);
      position: relative;
    }
    .card-header::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 4px;
      background: linear-gradient(90deg, var(--accent) 0%, #2ecc71 100%);
    }

    h1 {
      font-family: 'DM Serif Display', serif;
      font-size: 2rem;
      letter-spacing: -0.5px;
      color: var(--ink);
      line-height: 1.1;
    }
    .sub {
      color: var(--ink-soft);
      font-size: 0.88rem;
      margin-top: 6px;
      font-weight: 400;
    }

    .card-body { padding: 28px 36px 32px; }

    /* ── Field label ── */
    .field-label {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 10px;
    }
    .field-label span {
      font-weight: 600;
      font-size: 0.88rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--ink-soft);
    }
    .count-badge {
      font-size: 0.78rem;
      font-weight: 500;
      color: var(--accent);
      background: var(--accent-lt);
      padding: 2px 10px;
      border-radius: 99px;
      transition: opacity .2s;
    }

    /* ── Multi-select container ── */
    .ms-wrap {
      border: 1.5px solid var(--rule);
      border-radius: var(--radius);
      background: #fff;
      transition: border-color .2s, box-shadow .2s;
      position: relative;
    }
    .ms-wrap:focus-within {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(26,107,74,.10);
    }

    /* selected tags + search input row */
    .ms-input-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 10px 12px;
      min-height: 50px;
      cursor: text;
    }

    .sel-tag {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: var(--tag-bg);
      color: var(--tag-text);
      font-size: 0.8rem;
      font-weight: 500;
      padding: 4px 10px 4px 12px;
      border-radius: 99px;
      white-space: nowrap;
      animation: pop .15s ease;
    }
    @keyframes pop {
      from { transform: scale(.85); opacity: 0; }
      to   { transform: scale(1);   opacity: 1; }
    }
    .sel-tag button {
      background: none; border: none;
      cursor: pointer;
      color: var(--accent);
      font-size: 1rem;
      line-height: 1;
      padding: 0;
      display: flex; align-items: center;
      opacity: .6;
      transition: opacity .15s;
    }
    .sel-tag button:hover { opacity: 1; }

    /* search box */
    #searchInput {
      border: none; outline: none;
      font-family: 'DM Sans', sans-serif;
      font-size: 0.92rem;
      color: var(--ink);
      background: transparent;
      flex: 1;
      min-width: 140px;
      padding: 4px 2px;
    }
    #searchInput::placeholder { color: var(--ink-muted); }

    /* dropdown list */
    .ms-dropdown {
      border-top: 1px solid var(--rule);
      max-height: 0;
      overflow: hidden;
      transition: max-height .25s ease;
    }
    .ms-dropdown.open {
      max-height: 260px;
      overflow-y: auto;
    }
    .ms-dropdown::-webkit-scrollbar { width: 5px; }
    .ms-dropdown::-webkit-scrollbar-thumb { background: var(--rule); border-radius: 99px; }

    .ms-no-results {
      padding: 14px 16px;
      font-size: 0.85rem;
      color: var(--ink-muted);
      display: none;
    }

    .ms-option {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 16px;
      font-size: 0.88rem;
      color: var(--ink);
      cursor: pointer;
      transition: background .12s;
      user-select: none;
    }
    .ms-option:hover  { background: #f8faf9; }
    .ms-option.active { background: var(--accent-lt); color: var(--accent-dk); }

    /* custom checkbox */
    .ms-check {
      width: 16px; height: 16px;
      border-radius: 4px;
      border: 1.5px solid var(--rule);
      flex-shrink: 0;
      display: grid; place-items: center;
      transition: background .12s, border-color .12s;
    }
    .ms-option.active .ms-check {
      background: var(--accent);
      border-color: var(--accent);
    }
    .ms-option.active .ms-check::after {
      content: '';
      width: 5px; height: 9px;
      border: 2px solid #fff;
      border-top: none; border-left: none;
      transform: rotate(45deg) translate(-1px, -1px);
      display: block;
    }

    .ms-option-text { flex: 1; }
    .ms-option-text em {
      font-style: normal;
      background: #fff3cd;
      border-radius: 2px;
    }



    /* ── Clear all button (label row) ── */
    .clear-all-btn {
      font-family: 'DM Sans', sans-serif;
      font-size: 0.78rem;
      font-weight: 500;
      color: var(--ink-muted);
      background: none;
      border: none;
      cursor: pointer;
      padding: 0;
      transition: color .15s, opacity .2s;
    }
    .clear-all-btn:hover { color: var(--warn); }

    /* ── Predict button ── */
    .predict-btn {
      margin-top: 18px;
      width: 100%;
      padding: 14px;
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: var(--radius);
      font-family: 'DM Sans', sans-serif;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      letter-spacing: 0.02em;
      transition: background .2s, transform .1s, box-shadow .2s;
      box-shadow: 0 4px 14px rgba(26,107,74,.25);
    }
    .predict-btn:hover   { background: var(--accent-dk); box-shadow: 0 6px 20px rgba(26,107,74,.35); }
    .predict-btn:active  { transform: scale(.99); }
    .predict-btn:disabled { background: var(--ink-muted); box-shadow: none; cursor: not-allowed; }

    /* ── Error ── */
    .error-box {
      margin-top: 16px;
      background: var(--warn-lt);
      border: 1px solid #f5c6c3;
      border-radius: var(--radius);
      padding: 13px 16px;
      color: var(--warn);
      font-size: 0.88rem;
      display: none;
    }

    /* ── Result ── */
    .result { display: none; margin-top: 28px; }

    .result-hero {
      background: linear-gradient(135deg, #f0fdf6 0%, #e8f5ee 100%);
      border: 1px solid #c3e6d0;
      border-radius: var(--radius);
      padding: 22px 24px;
      margin-bottom: 16px;
    }
    .result-label {
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: .1em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 4px;
    }
    .result-disease {
      font-family: 'DM Serif Display', serif;
      font-size: 1.7rem;
      color: var(--accent-dk);
      line-height: 1.15;
    }

    .conf-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
    }
    .conf-track {
      flex: 1;
      height: 6px;
      background: #c3e6d0;
      border-radius: 99px;
      overflow: hidden;
    }
    .conf-fill {
      height: 100%;
      background: var(--accent);
      border-radius: 99px;
      transition: width .7s cubic-bezier(.16,1,.3,1);
    }
    .conf-pct {
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--accent-dk);
      width: 44px;
      text-align: right;
    }

    /* recognised tags */
    .rec-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
    .rec-tag {
      font-size: 0.75rem;
      background: rgba(26,107,74,.1);
      color: var(--accent-dk);
      border-radius: 99px;
      padding: 3px 10px;
      font-weight: 500;
    }

    /* top 5 table */
    .top5-box {
      background: #fafaf8;
      border: 1px solid var(--rule);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .top5-head {
      padding: 14px 20px 10px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: .1em;
      text-transform: uppercase;
      color: var(--ink-muted);
      border-bottom: 1px solid var(--rule);
    }
    .top5-row {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 11px 20px;
      border-bottom: 1px solid var(--rule);
      transition: background .12s;
    }
    .top5-row:last-child { border-bottom: none; }
    .top5-row:hover { background: #f5f3ee; }
    .top5-rank {
      font-size: 0.75rem;
      color: var(--ink-muted);
      font-weight: 600;
      width: 18px;
    }
    .top5-name { flex: 1; font-size: 0.88rem; color: var(--ink); }
    .top5-track {
      width: 120px;
      height: 5px;
      background: var(--rule);
      border-radius: 99px;
      overflow: hidden;
    }
    .top5-fill {
      height: 100%;
      background: var(--accent);
      border-radius: 99px;
      transition: width .7s cubic-bezier(.16,1,.3,1);
      opacity: .7;
    }
    .top5-pct {
      width: 44px;
      text-align: right;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--ink-soft);
    }

    .disclaimer {
      text-align: center;
      font-size: 0.76rem;
      color: var(--ink-muted);
      margin-top: 20px;
      line-height: 1.6;
    }

    /* ── Responsive ── */
    @media (max-width: 480px) {
      .card-header, .card-body { padding-left: 22px; padding-right: 22px; }
      h1 { font-size: 1.6rem; }
      .top5-track { width: 70px; }
    }
  </style>
</head>
<body>

<div class="card">
  <div class="card-header">
    <h1>Disease Predictor</h1>
    <p class="sub">Select your symptoms below to receive a differential diagnosis.</p>
  </div>

  <div class="card-body">

    <!-- Field label row -->
    <div class="field-label">
      <span>Symptoms</span>
      <div style="display:flex;align-items:center;gap:8px;">
        <span class="count-badge" id="countBadge" style="opacity:0">0 selected</span>
        <button class="clear-all-btn" id="clearAllBtn" style="opacity:0" onmousedown="event.preventDefault(); clearAll()">Clear all</button>
      </div>
    </div>

    <!-- Multi-select widget -->
    <div class="ms-wrap" id="msWrap">

      <!-- Tags + search -->
      <div class="ms-input-row" id="inputRow" onclick="focusSearch()">
        <input
          id="searchInput"
          type="text"
          placeholder="Search symptoms…"
          autocomplete="off"
          oninput="filterOptions()"
          onfocus="openDropdown()"
          onkeydown="handleKeydown(event)"
        />
      </div>

      <!-- Dropdown list -->
      <div class="ms-dropdown" id="dropdown">
        <div class="ms-no-results" id="noResults">No symptoms match your search.</div>
        <div id="optionsList"></div>

      </div>

    </div>

    <button class="predict-btn" id="predictBtn" onclick="predict()">
      Predict Disease
    </button>

    <div class="error-box" id="errorBox"></div>

    <!-- Result -->
    <div class="result" id="result">

      <div class="result-hero">
        <div class="result-label">Predicted Diagnosis</div>
        <div class="result-disease" id="rDisease">—</div>
        <div class="conf-row">
          <div class="conf-track">
            <div class="conf-fill" id="confFill" style="width:0%"></div>
          </div>
          <div class="conf-pct" id="confPct">—</div>
        </div>
        <div class="rec-tags" id="recTags"></div>
      </div>

      <div class="top5-box">
        <div class="top5-head">Differential — Top 5</div>
        <div id="top5Rows"></div>
      </div>

      <p class="disclaimer">
        ⚕ For educational purposes only.<br>
        Always consult a qualified medical professional for diagnosis and treatment.
      </p>
    </div>

  </div>
</div>

<script>
// ── State ────────────────────────────────────────────────
let ALL_SYMPTOMS = [];
let selected = new Set();

// ── Bootstrap ────────────────────────────────────────────
fetch('/symptoms')
  .then(r => r.json())
  .then(d => {
    ALL_SYMPTOMS = d.symptoms;
    renderOptions(ALL_SYMPTOMS);
  });

// ── Dropdown open/close ──────────────────────────────────
function openDropdown() {
  document.getElementById('dropdown').classList.add('open');
  document.addEventListener('click', outsideClick);
}

function closeDropdown() {
  document.getElementById('dropdown').classList.remove('open');
  document.removeEventListener('click', outsideClick);
}

function outsideClick(e) {
  if (!document.getElementById('msWrap').contains(e.target)) closeDropdown();
}

function focusSearch() {
  document.getElementById('searchInput').focus();
}

// ── Render option list ───────────────────────────────────
function renderOptions(symptoms) {
  const query = document.getElementById('searchInput').value.toLowerCase();
  const container = document.getElementById('optionsList');

  container.innerHTML = symptoms.map(s => {
    const active = selected.has(s) ? 'active' : '';
    const label  = highlight(s, query);
    // Use mousedown + preventDefault so the input never loses focus,
    // which prevents the outsideClick handler from firing and closing the dropdown.
    return `
      <div class="ms-option ${active}" data-value="${s}"
           onmousedown="event.preventDefault(); toggleSymptom('${s}')">
        <div class="ms-check"></div>
        <div class="ms-option-text">${label}</div>
      </div>`;
  }).join('');

  document.getElementById('noResults').style.display =
    symptoms.length === 0 ? 'block' : 'none';
}

function highlight(text, query) {
  if (!query) return text;
  const idx = text.toLowerCase().indexOf(query);
  if (idx === -1) return text;
  return text.slice(0, idx)
    + '<em>' + text.slice(idx, idx + query.length) + '</em>'
    + text.slice(idx + query.length);
}

// ── Filter ───────────────────────────────────────────────
function filterOptions() {
  const q = document.getElementById('searchInput').value.toLowerCase();
  const filtered = ALL_SYMPTOMS.filter(s => s.includes(q));
  renderOptions(filtered);
  openDropdown();
}

// ── Toggle selection ─────────────────────────────────────
function toggleSymptom(s) {
  if (selected.has(s)) {
    selected.delete(s);
  } else {
    selected.add(s);
  }
  // Clear the search text after selecting so the user sees the full list again
  document.getElementById('searchInput').value = '';
  syncTags();
  renderOptions(ALL_SYMPTOMS);
}

function removeTag(s) {
  selected.delete(s);
  syncTags();
  renderOptions(ALL_SYMPTOMS);
}

function clearAll() {
  selected.clear();
  document.getElementById('searchInput').value = '';
  syncTags();
  renderOptions(ALL_SYMPTOMS);
}

function getFiltered() {
  const q = document.getElementById('searchInput').value.toLowerCase();
  return q ? ALL_SYMPTOMS.filter(s => s.includes(q)) : ALL_SYMPTOMS;
}

// ── Sync tags in input row ───────────────────────────────
function syncTags() {
  const row   = document.getElementById('inputRow');
  const badge = document.getElementById('countBadge');
  const input = document.getElementById('searchInput');

  // Remove existing tags (keep input)
  row.querySelectorAll('.sel-tag').forEach(el => el.remove());

  selected.forEach(s => {
    const tag = document.createElement('div');
    tag.className = 'sel-tag';
    tag.innerHTML = `${s} <button onclick="removeTag('${s}');event.stopPropagation()">×</button>`;
    row.insertBefore(tag, input);
  });

  const n = selected.size;
  badge.textContent = n === 0 ? '0 selected' : `${n} selected`;
  badge.style.opacity = n === 0 ? '0' : '1';
  const clearBtn = document.getElementById('clearAllBtn');
  clearBtn.style.opacity = n === 0 ? '0' : '1';
  clearBtn.style.pointerEvents = n === 0 ? 'none' : 'auto';

  // Update placeholder
  input.placeholder = n === 0 ? 'Search symptoms…' : 'Add more…';

  // Hide result + error when selection changes
  document.getElementById('result').style.display = 'none';
  document.getElementById('errorBox').style.display = 'none';
}

// ── Keyboard: backspace deletes last tag ────────────────
function handleKeydown(e) {
  if (e.key === 'Backspace' && e.target.value === '' && selected.size > 0) {
    // Remove the last added symptom
    const arr = [...selected];
    removeTag(arr[arr.length - 1]);
  }
}

// ── Predict ──────────────────────────────────────────────
async function predict() {
  const btn    = document.getElementById('predictBtn');
  const errBox = document.getElementById('errorBox');
  const result = document.getElementById('result');

  errBox.style.display = 'none';
  result.style.display = 'none';

  if (selected.size === 0) {
    showError('Please select at least one symptom.');
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Analysing…';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symptoms: [...selected].join(', ') }),
    });
    const data = await res.json();

    if (data.error) { showError(data.error); return; }

    // Hero
    document.getElementById('rDisease').textContent = data.predicted_disease;
    document.getElementById('confFill').style.width  = data.confidence + '%';
    document.getElementById('confPct').textContent   = data.confidence.toFixed(1) + '%';

    // Tags
    document.getElementById('recTags').innerHTML =
      data.recognized_symptoms.map(s =>
        `<span class="rec-tag">${s.replace(/_/g,' ')}</span>`
      ).join('');

    // Top 5
    document.getElementById('top5Rows').innerHTML = data.top5.map((item, i) => `
      <div class="top5-row">
        <div class="top5-rank">${i + 1}</div>
        <div class="top5-name">${item.disease}</div>
        <div class="top5-track">
          <div class="top5-fill" style="width:${item.probability}%"></div>
        </div>
        <div class="top5-pct">${item.probability.toFixed(1)}%</div>
      </div>`).join('');

    result.style.display = 'block';
    result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch(e) {
    showError('Request failed. Please check the server is running.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Predict Disease';
  }
}

function showError(msg) {
  const b = document.getElementById('errorBox');
  b.textContent = msg;
  b.style.display = 'block';
}
</script>
</body>
</html>"""


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/symptoms", methods=["GET"])
def list_symptoms():
    return jsonify({"symptoms": sorted(clean_feature_columns)})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    user_input = data.get("symptoms", "").strip()

    if not user_input:
        return jsonify({"error": "No symptoms provided."}), 400

    input_symptoms = [
        sym.strip().lower().replace(" ", "_")
        for sym in user_input.split(",")
        if sym.strip()
    ]

    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
    recognized, unrecognized = [], []

    for symptom in input_symptoms:
        if symptom in feature_map:
            input_data.loc[0, feature_map[symptom]] = 1
            recognized.append(symptom)
        else:
            unrecognized.append(symptom)

    if not recognized:
        return jsonify({
            "error": f"No valid symptoms recognized. Unrecognized: {', '.join(unrecognized)}"
        }), 400

    prediction       = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]
    predicted_disease = label_encoder.inverse_transform(prediction)[0]
    confidence        = float(np.max(prediction_proba) * 100)

    top5_indices = np.argsort(prediction_proba)[::-1][:5]
    top5 = [
        {
            "disease":     label_encoder.inverse_transform([int(i)])[0],
            "probability": float(prediction_proba[i] * 100),
        }
        for i in top5_indices
    ]

    return jsonify({
        "predicted_disease":     predicted_disease,
        "confidence":            confidence,
        "recognized_symptoms":   recognized,
        "unrecognized_symptoms": unrecognized,
        "top5":                  top5,
    })


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    print("Starting Disease Prediction Web App...")
    print("Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=5000)