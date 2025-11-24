# PhishGuard — Technical Report

**Date:** November 2025    
**Project:** End-to-End Phishing URL Detection System (PhishGuard)

## 1. Executive Summary

PhishGuard is an end-to-end phishing detection system integrating:

* A hybrid deep learning URL classifier (tokenized URLs + lexical features)
* A custom preprocessing pipeline serialized for inference
* A locally hosted Flask API for real-time classification
* A Chrome browser extension with an Admin Dashboard for logging phishing detections

The resulting system achieves high accuracy (~94–95%) with fast inference (<70 ms) and is fully deployable on Windows machines without external cloud dependencies.

## 2. Approach

### 2.1 Data Preprocessing

The preprocessing pipeline was designed to convert raw URLs into structured representations that combine sequence modeling with lexical metadata.

#### Tokenization

* Trained on URL corpus
* Retained top 5000 tokens
* Padded sequences to length 100

#### Lexical Features (Engineered)

Extracted URL characteristics known to correlate with phishing:

| Feature | Description |
|---------|-------------|
| URL length | Total characters |
| Digit count | Number of digits |
| Special characters | `. ?, -, _, &, %` etc. |
| Subdomain depth | Count of subdomain segments |
| TLD length | e.g., `com` = 3 |
| HTTPS flag | 1/0 |
| IP-address flag | Detects numeric hosts |
| Path length | Tokens after domain |
| Query parameters | Count of `?key=value` |

Lexical features were normalized using StandardScaler, stored inside `preprocessor.pkl`.

### 2.2 Train/Validation/Test Split

* 70% training
* 15% validation
* 15% testing
* Stratified to preserve phishing/legit ratio

## 3. Model Architecture

The classifier is a dual-input neural network:

### A. Sequence Branch

* Embedding (128 dims)
* Conv1D for feature extraction
* BiLSTM (64 hidden units)

### B. Lexical Feature Branch

* Dense layers with ReLU
* Dropout for regularization

The branches are concatenated and passed to:

* Dense → Sigmoid output
* Binary classification (phishing vs. legit)

### Training Configuration

* **Loss:** Binary Crossentropy
* **Optimizer:** Adam (lr=0.001)
* **Batch size:** 64
* **Early stopping** (patience=3)

## 4. Results Summary

| Metric | Score |
|--------|-------|
| Accuracy | 93–95% |
| F1 Score | ~92% |
| ROC-AUC | ~0.95 |
| Inference Time | ~50–70 ms |

### 4.2 Observations

**Strengths:**

* High detection accuracy
* Fast enough for real-time browser scanning
* Hybrid architecture improves robustness
* Strong recall minimizes missed phishing URLs

**Weaknesses:**

* Rare, exotic phishing formats less represented
* Some benign but complicated URLs can trigger false positives

## 5. Challenges Faced

### 5.1 Dependency Mismatch (Linux → Windows)

Training occurred in Google Colab using:

* TensorFlow 2.19
* NumPy 2.x
* Scikit-learn 1.6.x

Windows only supports:

* TensorFlow 2.17
* NumPy 1.26.x
* Scikit-learn 1.3.2

**Resolution:** Downgraded Windows dependencies while ensuring the pipeline and model remained fully compatible.

### 5.2 Preprocessor Serialization Challenges

Because the preprocessor contains:

* Tokenizer
* StandardScaler
* Feature functions
* Column mapping

…it had to be reconstructed exactly during inference.

**Solution:** Store full pipeline inside `preprocessor.pkl` and load safely via `utils.py`.

### 5.3 Browser Extension Integration

Key challenges included:

* Chrome Manifest V3 limitations
* Cross-origin requests to localhost API
* Managing asynchronous predictions
* Designing persistent phishing logs

**Solution:**

* `chrome.storage.local` for log persistence
* Robust popup → API → dashboard flow
* CORS-enabled Flask API

## 6. System Architecture

### 6.1 Local API (Flask)

**Endpoint:** `POST /predict`

**Request:**

```json
{
  "url": "https://example.com/login"
}
```

**Response:**

```json
{
  "label": "phishing",
  "confidence": 0.982,
  "probs": {
    "phishing": 0.982,
    "legit": 0.018
  }
}
```

**Artifacts loaded at startup:**

* `phishguard.h5` (model)
* `preprocessor.pkl` (tokenizer + scaler + pipeline)
* `label_encoder.pkl`

### 6.2 Browser Extension

**Features:**

* Extracts current tab URL
* Sends to Flask API for prediction
* Displays confidence and label
* Logs phishing URLs to dashboard
* User-managed log deletion & clearing

**Dashboard Fields:**

* URL
* Confidence
* Timestamp
* Delete button

## 7. Production Improvements

### 7.1 Short-Term (1–2 Weeks)

* Add automatic scanning on page load
* Badge color indicators (safe/warning/danger)
* Model threshold optimization

### 7.2 Medium-Term (1–2 Months)

* Expand training dataset with new phishing sources
* Add entropy, randomness, or character diversity features
* Switch to FastAPI for async backend

### 7.3 Long-Term (3–6 Months)

* Deploy with TensorFlow Serving
* Add central threat intelligence aggregation
* On-device WASM inference for browser-only deployment

## 8. Conclusion

PhishGuard successfully demonstrates a complete ML cybersecurity pipeline:

* End-to-end preprocessing
* Hybrid deep-learning model
* Local inference API
* Browser extension with real-time detection
* Integrated admin dashboard
* Fully deployable and privacy-preserving

This system meets the requirements of a practical phishing detection solution while showcasing skills in:

* ML engineering
* Full-stack ML deployment
* Browser extension development
* Real-world cybersecurity application design

The project is ready for extension and production adaptation.

## 9. Appendix

### Deliverables Checklist

* Preprocessing pipeline (`utils.py`)
* Model training artifacts (`phishguard.h5`)
* Preprocessor serialization (`preprocessor.pkl`)
* Local Flask API (`server.py`)
* Browser extension with popup + dashboard
* Documentation (`README.md`, `REPORT.md`)
* Requirements (`requirements.txt`)

### Repository Structure

```
PhishGuard/
│
├── server/
│   ├── server.py
│   ├── utils.py
│   ├── preprocessor.pkl
│   ├── phishguard.h5
│   ├── label_encoder.pkl
│   └── venv/
│
└── extension/
    ├── popup.html
    ├── popup.js
    ├── manifest.json
    ├── service_worker.js
    ├── content_script.js
    ├── admin/
    │   ├── dashboard.html
    │   └── dashboard.js
    └── icons/
```

### Computational Environment

**Training:**

* Google Colab (Linux, Python 3.12)
* TensorFlow 2.19
* NumPy 2.0
* Scikit-learn 1.6.1

**Serving (Windows):**

* Python 3.10
* TensorFlow 2.17
* NumPy 1.26.4
* Scikit-learn 1.3.2