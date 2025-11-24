# PhishGuard: An On-Device Machine Learning System for Real-Time Phishing URL Classification.

PhishGuard is an end to end phishing URL detection system powered by a custom machine learning model. It includes:

* A TensorFlow phishing detection model
* A custom URL preprocessor (lexical feature extraction + sequence tokenizer)
* A local Flask API that serves predictions
* A Chrome/Edge browser extension for real-time phishing detection
* An admin dashboard inside the extension that stores detected phishing URLs

## Project Overview

PhishGuard analyzes URLs using:

* Tokenized URL sequences (top 5000 tokens, padded to length 100)
* Lexical features (URL length, digits count, special chars, TLD length, HTTPS flag, etc.)
* Deep learning classifier built with TensorFlow/Keras
* Local API inference for real-time predictions
* Browser extension communicating directly with your local server

This system runs fully offline, making it ideal for privacy-preserving deployments.

## Project Structure

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
    ├── manifest.json
    ├── popup.html
    ├── popup.js
    ├── service_worker.js
    ├── content_script.js
    ├── admin/
    │   ├── dashboard.html
    │   └── dashboard.js
    └── icons/
```

## Installation (Windows)

### 1. Install Python 3.10

TensorFlow 2.17 (Windows supported) requires Python 3.10.

Download: https://www.python.org/downloads/release/python-31012/

### 2. Clone the Repository

```bash
git clone https://github.com/nathanielshibadu/Phishing-Detection-System.git
cd PhishGuard
```

### 3. Set Up Virtual Environment

```bash
cd server
python3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

Required packages:

```
tensorflow==2.17.0
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.3.2
tldextract==5.3.0
flask==2.2.5
flask-cors==3.0.10
requests==2.31.0
protobuf==4.25.3
h5py==3.10.0
```

Install:

```bash
pip install -r requirements.txt
```

### 5. Add Model Artifacts

Place in `/server`:

* `phishguard.h5`
* `preprocessor.pkl`
* `label_encoder.pkl` (optional)
* `utils.py`
* `server.py`

### 6. Start the Local PhishGuard API

```bash
python server.py
```

You will see:

```
PhishGuard API running at http://127.0.0.1:5000
```

## API Usage

### POST /predict

**Request**

```json
{
  "url": "http://example.com"
}
```

**Response**

```json
{
  "url": "http://example.com",
  "label": "phishing",
  "confidence": 0.97,
  "probs": {
    "phishing": 0.97,
    "legit": 0.03
  }
}
```

## Browser Extension Setup

* Open Chrome or Edge
* Navigate to: `chrome://extensions/`
* Enable **Developer Mode**
* Click **Load Unpacked**
* Select the `extension` folder

You will now see the PhishGuard icon in your browser.

## Features of the Browser Extension

### Real-time URL scanning

Click **Scan Current Page** to classify any website.

### Admin Dashboard

A complete UI that stores all detected phishing URLs:

* URL
* Detection timestamp
* Confidence score
* Delete single entries
* Clear all logs

Stored using `chrome.storage.local`.

### Local API Communication

All detection happens on your machine — no external requests.

## Admin Dashboard Structure

```
extension/admin/
│
├── dashboard.html
└── dashboard.js
```

The dashboard loads dynamically and provides:

* A table of logged phishing URLs
* Buttons to clear or delete entries
* Automatic log updates

## Important Notes

### 1. TensorFlow Version

Your model was trained in Colab using TensorFlow 2.19, but Windows only supports TensorFlow up to 2.17.

The model loads successfully using TF 2.17 as long as:

* NumPy is below 2.0 (`numpy==1.26.4`)
* Scikit-learn is below 1.4 (`1.3.2` matches perfectly)

### 2. Preprocessor Compatibility

`utils.py` must remain identical to the version used during training so that your custom class unpickles correctly.

### 3. API Must Be Running

The extension depends on the local API. If the server is offline, you will see an "API unreachable" message.

## Future Enhancements

Optional improvements:

* Auto-scan page on visit
* Badge color indicators (red/yellow/green)
* API analytics dashboard
* Cloud-hosted version
* Export phishing logs to CSV
* Advanced ML model retraining pipeline
