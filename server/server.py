############################################################
#  PhishGuard – Local ML Inference Server (Final Version)
############################################################

import os
import pickle
import importlib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

# Paths (relative to this file)
MODEL_PATH = "phishguard.h5"
PREP_PATH = "preprocessor.pkl"
LBL_PATH = "label_encoder.pkl"

# Ensure utils.py exists (Preprocessor class)
if not os.path.exists("utils.py"):
    raise FileNotFoundError("utils.py is required and must define Preprocessor class.")

utils = importlib.import_module("utils")
Preprocessor = getattr(utils, "Preprocessor")

print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading preprocessor...")
with open(PREP_PATH, "rb") as f:
    preprocessor = pickle.load(f)

label_encoder = None
if os.path.exists(LBL_PATH):
    with open(LBL_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print("Loaded label encoder.")
else:
    print("No label_encoder.pkl found — assuming binary/sigmoid model.")

app = Flask(__name__)
CORS(app)

def prepare_input(url: str):
    # The Preprocessor expects a DataFrame with columns ['url','label']
    df = pd.DataFrame([[url, None]], columns=["url", "label"])
    X = preprocessor.transform(df)
    return X

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if "url" not in data:
        return jsonify({"error": "Missing required field: url"}), 400
    url = data["url"]
    try:
        X = prepare_input(url)
        preds = model.predict(X)

        # Binary (sigmoid)
        if preds.shape[1] == 1:
            p = float(preds[0][0])
            # your training helper used `prediction[0][0] < 0.5` to indicate phishing
            label = "phishing" if p < 0.5 else "legit"
            confidence = (1 - p) if p < 0.5 else p
            probs = {"phishing": 1 - p, "legit": p}
        else:
            arr = preds[0]
            idx = int(np.argmax(arr))
            if label_encoder is not None:
                classes = label_encoder.inverse_transform(range(len(arr)))
            else:
                classes = [f"class_{i}" for i in range(len(arr))]
            label = classes[idx]
            confidence = float(arr[idx])
            probs = {classes[i]: float(arr[i]) for i in range(len(arr))}

        return jsonify({"url": url, "label": label, "confidence": confidence, "probs": probs})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "PhishGuard API is running"})

if __name__ == "__main__":
    print("PhishGuard API running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
