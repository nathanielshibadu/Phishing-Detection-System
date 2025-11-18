# server_side_api/app.py
import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from database import db, URLStat
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from urllib.parse import urlparse
import requests

# CONFIG
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'phishguard.h5')
PREPROC_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'preprocessor.pkl')
DB_PATH = os.path.join(os.path.dirname(__file__), 'instance', 'database.db')

# create app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_PATH}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app, resources={r"/api/*": {"origins": "*"}})  # restrict origins in production

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phishguard-api")

# load model + preprocessor once
logger.info("Loading model and preprocessor...")
model = load_model(MODEL_PATH)
with open(PREPROC_PATH, 'rb') as f:
    preprocessor = pickle.load(f)
logger.info("Model loaded.")

# init DB and admin
db.init_app(app)
admin = Admin(app, template_mode='bootstrap3')
admin.add_view(ModelView(URLStat, db.session))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return jsonify({"status": "PhishGuard API running"})

def is_phishing(url):
    # quick validate
    if not url or not isinstance(url, str):
        raise ValueError("Invalid URL")
    # ignore unsupported schemes
    scheme = urlparse(url).scheme
    if scheme in ('chrome', 'about', 'file', 'data', 'javascript'):
        return None  # ignored

    df = pd.DataFrame([[url, None]], columns=['url', 'label'])
    X = preprocessor.transform(df)
    prob = float(model.predict(X)[0][0])
    label = "phishing" if prob < 0.5 else "legitimate"
    return {"url": url, "label": label, "probability": prob}

@app.route('/api/check_url', methods=['POST'])
def check_url():
    payload = request.get_json(force=True)
    url = payload.get('url')
    ip_address = request.remote_addr
    try:
        result = is_phishing(url)
        if result is None:
            return jsonify({'message': 'Ignored URL'}), 200

        # store stat
        url_stat = URLStat(url=url, is_phishing=1 if result['label']=='phishing' else 0, ip_address=ip_address)
        db.session.add(url_stat)
        db.session.commit()

        return jsonify(result)
    except Exception as e:
        logger.exception("Error checking URL")
        return jsonify({"error": str(e)}), 500

# VirusTotal endpoint (optional)
@app.route('/api/check_url_virustotal', methods=['POST'])
def check_url_virustotal():
    api_key = os.environ.get('VT_API_KEY')
    if not api_key:
        return jsonify({'error': 'VirusTotal API key not set'}), 500

    url_to_check = request.json.get('url')
    if not url_to_check:
        return jsonify({'error':'Missing url'}), 400

    # Use VirusTotal URL submission/lookup - note: real flow requires URL to be encoded and endpoint may differ
    endpoint = "https://www.virustotal.com/api/v3/urls"
    headers = {"x-apikey": api_key}
    resp = requests.post(endpoint, data={'url': url_to_check}, headers=headers)
    return jsonify(resp.json()), resp.status_code

@app.route('/dashboard')
def dashboard():
    data = URLStat.query.all()
    chart_data = convert_to_chartjs_format(data)
    return render_template('dashboard.html', data=chart_data)

# helper functions: convert_to_chartjs_format & generate_random_color
# (You can reuse the detailed implementation from your surfhound app)
# ... include the convert_to_chartjs_format & generate_random_color functions here ...

if __name__ == '__main__':
    # For local dev only. In production use Gunicorn/Uvicorn.
    app.run(host='0.0.0.0', port=5000, debug=True)
