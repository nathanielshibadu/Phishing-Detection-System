from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import pickle
import pandas as pd
import numpy as np
import sys
import os

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

app = Flask(__name__)
CORS(app)  # Enable CORS for browser extension

# Load models at startup
print("Loading models...")
model = keras.models.load_model('model/phishguard.h5')

with open('model/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("Models loaded successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'PhishGuard API is running'
    })

@app.route('/check-url', methods=['POST'])
def check_url():
    """
    Check if a URL is phishing or legitimate
    
    Expected JSON input:
    {
        "url": "https://example.com"
    }
    """
    try:
        # Get URL from request
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({
                'error': 'No URL provided',
                'message': 'Please send a JSON object with a "url" field'
            }), 400
        
        url = data['url']
        
        if not url:
            return jsonify({
                'error': 'Empty URL',
                'message': 'URL cannot be empty'
            }), 400
        
        # Prepare URL for prediction
        df_test = pd.DataFrame([[url, None]], columns=['url', 'label'])
        
        # Preprocess URL
        X_test = preprocessor.transform(df_test)
        
        # Make prediction
        prediction_prob = model.predict(X_test, verbose=0)[0][0]
        
        # Determine result (assuming 0=bad/phishing, 1=good/legitimate based on your training)
        is_phishing = prediction_prob < 0.5
        confidence = float(1 - prediction_prob if is_phishing else prediction_prob)
        
        result = {
            'url': url,
            'is_phishing': bool(is_phishing),
            'prediction': 'phishing' if is_phishing else 'legitimate',
            'confidence': round(confidence * 100, 2),
            'risk_score': round(float(prediction_prob), 4)
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/batch-check', methods=['POST'])
def batch_check():
    """
    Check multiple URLs at once
    
    Expected JSON input:
    {
        "urls": ["https://example1.com", "https://example2.com"]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'urls' not in data:
            return jsonify({
                'error': 'No URLs provided',
                'message': 'Please send a JSON object with a "urls" array'
            }), 400
        
        urls = data['urls']
        
        if not isinstance(urls, list):
            return jsonify({
                'error': 'Invalid format',
                'message': 'URLs must be provided as an array'
            }), 400
        
        results = []
        for url in urls:
            if not url:
                continue
                
            df_test = pd.DataFrame([[url, None]], columns=['url', 'label'])
            X_test = preprocessor.transform(df_test)
            prediction_prob = model.predict(X_test, verbose=0)[0][0]
            
            is_phishing = prediction_prob < 0.5
            confidence = float(1 - prediction_prob if is_phishing else prediction_prob)
            
            results.append({
                'url': url,
                'is_phishing': bool(is_phishing),
                'prediction': 'phishing' if is_phishing else 'legitimate',
                'confidence': round(confidence * 100, 2),
                'risk_score': round(float(prediction_prob), 4)
            })
        
        return jsonify({
            'results': results,
            'total_checked': len(results)
        })
    
    except Exception as e:
        print(f"Error processing batch request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Run on localhost, port 5000
    print("\n" + "="*50)
    print("PhishGuard API Server Starting...")
    print("Server will be available at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=True)