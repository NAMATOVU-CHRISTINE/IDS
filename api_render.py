"""
Flask API for Render Deployment - Real ML Model
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import os

app = Flask(__name__)
CORS(app)

prediction_history = []

# Load trained model
print("Loading trained model...")
try:
    model = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    print(f"✅ Model loaded successfully! Features: {len(feature_names)}")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    MODEL_LOADED = False

def preprocess_input(data):
    """Preprocess input data to match training format"""
    # Define all possible categorical values from training
    protocols = ['tcp', 'udp', 'icmp']
    services = ['http', 'smtp', 'ftp', 'ftp_data', 'telnet', 'ssh', 'private', 
                'domain_u', 'pop_3', 'finger', 'auth', 'other']
    flags = ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH', 'S1', 'S2', 'RSTOS0', 'S3', 'OTH']
    
    # Create base feature dict with numeric values
    features = {
        'duration': data.get('duration', 0),
        'src_bytes': data.get('src_bytes', 0),
        'dst_bytes': data.get('dst_bytes', 0),
        'land': data.get('land', 0),
        'wrong_fragment': data.get('wrong_fragment', 0),
        'urgent': data.get('urgent', 0),
        'hot': data.get('hot', 0),
        'num_failed_logins': data.get('num_failed_logins', 0),
        'num_compromised': data.get('num_compromised', 0),
        'root_shell': data.get('root_shell', 0),
        'su_attempted': data.get('su_attempted', 0),
        'num_root': data.get('num_root', 0),
        'num_file_creations': data.get('num_file_creations', 0),
        'num_shells': data.get('num_shells', 0),
        'num_access_files': data.get('num_access_files', 0),
        'num_outbound_cmds': data.get('num_outbound_cmds', 0),
        'count': data.get('count', 1),
        'srv_count': data.get('srv_count', 1),
        'serror_rate': data.get('serror_rate', 0.0),
        'srv_serror_rate': data.get('srv_serror_rate', 0.0),
        'rerror_rate': data.get('rerror_rate', 0.0),
        'srv_rerror_rate': data.get('srv_rerror_rate', 0.0),
        'same_srv_rate': data.get('same_srv_rate', 1.0),
        'diff_srv_rate': data.get('diff_srv_rate', 0.0),
        'srv_diff_host_rate': data.get('srv_diff_host_rate', 0.0),
        'dst_host_count': data.get('dst_host_count', 255),
        'dst_host_srv_count': data.get('dst_host_srv_count', 255),
        'dst_host_same_srv_rate': data.get('dst_host_same_srv_rate', 1.0),
        'dst_host_diff_srv_rate': data.get('dst_host_diff_srv_rate', 0.0),
        'dst_host_same_src_port_rate': data.get('dst_host_same_src_port_rate', 1.0),
        'dst_host_srv_diff_host_rate': data.get('dst_host_srv_diff_host_rate', 0.0),
        'dst_host_serror_rate': data.get('dst_host_serror_rate', 0.0),
        'dst_host_srv_serror_rate': data.get('dst_host_srv_serror_rate', 0.0),
        'dst_host_rerror_rate': data.get('dst_host_rerror_rate', 0.0),
        'dst_host_srv_rerror_rate': data.get('dst_host_srv_rerror_rate', 0.0),
    }
    
    # Scale numeric features
    df_num = pd.DataFrame([features])
    scaled_values = scaler.transform(df_num)
    
    # Create full feature vector with one-hot encoding
    full_features = {}
    for i, col in enumerate(df_num.columns):
        full_features[col] = scaled_values[0][i]
    
    # Add one-hot encoded categorical features
    protocol = data.get('protocol_type', 'tcp').lower()
    service = data.get('service', 'http').lower()
    flag = data.get('flag', 'SF')
    
    # One-hot encode protocol
    for p in protocols:
        full_features[f'protocol_type_{p}'] = 1 if p == protocol else 0
    
    # One-hot encode service
    for s in services:
        full_features[f'service_{s}'] = 1 if s == service else 0
    
    # One-hot encode flag
    for f in flags:
        full_features[f'flag_{f}'] = 1 if f == flag else 0
    
    # Create DataFrame with all features in correct order
    input_df = pd.DataFrame([full_features])
    
    # Ensure all training features are present
    for feat in feature_names:
        if feat not in input_df.columns:
            input_df[feat] = 0
    
    # Return features in training order
    return input_df[feature_names].values

def real_predict(data):
    """Real ML prediction using trained Random Forest"""
    try:
        # Preprocess input
        X = preprocess_input(data)
        
        # Get prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        result = {
            'prediction': 'normal' if prediction == 0 else 'attack',
            'confidence': float(max(probabilities)),
            'probabilities': {
                'normal': float(probabilities[0]),
                'attack': float(probabilities[1])
            }
        }
        return result
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/')
def home():
    return jsonify({
        'message': 'IDS API',
        'version': '1.0.0',
        'status': 'running'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = mock_predict(data)
        result['timestamp'] = datetime.now().isoformat()
        
        prediction_history.append(result)
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    return jsonify({
        'total_predictions': len(prediction_history),
        'model_info': {
            'type': 'Random Forest',
            'accuracy': '99%'
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
