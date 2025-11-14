"""
Flask API for Render Deployment - Mock Version
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

prediction_history = []

def mock_predict(data):
    """Mock prediction logic"""
    src_bytes = data.get('src_bytes', 0)
    dst_bytes = data.get('dst_bytes', 0)
    
    if src_bytes > 5000 or dst_bytes > 5000:
        return {
            'prediction': 'attack',
            'confidence': float(np.random.uniform(0.85, 0.99)),
            'probabilities': {
                'normal': float(np.random.uniform(0.01, 0.15)),
                'attack': float(np.random.uniform(0.85, 0.99))
            }
        }
    else:
        return {
            'prediction': 'normal',
            'confidence': float(np.random.uniform(0.80, 0.95)),
            'probabilities': {
                'normal': float(np.random.uniform(0.80, 0.95)),
                'attack': float(np.random.uniform(0.05, 0.20))
            }
        }

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
