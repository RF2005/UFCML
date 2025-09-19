#!/usr/bin/env python3
"""
UFC Fight Predictor Web Application
===================================

Simple Flask web UI for UFC fight predictions with black/red UFC theming.
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
from pathlib import Path
from datetime import datetime

# Resolve project root and ensure it's on sys.path so modules load in any environment
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhanced_random_forest import EnhancedUFCRandomForest

app = Flask(__name__)

# Global model variable
model = None

# Allow overriding the model path via environment variable
DEFAULT_MODEL_PATH = PROJECT_ROOT / 'models' / 'enhanced_ufc_random_forest.pkl'
MODEL_PATH = Path(os.environ.get('MODEL_PATH', DEFAULT_MODEL_PATH))

def load_model():
    """Load the trained UFC prediction model."""
    global model
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        absolute_path = MODEL_PATH.resolve()
        file_size_mb = absolute_path.stat().st_size / (1024 * 1024)
        print(f"📦 Loading model from {absolute_path} ({file_size_mb:.2f} MB)")

        model = EnhancedUFCRandomForest.load_model(str(absolute_path))
        print("✅ UFC prediction model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

@app.route('/')
def index():
    """Main page with fight prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_fight():
    """Handle fight prediction requests."""
    try:
        # Get form data
        fighter_a = request.form.get('fighter_a', '').strip()
        fighter_b = request.form.get('fighter_b', '').strip()
        weight_class = request.form.get('weight_class', '')
        title_fight = request.form.get('title_fight') == 'on'

        # Validate inputs
        if not fighter_a or not fighter_b:
            return jsonify({
                'error': 'Both fighter names are required',
                'success': False
            })

        if fighter_a.lower() == fighter_b.lower():
            return jsonify({
                'error': 'Fighter names must be different',
                'success': False
            })

        # Make prediction
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            })

        # Use current date for prediction
        fight_date = datetime.now()

        # Get prediction from model (simplified like debug app)
        print(f"🔄 Predicting: {fighter_a} vs {fighter_b}")
        prediction = model.predict_fight(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            fight_date=fight_date
        )
        print(f"✅ Prediction successful: {prediction.get('predicted_winner', 'Unknown')}")

        # Format response
        response = {
            'success': True,
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'predicted_winner': prediction['predicted_winner'],
            'confidence': round(prediction['confidence'] * 100, 1),
            'fighter_a_probability': round(prediction['fighter_a_prob'] * 100, 1),
            'fighter_b_probability': round(prediction['fighter_b_prob'] * 100, 1),
            'weight_class': weight_class,
            'title_fight': title_fight
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        })

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🥊 Starting UFC Fight Predictor Web App")
    print("=" * 50)

    # Load the model
    if load_model():
        print("🚀 Starting Flask server...")
        print("Visit: http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Cannot start app without model")
        sys.exit(1)
