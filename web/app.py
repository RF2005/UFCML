#!/usr/bin/env python3
"""
UFC Fight Predictor Web Application
===================================

Simple Flask web UI for UFC fight predictions with black/red UFC theming.
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import pickle
from datetime import datetime

# Add parent directory to path
sys.path.append('/Users/ralphfrancolini/UFCML')

from enhanced_random_forest import EnhancedUFCRandomForest

app = Flask(__name__)

# Global model variable
model = None

def load_model():
    """Load the trained UFC prediction model."""
    global model
    try:
        model_path = '/Users/ralphfrancolini/UFCML/models/enhanced_ufc_random_forest.pkl'
        model = EnhancedUFCRandomForest.load_model(model_path)
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
