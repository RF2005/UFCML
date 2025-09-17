#!/usr/bin/env python3
"""
UFC Fighter Matchup Predictor Web App
Sports betting prediction interface - just enter fighter names!
Automatically pulls career stats and predicts fight outcomes
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append('/Users/ralphfrancolini/UFCML')

from src.core.fighter_matchup_predictor import FighterMatchupPredictor
import traceback

app = Flask(__name__)

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the predictor instance"""
    global predictor
    try:
        predictor = FighterMatchupPredictor()
        return True
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/fighters')
def get_fighters():
    """Get list of all available fighters"""
    global predictor
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500

    try:
        fighters = list(predictor.fighters.keys())
        fighters.sort()
        return jsonify({'fighters': fighters})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_matchup():
    """Predict matchup between two fighters"""
    global predictor
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500

    try:
        data = request.get_json()
        fighter_a = data.get('fighter_a')
        fighter_b = data.get('fighter_b')

        if not fighter_a or not fighter_b:
            return jsonify({'error': 'Both fighters must be specified'}), 400

        if fighter_a == fighter_b:
            return jsonify({'error': 'Cannot predict a fighter against themselves'}), 400

        # Make prediction
        result = predictor.predict_matchup(fighter_a, fighter_b)

        if not result:
            return jsonify({'error': 'Prediction failed'}), 500

        # Format response
        fight_data = result['fight_simulation']
        fighter_a_profile = result['fighter_a_profile']
        fighter_b_profile = result['fighter_b_profile']

        # Get style scores
        a_grappling = predictor.get_fighting_style_multiplier(fighter_a_profile)
        b_grappling = predictor.get_fighting_style_multiplier(fighter_b_profile)

        response = {
            'prediction': result['final_prediction'],
            'confidence': f"{result['final_confidence']:.1%}",
            'fighter_a': {
                'name': fighter_a_profile.name,
                'weight_class': fighter_a_profile.weight_class,
                'record': f"{fighter_a_profile.record[0]}-{fighter_a_profile.record[1]}-{fighter_a_profile.record[2] if len(fighter_a_profile.record) > 2 else 0}",
                'style': 'Grappler' if a_grappling > 0.6 else 'Striker' if a_grappling < 0.4 else 'Mixed',
                'style_score': f"{a_grappling:.2f}",
                'stats': {
                    'sig_strikes': fight_data['r_sig_str_landed'],
                    'takedowns': fight_data['r_td'],
                    'knockdowns': fight_data['r_kd'],
                    'control_time': f"{fight_data['r_ctrl_time_sec']//60}:{fight_data['r_ctrl_time_sec']%60:02d}"
                }
            },
            'fighter_b': {
                'name': fighter_b_profile.name,
                'weight_class': fighter_b_profile.weight_class,
                'record': f"{fighter_b_profile.record[0]}-{fighter_b_profile.record[1]}-{fighter_b_profile.record[2] if len(fighter_b_profile.record) > 2 else 0}",
                'style': 'Grappler' if b_grappling > 0.6 else 'Striker' if b_grappling < 0.4 else 'Mixed',
                'style_score': f"{b_grappling:.2f}",
                'stats': {
                    'sig_strikes': fight_data['b_sig_str_landed'],
                    'takedowns': fight_data['b_td'],
                    'knockdowns': fight_data['b_kd'],
                    'control_time': f"{fight_data['b_ctrl_time_sec']//60}:{fight_data['b_ctrl_time_sec']%60:02d}"
                }
            },
            'models': {
                name: pred for name, pred in result['individual_predictions'].items()
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🥊 UFC Fighter Matchup Predictor Web App")
    print("=" * 50)
    print("Initializing predictor...")

    if initialize_predictor():
        print("✅ Predictor initialized successfully!")
        print(f"📊 Loaded {len(predictor.fighters)} fighters")
        print("\n🌐 Starting web server...")
        print("Visit: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize predictor")