#!/usr/bin/env python3
"""
Future Fight Prediction Framework
=================================

Enables predictions for upcoming UFC fights that haven't happened yet.
This addresses testing on truly future, unseen data.

Features:
1. Input upcoming fight cards
2. Generate predictions for each fight
3. Track prediction accuracy when results become available
4. Maintain prediction logs for validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import sys

sys.path.append('/Users/ralphfrancolini/UFCML')
from enhanced_random_forest import EnhancedUFCRandomForest
from enhanced_feature_engineering import EnhancedFeatureEngineer

class FutureFightPredictor:
    """Framework for predicting upcoming UFC fights and tracking results."""

    def __init__(self, model_path='models/enhanced_ufc_random_forest.pkl'):
        self.model_path = model_path
        self.model = None
        self.predictions_log = 'future_predictions_log.json'
        self.load_model()

    def load_model(self):
        """Load the trained UFC prediction model."""
        try:
            self.model = EnhancedUFCRandomForest.load_model(self.model_path)
            if self.model:
                print("✅ UFC prediction model loaded successfully")
                return True
            else:
                print("❌ Failed to load model")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def predict_upcoming_fight(self, fighter_a, fighter_b, fight_date=None,
                             title_fight=False, weight_class=None, event_name=None):
        """
        Predict outcome of an upcoming fight.

        Args:
            fighter_a: Name of fighter A
            fighter_b: Name of fighter B
            fight_date: Date of the fight (defaults to today)
            title_fight: Whether it's a title fight
            weight_class: Weight class
            event_name: Name of the UFC event

        Returns:
            Dictionary with prediction details
        """
        if not self.model or not self.model.is_trained:
            print("❌ Model not available")
            return None

        if fight_date is None:
            fight_date = datetime.now()
        elif isinstance(fight_date, str):
            fight_date = datetime.strptime(fight_date, '%Y-%m-%d')

        print(f"🥊 Predicting: {fighter_a} vs {fighter_b}")
        if event_name:
            print(f"📅 Event: {event_name}")
        print(f"📅 Date: {fight_date.strftime('%Y-%m-%d')}")

        # Generate prediction
        result = self.model.predict_fight(
            fighter_a, fighter_b, fight_date, title_fight, weight_class
        )

        if result is None:
            print("❌ Prediction failed")
            return None

        # Enhanced prediction details
        prediction = {
            'prediction_id': f"{fighter_a.replace(' ', '_')}_vs_{fighter_b.replace(' ', '_')}_{fight_date.strftime('%Y%m%d')}",
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'fight_date': fight_date.strftime('%Y-%m-%d'),
            'event_name': event_name,
            'title_fight': title_fight,
            'weight_class': weight_class,
            'predicted_winner': result['predicted_winner'],
            'confidence': result['confidence'],
            'fighter_a_probability': result['fighter_a_prob'],
            'fighter_b_probability': result['fighter_b_prob'],
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'actual_result': None,  # To be filled when fight happens
            'result_verified': False
        }

        # Display prediction
        print(f"🏆 Predicted Winner: {result['predicted_winner']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"⚔️  {fighter_a}: {result['fighter_a_prob']:.1%}")
        print(f"⚔️  {fighter_b}: {result['fighter_b_prob']:.1%}")

        # Log prediction
        self._log_prediction(prediction)

        return prediction

    def predict_fight_card(self, fights, event_name=None, event_date=None):
        """
        Predict an entire fight card.

        Args:
            fights: List of fight dictionaries with fighter_a, fighter_b, etc.
            event_name: Name of the event
            event_date: Date of the event

        Returns:
            List of predictions
        """
        print(f"🥊 PREDICTING FIGHT CARD")
        if event_name:
            print(f"📅 Event: {event_name}")
        if event_date:
            print(f"📅 Date: {event_date}")
        print("=" * 50)

        predictions = []

        for i, fight in enumerate(fights, 1):
            print(f"\n🥊 Fight {i}: {fight['fighter_a']} vs {fight['fighter_b']}")

            prediction = self.predict_upcoming_fight(
                fighter_a=fight['fighter_a'],
                fighter_b=fight['fighter_b'],
                fight_date=event_date or fight.get('fight_date'),
                title_fight=fight.get('title_fight', False),
                weight_class=fight.get('weight_class'),
                event_name=event_name
            )

            if prediction:
                predictions.append(prediction)

        print(f"\n✅ Generated {len(predictions)} predictions for {event_name or 'fight card'}")
        return predictions

    def _log_prediction(self, prediction):
        """Log prediction to file for future validation."""
        try:
            # Load existing log
            try:
                with open(self.predictions_log, 'r') as f:
                    log = json.load(f)
            except FileNotFoundError:
                log = []

            # Add new prediction
            log.append(prediction)

            # Save updated log
            with open(self.predictions_log, 'w') as f:
                json.dump(log, f, indent=2)

        except Exception as e:
            print(f"⚠️  Warning: Could not log prediction: {e}")

    def update_fight_result(self, prediction_id, actual_winner):
        """
        Update a prediction with the actual fight result.

        Args:
            prediction_id: ID of the prediction to update
            actual_winner: Name of the actual winner
        """
        try:
            with open(self.predictions_log, 'r') as f:
                log = json.load(f)

            # Find and update the prediction
            updated = False
            for prediction in log:
                if prediction['prediction_id'] == prediction_id:
                    prediction['actual_result'] = actual_winner
                    prediction['result_verified'] = True
                    prediction['result_update_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Calculate if prediction was correct
                    prediction['correct'] = (prediction['predicted_winner'] == actual_winner)

                    updated = True
                    print(f"✅ Updated result for {prediction['fighter_a']} vs {prediction['fighter_b']}")
                    print(f"🏆 Actual winner: {actual_winner}")
                    print(f"🎯 Prediction {'✅ CORRECT' if prediction['correct'] else '❌ INCORRECT'}")
                    break

            if not updated:
                print(f"❌ Prediction ID not found: {prediction_id}")
                return False

            # Save updated log
            with open(self.predictions_log, 'w') as f:
                json.dump(log, f, indent=2)

            return True

        except Exception as e:
            print(f"❌ Error updating result: {e}")
            return False

    def analyze_prediction_accuracy(self):
        """Analyze accuracy of verified predictions."""
        try:
            with open(self.predictions_log, 'r') as f:
                log = json.load(f)

            verified_predictions = [p for p in log if p['result_verified']]

            if not verified_predictions:
                print("📊 No verified predictions available yet")
                return None

            correct_predictions = sum(1 for p in verified_predictions if p['correct'])
            total_predictions = len(verified_predictions)
            accuracy = correct_predictions / total_predictions

            print(f"📊 PREDICTION ACCURACY ANALYSIS")
            print("=" * 50)
            print(f"✅ Verified predictions: {total_predictions}")
            print(f"🎯 Correct predictions: {correct_predictions}")
            print(f"📈 Overall accuracy: {accuracy:.1%}")

            # Show recent predictions
            recent_predictions = sorted(verified_predictions,
                                      key=lambda x: x['fight_date'], reverse=True)[:10]

            print(f"\n🔍 Recent Verified Predictions:")
            for pred in recent_predictions:
                status = "✅" if pred['correct'] else "❌"
                print(f"  {status} {pred['fight_date']}: {pred['fighter_a']} vs {pred['fighter_b']}")
                print(f"      Predicted: {pred['predicted_winner']} ({pred['confidence']:.1%})")
                print(f"      Actual: {pred['actual_result']}")

            return {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'verified_predictions': verified_predictions
            }

        except Exception as e:
            print(f"❌ Error analyzing predictions: {e}")
            return None

    def list_pending_predictions(self):
        """List predictions that haven't been verified yet."""
        try:
            with open(self.predictions_log, 'r') as f:
                log = json.load(f)

            pending = [p for p in log if not p['result_verified']]

            if not pending:
                print("📊 No pending predictions")
                return []

            print(f"📊 PENDING PREDICTIONS ({len(pending)})")
            print("=" * 50)

            for pred in pending:
                print(f"🥊 {pred['fight_date']}: {pred['fighter_a']} vs {pred['fighter_b']}")
                print(f"   Predicted: {pred['predicted_winner']} ({pred['confidence']:.1%})")
                print(f"   ID: {pred['prediction_id']}")
                if pred.get('event_name'):
                    print(f"   Event: {pred['event_name']}")

            return pending

        except FileNotFoundError:
            print("📊 No prediction log found")
            return []
        except Exception as e:
            print(f"❌ Error listing predictions: {e}")
            return []

def example_upcoming_fights():
    """Example of how to use the future fight predictor."""
    predictor = FutureFightPredictor()

    # Example upcoming fight card (replace with actual upcoming fights)
    upcoming_fights = [
        {
            'fighter_a': 'Jon Jones',
            'fighter_b': 'Stipe Miocic',
            'title_fight': True,
            'weight_class': 'Heavyweight'
        },
        {
            'fighter_a': 'Israel Adesanya',
            'fighter_b': 'Alex Pereira',
            'title_fight': False,
            'weight_class': 'Middleweight'
        }
    ]

    # Predict the fights
    predictions = predictor.predict_fight_card(
        upcoming_fights,
        event_name="UFC Example Event",
        event_date="2025-10-01"
    )

    # Later, update results when fights happen
    # predictor.update_fight_result("Jon_Jones_vs_Stipe_Miocic_20251001", "Jon Jones")

    return predictions

def main():
    """Demonstrate the future fight prediction system."""
    print("🔮 FUTURE FIGHT PREDICTION SYSTEM")
    print("=" * 50)

    predictor = FutureFightPredictor()

    # Show any pending predictions
    predictor.list_pending_predictions()

    # Analyze any verified predictions
    predictor.analyze_prediction_accuracy()

    # Run example predictions
    example_upcoming_fights()

if __name__ == "__main__":
    main()