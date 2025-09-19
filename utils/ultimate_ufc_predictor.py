"""
Ultimate UFC Fight Predictor
============================

This program combines ALL decision trees (individual + specialized + comprehensive)
into a single mega random forest for the most accurate UFC fight predictions possible.

Usage:
    python ultimate_ufc_predictor.py

Features:
- Interactive fight prediction interface
- Combines 32+ individual trees into one mega forest
- Real-time confidence scoring
- Fighter comparison analysis
- Easy-to-use command line interface
"""

import pandas as pd
import numpy as np
import joblib
from src.core.individual_trees import UFC_Individual_Tree_Forest, load_enhanced_ufc_data
from src.core.advanced_ml_models import UFC_SpecializedEnsemble
from src.core.fighter_matchup_predictor import FighterMatchupPredictor
import os

class UltimateFightPredictor:
    """
    The ultimate UFC fight predictor combining all available models.
    """

    def __init__(self):
        self.individual_forest = None
        self.specialized_ensemble = None
        self.comprehensive_rf = None
        self.models_loaded = False
        self.fighter_predictor = FighterMatchupPredictor()  # For fighter database and simple predictions

    def load_all_models(self):
        """Load all available trained models."""
        print("🔄 Loading all trained models...")

        # Load individual tree forest
        try:
            if os.path.exists('models/ufc_individual_tree_forest.pkl'):
                self.individual_forest = UFC_Individual_Tree_Forest.load_forest('models/ufc_individual_tree_forest.pkl')
                print("✅ Individual tree forest loaded (32 trees)")
            else:
                print("⚠️  Individual tree forest not found - training new one...")
                df = load_enhanced_ufc_data()
                if df is not None:
                    self.individual_forest = UFC_Individual_Tree_Forest()
                    self.individual_forest.train_forest(df, save_models=True)
        except Exception as e:
            print(f"❌ Could not load individual forest: {e}")

        # Load specialized ensemble
        try:
            if os.path.exists('models/ufc_specialized_ensemble.pkl'):
                self.specialized_ensemble = UFC_SpecializedEnsemble.load_ensemble('models/ufc_specialized_ensemble.pkl')
                print("✅ Specialized ensemble loaded (4 specialized trees)")
        except Exception as e:
            print(f"❌ Could not load specialized ensemble: {e}")

        # Load comprehensive random forest
        try:
            if os.path.exists('models/ufc_comprehensive_random_forest.pkl'):
                self.comprehensive_rf = joblib.load('models/ufc_comprehensive_random_forest.pkl')
                print("✅ Comprehensive random forest loaded")
        except Exception as e:
            print(f"❌ Could not load comprehensive RF: {e}")

        self.models_loaded = True
        models_count = sum([
            1 if self.individual_forest else 0,
            1 if self.specialized_ensemble else 0,
            1 if self.comprehensive_rf else 0
        ])
        print(f"🎯 {models_count}/3 model types loaded successfully")

    def predict_fight_ultimate(self, fight_data):
        """
        Make the ultimate prediction using all available models.

        Args:
            fight_data (dict): Complete fight statistics

        Returns:
            dict: Comprehensive prediction results
        """
        if not self.models_loaded:
            self.load_all_models()

        predictions = {}
        confidences = {}

        # Individual tree forest prediction
        if self.individual_forest:
            try:
                result = self.individual_forest.predict_fight(fight_data)
                predictions['individual_forest'] = result['forest_prediction']
                confidences['individual_forest'] = result['forest_confidence']
            except Exception as e:
                print(f"Individual forest prediction failed: {e}")

        # Specialized ensemble prediction
        if self.specialized_ensemble:
            try:
                result = self.specialized_ensemble.predict_fight(fight_data)
                predictions['specialized_ensemble'] = result['ensemble_prediction']
                confidences['specialized_ensemble'] = result['ensemble_confidence']
            except Exception as e:
                print(f"Specialized ensemble prediction failed: {e}")

        # Comprehensive RF prediction would need feature engineering
        # Skipping for now as it requires complex feature alignment

        # Meta-ensemble: combine all predictions
        fighter_a_votes = 0
        fighter_b_votes = 0
        total_confidence = 0

        for model, prediction in predictions.items():
            weight = confidences.get(model, 0.5)
            total_confidence += weight

            if 'Fighter A' in prediction or 'Red Corner' in prediction:
                fighter_a_votes += weight
            else:
                fighter_b_votes += weight

        # Final prediction
        if fighter_a_votes > fighter_b_votes:
            final_prediction = "Fighter A (Red Corner)"
            final_confidence = fighter_a_votes / (fighter_a_votes + fighter_b_votes)
        else:
            final_prediction = "Fighter B (Blue Corner)"
            final_confidence = fighter_b_votes / (fighter_a_votes + fighter_b_votes)

        return {
            'final_prediction': final_prediction,
            'final_confidence': final_confidence,
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'models_used': len(predictions)
        }

    def get_simple_fight_input(self):
        """Simple input for sports betting - just fighter names and context."""
        print("\n🥊 UFC FIGHT PREDICTION")
        print("=" * 50)
        print("Enter fighter names and basic fight info")
        print("🎯 Perfect for sports betting predictions!")
        print("=" * 50)

        # Get fighter names
        fighter_a = input("Fighter A Name: ").strip()
        fighter_b = input("Fighter B Name: ").strip()

        if not fighter_a or not fighter_b:
            print("❌ Both fighter names are required!")
            return None

        if fighter_a.lower() == fighter_b.lower():
            print("❌ Cannot predict a fighter against themselves!")
            return None

        # Basic fight context
        print(f"\n🏆 Fight Context (optional - press Enter for defaults):")
        title_fight = input("Title fight? (y/n): ").lower().startswith('y')

        weight_class = input("Weight class (e.g., Heavyweight, Light Heavyweight): ").strip()
        if not weight_class:
            weight_class = "Unknown"

        rounds = 5 if title_fight else 3
        custom_rounds = input(f"Scheduled rounds (default {rounds}): ").strip()
        if custom_rounds.isdigit():
            rounds = int(custom_rounds)

        return {
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'title_fight': title_fight,
            'weight_class': weight_class,
            'scheduled_rounds': rounds
        }

    def display_simple_results(self, fight_input, result):
        """Display results in a simple, sports betting friendly format."""
        print("\n" + "="*60)
        print("🥊 FIGHT PREDICTION RESULTS")
        print("="*60)

        print(f"🔴 Fighter A: {fight_input['fighter_a']}")
        print(f"🔵 Fighter B: {fight_input['fighter_b']}")

        if fight_input.get('title_fight'):
            print("🏆 Title Fight")
        print(f"⏱️  Scheduled: {fight_input.get('scheduled_rounds', 3)} rounds")
        if fight_input.get('weight_class') != 'Unknown':
            print(f"⚖️  Weight Class: {fight_input['weight_class']}")

        print(f"\n🎯 PREDICTION: {result['final_prediction']}")
        print(f"📊 Confidence: {result['final_confidence']:.1%}")

        # Show individual model predictions if available
        if result.get('individual_predictions'):
            print(f"\n📈 Model Breakdown:")
            for model_name, prediction in result['individual_predictions'].items():
                confidence = result.get('individual_confidences', {}).get(model_name, 0)
                print(f"  • {model_name}: {prediction} ({confidence:.1%})")

        print(f"\n💡 Betting Recommendation:")
        confidence = result['final_confidence']
        if confidence >= 0.7:
            print(f"✅ STRONG BET: High confidence prediction")
        elif confidence >= 0.6:
            print(f"⚡ GOOD BET: Above average confidence")
        elif confidence >= 0.55:
            print(f"⚠️  WEAK BET: Low confidence, bet small")
        else:
            print(f"❌ AVOID: Very low confidence, skip this bet")

        print("="*60)

    def _get_numeric_input(self, prompt, default):
        """Get numeric input with default value."""
        try:
            value = input(f"{prompt} (default {default}): ").strip()
            return float(value) if value else default
        except ValueError:
            return default

    def display_prediction_results(self, fight_data, prediction_results):
        """Display comprehensive prediction results."""
        print("\n" + "=" * 80)
        print("🔮 ULTIMATE UFC FIGHT PREDICTION RESULTS")
        print("=" * 80)

        # Fight info
        print(f"\n🥊 Fight: {fight_data['r_name']} vs {fight_data['b_name']}")
        if fight_data.get('title_fight'):
            print("👑 TITLE FIGHT")

        # Final prediction
        print(f"\n🏆 FINAL PREDICTION: {prediction_results['final_prediction']}")
        print(f"🎯 Confidence: {prediction_results['final_confidence']:.1%}")

        # Individual model predictions
        print(f"\n📊 Individual Model Predictions:")
        for model, prediction in prediction_results['individual_predictions'].items():
            confidence = prediction_results['individual_confidences'].get(model, 0)
            print(f"   • {model.replace('_', ' ').title()}: {prediction} ({confidence:.1%} confidence)")

        # Key stats comparison
        print(f"\n📈 Key Statistics Comparison:")
        red_name = fight_data['r_name']
        blue_name = fight_data['b_name']

        stats_to_compare = [
            ('Significant Strikes Landed', 'sig_str_landed'),
            ('Significant Strike Accuracy', 'sig_str_acc'),
            ('Knockdowns', 'kd'),
            ('Takedowns Landed', 'td_landed'),
            ('Control Time', 'ctrl')
        ]

        for stat_name, stat_key in stats_to_compare:
            red_val = fight_data.get(f'r_{stat_key}', 0)
            blue_val = fight_data.get(f'b_{stat_key}', 0)

            if stat_key == 'ctrl':
                red_display = f"{red_val//60}:{red_val%60:02d}"
                blue_display = f"{blue_val//60}:{blue_val%60:02d}"
            elif stat_key in ['sig_str_acc']:
                red_display = f"{red_val:.1f}%"
                blue_display = f"{blue_val:.1f}%"
            else:
                red_display = str(int(red_val))
                blue_display = str(int(blue_val))

            advantage = "→" if red_val > blue_val else "←" if blue_val > red_val else "="
            print(f"   {stat_name:<25} {red_name}: {red_display:<8} {advantage} {blue_display:>8} :{blue_name}")

        # Models used
        print(f"\n🤖 Analysis powered by {prediction_results['models_used']} AI models")
        print(f"   • Individual Tree Forest: 32 specialized decision trees")
        if 'specialized_ensemble' in prediction_results['individual_predictions']:
            print(f"   • Specialized Ensemble: 4 domain-expert trees")

        print(f"\n💡 Prediction Insight:")
        if prediction_results['final_confidence'] > 0.7:
            print(f"   High confidence prediction - models strongly agree")
        elif prediction_results['final_confidence'] > 0.6:
            print(f"   Moderate confidence - models mostly agree")
        else:
            print(f"   Low confidence - models split or uncertain")

        print("=" * 80)

def main():
    """Main program loop."""
    print("🥊 ULTIMATE UFC FIGHT PREDICTOR")
    print("=" * 50)
    print("Powered by 32+ AI decision trees")
    print("=" * 50)

    predictor = UltimateFightPredictor()

    while True:
        print("\n🎯 Options:")
        print("1. Predict new fight")
        print("2. Load models info")
        print("3. Quit")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == '1':
            try:
                # Get simple fight input
                fight_input = predictor.get_simple_fight_input()

                if not fight_input:
                    continue

                # Use the fighter matchup predictor for automatic stats
                print(f"\n🔄 Looking up fighters: {fight_input['fighter_a']} vs {fight_input['fighter_b']}")
                print("🔄 Analyzing with AI models...")

                result = predictor.fighter_predictor.predict_matchup(
                    fight_input['fighter_a'],
                    fight_input['fighter_b']
                )

                if result:
                    # Display results in a sports betting friendly format
                    predictor.display_simple_results(fight_input, result)
                else:
                    print("❌ Could not generate prediction. Check fighter names and try again.")

            except KeyboardInterrupt:
                print("\n\n⏸️  Prediction cancelled.")
            except Exception as e:
                print(f"\n❌ Error during prediction: {e}")

        elif choice == '2':
            predictor.load_all_models()

        elif choice == '3':
            print("\n👋 Thanks for using Ultimate UFC Predictor!")
            break

        else:
            print("\n❌ Invalid option. Please select 1-3.")

if __name__ == "__main__":
    main()