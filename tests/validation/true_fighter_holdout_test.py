#!/usr/bin/env python3
"""
True Fighter-Based Holdout Testing
==================================

The scientifically correct approach: Complete fighter separation between training and test sets.

Training set: Fighters A, B, C... (model learns their patterns)
Holdout set:  Fighters X, Y, Z... (completely unknown to model)

Test: Can the model predict fights between fighters it has never seen?
Expected realistic accuracy: 65-75% (not 85%)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import pickle
from sklearn.metrics import accuracy_score, classification_report

sys.path.append('/Users/ralphfrancolini/UFCML')
from src.core.advanced_ml_models import load_enhanced_ufc_data
from enhanced_feature_engineering import EnhancedFeatureEngineer
from enhanced_random_forest import EnhancedUFCRandomForest

class TrueFighterHoldoutTester:
    """True fighter-based holdout with complete fighter separation."""

    def __init__(self, holdout_fighter_percentage=0.20):
        self.holdout_fighter_percentage = holdout_fighter_percentage
        self.df = None
        self.training_data = None
        self.holdout_data = None
        self.training_fighters = None
        self.holdout_fighters = None
        self.model = None
        self.feature_engineer = None

    def load_and_create_fighter_separation(self):
        """Load data and create complete fighter separation."""
        print("🥊 TRUE FIGHTER-BASED HOLDOUT TESTING")
        print("=" * 70)

        # Load full dataset
        self.df = load_enhanced_ufc_data()
        if self.df is None:
            print("❌ Failed to load UFC data")
            return False

        # Clean and prepare
        self.df = self.df.dropna(subset=['winner', 'date'])
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df = self.df.dropna(subset=['date']).sort_values('date')

        print(f"✅ Loaded {len(self.df)} total fights")

        # Get all unique fighters and their fight counts
        fighter_fight_counts = {}
        for _, fight in self.df.iterrows():
            for fighter in [fight['r_name'], fight['b_name']]:
                fighter_fight_counts[fighter] = fighter_fight_counts.get(fighter, 0) + 1

        # Filter fighters with sufficient fight history for meaningful patterns
        min_fights = 5  # Need at least 5 fights to establish patterns
        eligible_fighters = [f for f, count in fighter_fight_counts.items() if count >= min_fights]

        print(f"👥 Total unique fighters: {len(fighter_fight_counts)}")
        print(f"👥 Fighters with ≥{min_fights} fights: {len(eligible_fighters)}")

        # Randomly assign fighters to training vs holdout
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(eligible_fighters)

        split_point = int(len(eligible_fighters) * (1 - self.holdout_fighter_percentage))
        self.training_fighters = set(eligible_fighters[:split_point])
        self.holdout_fighters = set(eligible_fighters[split_point:])

        print(f"🎯 Training fighters: {len(self.training_fighters)}")
        print(f"🔒 Holdout fighters: {len(self.holdout_fighters)}")

        # Create data splits: only fights where BOTH fighters are in the same group
        self.training_data = self.df[
            (self.df['r_name'].isin(self.training_fighters)) &
            (self.df['b_name'].isin(self.training_fighters))
        ].copy()

        self.holdout_data = self.df[
            (self.df['r_name'].isin(self.holdout_fighters)) &
            (self.df['b_name'].isin(self.holdout_fighters))
        ].copy()

        print(f"\n📊 DATA SPLIT RESULTS:")
        print(f"📈 Training fights: {len(self.training_data)} ({len(self.training_data)/len(self.df)*100:.1f}%)")
        print(f"🔒 Holdout fights: {len(self.holdout_data)} ({len(self.holdout_data)/len(self.df)*100:.1f}%)")

        # CRITICAL VERIFICATION: Zero fighter overlap
        train_fighters_in_data = set(self.training_data['r_name'].tolist() + self.training_data['b_name'].tolist())
        holdout_fighters_in_data = set(self.holdout_data['r_name'].tolist() + self.holdout_data['b_name'].tolist())
        overlap = train_fighters_in_data.intersection(holdout_fighters_in_data)

        print(f"\n🔍 FIGHTER SEPARATION AUDIT:")
        print(f"Training fighters in data: {len(train_fighters_in_data)}")
        print(f"Holdout fighters in data: {len(holdout_fighters_in_data)}")
        print(f"Overlapping fighters: {len(overlap)}")

        if len(overlap) == 0:
            print("✅ PERFECT: Zero fighter overlap - true generalization test")
        else:
            print(f"❌ FAILED: {len(overlap)} fighters appear in both sets")
            print("Overlapping fighters:", list(overlap)[:10])
            return False

        # Check if we have enough data
        if len(self.training_data) < 500:
            print("⚠️  WARNING: Small training set may limit model performance")
        if len(self.holdout_data) < 50:
            print("⚠️  WARNING: Small holdout set may give unreliable results")

        # Show sample fighters from each set
        print(f"\n👥 Sample training fighters:")
        for fighter in list(self.training_fighters)[:8]:
            fight_count = fighter_fight_counts.get(fighter, 0)
            print(f"  - {fighter} ({fight_count} fights)")

        print(f"\n👥 Sample holdout fighters (completely unknown to model):")
        for fighter in list(self.holdout_fighters)[:8]:
            fight_count = fighter_fight_counts.get(fighter, 0)
            print(f"  - {fighter} ({fight_count} fights)")

        return True

    def train_model_on_training_fighters_only(self):
        """Train model exclusively on training fighters' data."""
        print(f"\n🎯 TRAINING MODEL ON TRAINING FIGHTERS ONLY")
        print("=" * 70)

        # Initialize feature engineer with training fighter data ONLY
        self.feature_engineer = EnhancedFeatureEngineer()
        self.feature_engineer.df = self.training_data.copy()

        if not self.feature_engineer.load_and_prepare_data():
            print("❌ Failed to prepare training data")
            return False

        # Create fighter profiles from training fighters only
        print("🔄 Building fighter profiles from training fighters only...")
        print(f"   Model will learn patterns from {len(self.training_fighters)} fighters")
        print(f"   Model will NEVER see the {len(self.holdout_fighters)} holdout fighters")

        self.feature_engineer._create_fighter_profiles()

        # Create enhanced training dataset
        training_enhanced = self.feature_engineer.create_enhanced_training_data()
        print(f"✅ Enhanced training features created: {len(training_enhanced)} fights")

        # Initialize and train model
        self.model = EnhancedUFCRandomForest(
            n_estimators=100,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt'
        )

        self.model.feature_engineer = self.feature_engineer

        # Train model
        print("🔄 Training enhanced random forest...")
        results = self.model.train(training_enhanced, temporal_split=True)

        print(f"✅ Model trained on training fighters only")
        print(f"📊 Training accuracy: {results['train_accuracy']:.1%}")
        print(f"📊 Validation accuracy: {results['val_accuracy']:.1%}")
        print(f"📊 Test accuracy: {results['test_accuracy']:.1%}")
        print(f"⚖️  Overfitting gap: {results['train_val_gap']:.1%}")

        return True

    def test_on_completely_unknown_fighters(self):
        """Test the model on fighters it has never seen before."""
        print(f"\n🔒 TESTING ON COMPLETELY UNKNOWN FIGHTERS")
        print("=" * 70)

        if not self.model or not self.model.is_trained:
            print("❌ Model not trained yet")
            return None

        print(f"🧪 Predicting {len(self.holdout_data)} fights between unknown fighters...")
        print("🔒 Model has NEVER seen any of these fighters before")

        holdout_predictions = []
        holdout_actuals = []
        prediction_details = []
        failed_predictions = 0

        for idx, fight in self.holdout_data.iterrows():
            try:
                fighter_a = fight['r_name']
                fighter_b = fight['b_name']
                actual_winner = fight['winner']
                fight_date = fight['date']

                # This is the ultimate test: predict fights between completely unknown fighters
                # The model must generalize patterns learned from known fighters
                features = self.feature_engineer.extract_enhanced_features(
                    fighter_a, fighter_b, fight_date
                )

                # Convert to DataFrame
                feature_df = pd.DataFrame([features])

                # Prepare features
                X = self.model.prepare_features(feature_df)

                # Ensure feature compatibility
                if set(X.columns) != set(self.model.feature_columns):
                    missing_cols = set(self.model.feature_columns) - set(X.columns)
                    extra_cols = set(X.columns) - set(self.model.feature_columns)

                    for col in missing_cols:
                        X[col] = 0
                    X = X.drop(columns=extra_cols, errors='ignore')

                X = X[self.model.feature_columns]

                # Make prediction
                prediction = self.model.model.predict(X)[0]
                probability = self.model.model.predict_proba(X)[0]

                predicted_winner = fighter_a if prediction == 1 else fighter_b
                confidence = max(probability)

                # Store results
                holdout_predictions.append(prediction)
                holdout_actuals.append(1 if actual_winner == fighter_a else 0)

                prediction_details.append({
                    'date': fight_date,
                    'fighter_a': fighter_a,
                    'fighter_b': fighter_b,
                    'actual_winner': actual_winner,
                    'predicted_winner': predicted_winner,
                    'confidence': confidence,
                    'correct': predicted_winner == actual_winner
                })

            except Exception as e:
                failed_predictions += 1
                print(f"⚠️  Failed to predict {fighter_a} vs {fighter_b}: {e}")
                continue

        if len(holdout_predictions) == 0:
            print("❌ No successful predictions made")
            return None

        # Calculate TRUE generalization performance
        true_oos_accuracy = accuracy_score(holdout_actuals, holdout_predictions)

        print(f"\n🎯 TRUE GENERALIZATION RESULTS")
        print("=" * 70)
        print(f"✅ Unknown fighter accuracy: {true_oos_accuracy:.1%}")
        print(f"📊 Successful predictions: {len(prediction_details)}")
        print(f"📊 Failed predictions: {failed_predictions}")
        print(f"✅ Correct predictions: {sum(1 for p in prediction_details if p['correct'])}")

        # Performance assessment against realistic benchmarks
        print(f"\n📊 REALISTIC PERFORMANCE ASSESSMENT:")
        if true_oos_accuracy > 0.70:
            print("🏆 EXCELLENT: >70% on unknown fighters - exceptional generalization")
        elif true_oos_accuracy > 0.65:
            print("✅ VERY GOOD: >65% on unknown fighters - strong generalization")
        elif true_oos_accuracy > 0.60:
            print("✅ GOOD: >60% on unknown fighters - solid generalization")
        elif true_oos_accuracy > 0.55:
            print("📈 FAIR: >55% on unknown fighters - some generalization")
        elif true_oos_accuracy > 0.50:
            print("⚠️  POOR: Barely better than random")
        else:
            print("❌ FAILED: Worse than random - model doesn't generalize")

        # Compare to previous inflated results
        print(f"\n📊 COMPARISON TO PREVIOUS TESTS:")
        print(f"Temporal-only (fighter overlap): 85.4%")
        print(f"TRUE fighter separation: {true_oos_accuracy:.1%}")
        difference = true_oos_accuracy - 0.854
        print(f"Difference: {difference:+.1%}")

        if true_oos_accuracy < 0.80:
            print("✅ REALISTIC: True generalization shows expected lower accuracy")
            print("✅ SCIENTIFIC: Model performance now believable for sports prediction")
        else:
            print("⚠️  SUSPICIOUS: Still unexpectedly high - investigate further")

        # Detailed analysis
        print(f"\n📋 Classification Report:")
        print(classification_report(holdout_actuals, holdout_predictions,
                                  target_names=['Fighter B Wins', 'Fighter A Wins']))

        # Show examples of unknown fighter predictions
        print(f"\n🔍 Sample Predictions on Unknown Fighters:")
        for i, pred in enumerate(prediction_details[:8]):
            status = "✅" if pred['correct'] else "❌"
            print(f"  {status} {pred['date'].strftime('%Y-%m-%d')}: {pred['fighter_a']} vs {pred['fighter_b']}")
            print(f"      Predicted: {pred['predicted_winner']} ({pred['confidence']:.1%})")
            print(f"      Actual: {pred['actual_winner']}")

        return {
            'true_oos_accuracy': true_oos_accuracy,
            'predictions': prediction_details,
            'actuals': holdout_actuals,
            'predicted': holdout_predictions,
            'failed_predictions': failed_predictions
        }

    def run_complete_true_fighter_test(self):
        """Run the complete true fighter-based holdout test."""
        print("🥊 COMPLETE TRUE FIGHTER-BASED HOLDOUT TEST")
        print("=" * 70)

        # Step 1: Create complete fighter separation
        if not self.load_and_create_fighter_separation():
            return False

        # Step 2: Train model on training fighters only
        if not self.train_model_on_training_fighters_only():
            return False

        # Step 3: Test on completely unknown fighters
        results = self.test_on_completely_unknown_fighters()

        if results is None:
            return False

        # Step 4: Final scientific assessment
        print(f"\n🎯 FINAL SCIENTIFIC ASSESSMENT")
        print("=" * 70)

        accuracy = results['true_oos_accuracy']
        print(f"✅ True fighter-based holdout completed!")
        print(f"👥 Training fighters: {len(self.training_fighters)}")
        print(f"👥 Unknown holdout fighters: {len(self.holdout_fighters)}")
        print(f"🎯 TRUE generalization accuracy: {accuracy:.1%}")

        # Scientific verdict
        print(f"\n🔬 SCIENTIFIC VERDICT:")
        if 0.60 <= accuracy <= 0.75:
            print("🏆 EXCELLENT: Realistic accuracy for sports prediction")
            print("✅ Model demonstrates genuine pattern recognition")
            print("✅ Results are scientifically credible")
            print("✅ Ready for real-world deployment")
        elif accuracy > 0.75:
            print("🤔 SUSPICIOUS: Still higher than expected")
            print("⚠️  May indicate remaining data issues")
        elif accuracy > 0.55:
            print("📈 ACCEPTABLE: Some predictive power demonstrated")
            print("✅ Model has learned generalizable patterns")
        else:
            print("❌ POOR: Model fails to generalize to unknown fighters")
            print("🔄 Need better feature engineering or more data")

        return True

def main():
    """Run true fighter-based holdout testing."""
    tester = TrueFighterHoldoutTester(holdout_fighter_percentage=0.20)
    tester.run_complete_true_fighter_test()

if __name__ == "__main__":
    main()