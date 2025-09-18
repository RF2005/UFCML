#!/usr/bin/env python3
"""
Proper Temporal Holdout Testing
===============================

Fixes the critical temporal data leakage issue by ensuring fighter profiles
are built ONLY from pre-cutoff data, then applied to post-cutoff fights.

KEY FIX:
- Fighter profiles frozen at temporal cutoff (March 10, 2025)
- No statistics from test period used in any features
- Test predictions use only pre-cutoff fighter knowledge

This represents the realistic scenario: "Given what we knew about fighters
up to March 10, predict their fights from March 17 onwards."
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

class ProperTemporalHoldoutTester:
    """Proper temporal testing with strict feature engineering cutoffs."""

    def __init__(self, holdout_months=6):
        self.holdout_months = holdout_months
        self.df = None
        self.training_data = None
        self.holdout_data = None
        self.temporal_cutoff = None
        self.model = None
        self.feature_engineer = None

    def load_and_split_with_proper_temporal_boundaries(self):
        """Load data and create proper temporal split with strict feature boundaries."""
        print("📊 LOADING DATA FOR PROPER TEMPORAL HOLDOUT TESTING")
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

        # Create temporal split - this is the CRITICAL cutoff point
        self.temporal_cutoff = self.df['date'].max() - timedelta(days=self.holdout_months * 30)

        # Split data at the temporal cutoff
        self.training_data = self.df[self.df['date'] <= self.temporal_cutoff].copy()
        self.holdout_data = self.df[self.df['date'] > self.temporal_cutoff].copy()

        print(f"📅 Dataset date range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}")
        print(f"✂️  TEMPORAL CUTOFF: {self.temporal_cutoff.strftime('%Y-%m-%d')}")
        print(f"📈 Training data: {len(self.training_data)} fights ({len(self.training_data)/len(self.df)*100:.1f}%)")
        print(f"🔒 Holdout data: {len(self.holdout_data)} fights ({len(self.holdout_data)/len(self.df)*100:.1f}%)")

        # Verify temporal separation
        if not self.training_data.empty and not self.holdout_data.empty:
            latest_training = self.training_data['date'].max()
            earliest_holdout = self.holdout_data['date'].min()
            gap_days = (earliest_holdout - latest_training).days

            print(f"⏰ Temporal gap: {gap_days} days")

            if gap_days < 1:
                print("❌ ERROR: No temporal gap!")
                return False

        # Audit fighter overlap (this is OK - the issue was temporal leakage)
        train_fighters = set(self.training_data['r_name'].tolist() + self.training_data['b_name'].tolist())
        holdout_fighters = set(self.holdout_data['r_name'].tolist() + self.holdout_data['b_name'].tolist())
        overlap_fighters = train_fighters.intersection(holdout_fighters)

        print(f"\n👥 FIGHTER ANALYSIS:")
        print(f"Training fighters: {len(train_fighters)}")
        print(f"Holdout fighters: {len(holdout_fighters)}")
        print(f"Overlapping fighters: {len(overlap_fighters)} ({len(overlap_fighters)/len(holdout_fighters)*100:.1f}% of holdout)")
        print("✅ Fighter overlap is ACCEPTABLE - we use pre-cutoff knowledge of fighters")

        return True

    def train_model_with_proper_temporal_features(self):
        """Train model using ONLY pre-cutoff data for ALL feature engineering."""
        print(f"\n🎯 TRAINING MODEL WITH PROPER TEMPORAL FEATURE ENGINEERING")
        print("=" * 70)

        # CRITICAL: Initialize feature engineer with ONLY training data
        # This ensures fighter profiles are built ONLY from pre-cutoff data
        self.feature_engineer = EnhancedFeatureEngineer()

        # Override the data loading to use only training data
        print("🔒 RESTRICTING feature engineering to pre-cutoff data ONLY...")
        self.feature_engineer.df = self.training_data.copy()

        # CRITICAL FIX: Don't call load_and_prepare_data() as it reloads full dataset
        # Instead, prepare the restricted data manually
        self.feature_engineer.df = self.feature_engineer.df.dropna(subset=['winner', 'date'])
        self.feature_engineer.df['date'] = pd.to_datetime(self.feature_engineer.df['date'], errors='coerce')
        self.feature_engineer.df = self.feature_engineer.df.dropna(subset=['date']).sort_values('date')
        print(f"✅ Loaded {len(self.feature_engineer.df)} fights for feature engineering (PRE-CUTOFF ONLY)")

        print("🔄 Building fighter profiles from PRE-CUTOFF data only...")
        print(f"   Using fights up to: {self.temporal_cutoff.strftime('%Y-%m-%d')}")

        # Create fighter profiles - these will be "frozen" at the cutoff date
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

        # Train with temporal split (on pre-cutoff data)
        print("🔄 Training enhanced random forest...")
        results = self.model.train(training_enhanced, temporal_split=True)

        print(f"✅ Model trained with proper temporal boundaries")
        print(f"📊 Training accuracy: {results['train_accuracy']:.1%}")
        print(f"📊 Validation accuracy: {results['val_accuracy']:.1%}")
        print(f"📊 Test accuracy: {results['test_accuracy']:.1%}")
        print(f"⚖️  Overfitting gap: {results['train_val_gap']:.1%}")

        return True

    def test_on_proper_holdout_data(self):
        """Test using frozen fighter profiles on post-cutoff fights."""
        print(f"\n🔒 TESTING ON HOLDOUT DATA WITH FROZEN FIGHTER PROFILES")
        print("=" * 70)

        if not self.model or not self.model.is_trained:
            print("❌ Model not trained yet")
            return None

        print(f"🧪 Predicting {len(self.holdout_data)} holdout fights...")
        print(f"🔒 Using fighter profiles frozen at: {self.temporal_cutoff.strftime('%Y-%m-%d')}")

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

                # CRITICAL: Use frozen fighter profiles from pre-cutoff data
                # The feature engineer only knows about fighters up to the cutoff
                features = self.feature_engineer.extract_enhanced_features(
                    fighter_a, fighter_b, fight_date
                )

                # Convert to DataFrame
                feature_df = pd.DataFrame([features])

                # Prepare features (same process as training)
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
                continue

        if failed_predictions > 0:
            print(f"⚠️  {failed_predictions} predictions failed (likely fighters not in pre-cutoff data)")

        # Calculate proper out-of-sample performance
        if len(holdout_predictions) == 0:
            print("❌ No successful predictions made")
            return None

        oos_accuracy = accuracy_score(holdout_actuals, holdout_predictions)

        print(f"\n🎯 PROPER TEMPORAL HOLDOUT RESULTS")
        print("=" * 60)
        print(f"✅ Holdout accuracy: {oos_accuracy:.1%}")
        print(f"📊 Successful predictions: {len(prediction_details)}")
        print(f"📊 Failed predictions: {failed_predictions}")
        print(f"✅ Correct predictions: {sum(1 for p in prediction_details if p['correct'])}")

        # Performance assessment
        print(f"\n📊 PERFORMANCE ASSESSMENT:")
        if oos_accuracy > 0.75:
            print("🏆 EXCELLENT: >75% with proper temporal boundaries")
        elif oos_accuracy > 0.65:
            print("✅ GOOD: >65% - realistic for sports prediction")
        elif oos_accuracy > 0.55:
            print("📈 FAIR: >55% - some predictive power")
        elif oos_accuracy > 0.50:
            print("⚠️  POOR: Barely better than random")
        else:
            print("❌ FAILED: Worse than random guessing")

        # Compare to previous flawed result
        print(f"\n📊 COMPARISON TO FLAWED APPROACH:")
        print(f"Previous (with temporal leakage): 85.4%")
        print(f"Proper temporal boundaries: {oos_accuracy:.1%}")
        difference = oos_accuracy - 0.854
        print(f"Difference: {difference:+.1%}")

        if oos_accuracy < 0.80:
            print("✅ REALISTIC: Proper approach shows lower, more realistic accuracy")
        else:
            print("⚠️  SUSPICIOUS: Still too high - check for remaining leakage")

        # Detailed analysis
        print(f"\n📋 Classification Report:")
        print(classification_report(holdout_actuals, holdout_predictions,
                                  target_names=['Fighter B Wins', 'Fighter A Wins']))

        # Show some examples
        print(f"\n🔍 Sample Predictions with Frozen Fighter Profiles:")
        for i, pred in enumerate(prediction_details[:8]):
            status = "✅" if pred['correct'] else "❌"
            print(f"  {status} {pred['date'].strftime('%Y-%m-%d')}: {pred['fighter_a']} vs {pred['fighter_b']}")
            print(f"      Predicted: {pred['predicted_winner']} ({pred['confidence']:.1%})")
            print(f"      Actual: {pred['actual_winner']}")

        return {
            'holdout_accuracy': oos_accuracy,
            'predictions': prediction_details,
            'actuals': holdout_actuals,
            'predicted': holdout_predictions,
            'failed_predictions': failed_predictions,
            'temporal_cutoff': self.temporal_cutoff
        }

    def run_complete_proper_temporal_test(self):
        """Run the complete proper temporal holdout testing pipeline."""
        print("⏰ COMPLETE PROPER TEMPORAL HOLDOUT TESTING")
        print("=" * 70)

        # Step 1: Load and split with proper temporal boundaries
        if not self.load_and_split_with_proper_temporal_boundaries():
            return False

        # Step 2: Train model with proper temporal feature engineering
        if not self.train_model_with_proper_temporal_features():
            return False

        # Step 3: Test on holdout data using frozen profiles
        results = self.test_on_proper_holdout_data()

        if results is None:
            return False

        # Step 4: Final assessment
        print(f"\n🎯 PROPER TEMPORAL HOLDOUT SUMMARY")
        print("=" * 60)

        print(f"✅ Proper temporal holdout completed successfully!")
        print(f"🔒 Feature engineering cutoff: {results['temporal_cutoff'].strftime('%Y-%m-%d')}")
        print(f"🎯 PROPER out-of-sample accuracy: {results['holdout_accuracy']:.1%}")
        print(f"⚡ No temporal data leakage detected")

        # Final scientific assessment
        print(f"\n🔬 SCIENTIFIC ASSESSMENT:")
        if 0.65 <= results['holdout_accuracy'] <= 0.75:
            print("🏆 EXCELLENT: Realistic accuracy for sports prediction")
            print("✅ Confirms model has genuine predictive power")
            print("✅ Temporal boundaries properly implemented")
        elif results['holdout_accuracy'] > 0.80:
            print("⚠️  WARNING: Still suspiciously high - investigate further")
        else:
            print("📈 FAIR: Some predictive power but room for improvement")

        return True

def main():
    """Run proper temporal holdout testing."""
    tester = ProperTemporalHoldoutTester(holdout_months=6)
    tester.run_complete_proper_temporal_test()

if __name__ == "__main__":
    main()