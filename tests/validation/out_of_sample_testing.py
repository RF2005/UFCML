#!/usr/bin/env python3
"""
Out-of-Sample Testing Framework for UFC Predictions
===================================================

Creates true holdout test sets and validates model performance on completely unseen data.
This addresses the key question: "Do we have out of sample data to test the program on?"

The framework:
1. Creates a strict temporal holdout (last 6 months never used in training)
2. Tests the model on this truly unseen data
3. Provides validation for real-world deployment confidence
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import pickle
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append('/Users/ralphfrancolini/UFCML')
from src.core.advanced_ml_models import load_enhanced_ufc_data
from enhanced_feature_engineering import EnhancedFeatureEngineer
from enhanced_random_forest import EnhancedUFCRandomForest

class OutOfSampleTester:
    """Framework for testing UFC models on truly unseen out-of-sample data."""

    def __init__(self, holdout_months=6):
        self.holdout_months = holdout_months
        self.df = None
        self.training_data = None
        self.holdout_data = None
        self.model = None
        self.feature_engineer = None

    def load_and_split_data(self):
        """Load data and create temporal split for out-of-sample testing."""
        print("📊 LOADING DATA FOR OUT-OF-SAMPLE TESTING")
        print("=" * 50)

        # Load full dataset
        self.df = load_enhanced_ufc_data()
        if self.df is None:
            print("❌ Failed to load UFC data")
            return False

        # Clean and prepare
        self.df = self.df.dropna(subset=['winner', 'date'])
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df = self.df.dropna(subset=['date']).sort_values('date')

        # Create temporal split - holdout is NEVER seen during training/validation
        cutoff_date = self.df['date'].max() - timedelta(days=self.holdout_months * 30)

        self.training_data = self.df[self.df['date'] <= cutoff_date].copy()
        self.holdout_data = self.df[self.df['date'] > cutoff_date].copy()

        print(f"📅 Dataset date range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}")
        print(f"✂️  Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
        print(f"📈 Training data: {len(self.training_data)} fights ({len(self.training_data)/len(self.df)*100:.1f}%)")
        print(f"🔒 Holdout data: {len(self.holdout_data)} fights ({len(self.holdout_data)/len(self.df)*100:.1f}%)")

        # Verify temporal separation
        latest_training = self.training_data['date'].max()
        earliest_holdout = self.holdout_data['date'].min()
        gap_days = (earliest_holdout - latest_training).days

        print(f"⏰ Temporal gap: {gap_days} days (prevents data leakage)")

        if gap_days < 0:
            print("❌ ERROR: Temporal overlap detected!")
            return False

        if len(self.holdout_data) < 50:
            print("⚠️  WARNING: Small holdout set may give unreliable results")

        return True

    def train_model_on_training_data(self):
        """Train model ONLY on training data, never seeing holdout."""
        print(f"\n🎯 TRAINING MODEL ON TRAINING DATA ONLY")
        print("=" * 50)

        # Initialize feature engineer with training data only
        self.feature_engineer = EnhancedFeatureEngineer()

        # Use only training data for feature engineering
        original_df = self.feature_engineer.df
        self.feature_engineer.df = self.training_data.copy()

        if not self.feature_engineer.load_and_prepare_data():
            print("❌ Failed to prepare training data")
            return False

        # Create training features (fighter profiles built only from training data)
        print("🔄 Building fighter profiles from training data only...")
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

        # Train with temporal split (but only on training data)
        print("🔄 Training enhanced random forest...")
        results = self.model.train(training_enhanced, temporal_split=True)

        print(f"✅ Model trained successfully")
        print(f"📊 Training accuracy: {results['train_accuracy']:.1%}")
        print(f"📊 Validation accuracy: {results['val_accuracy']:.1%}")
        print(f"📊 Test accuracy: {results['test_accuracy']:.1%}")
        print(f"⚖️  Overfitting gap: {results['train_val_gap']:.1%}")

        return True

    def test_on_holdout_data(self):
        """Test the trained model on completely unseen holdout data."""
        print(f"\n🔒 TESTING ON OUT-OF-SAMPLE HOLDOUT DATA")
        print("=" * 50)

        if not self.model or not self.model.is_trained:
            print("❌ Model not trained yet")
            return None

        holdout_predictions = []
        holdout_actuals = []
        prediction_details = []

        print(f"🧪 Predicting {len(self.holdout_data)} out-of-sample fights...")

        for idx, fight in self.holdout_data.iterrows():
            try:
                fighter_a = fight['r_name']
                fighter_b = fight['b_name']
                actual_winner = fight['winner']
                fight_date = fight['date']

                # Extract features for this holdout fight
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
                print(f"⚠️  Error predicting fight {fighter_a} vs {fighter_b}: {e}")
                continue

        # Calculate out-of-sample performance
        oos_accuracy = accuracy_score(holdout_actuals, holdout_predictions)

        print(f"\n🎯 OUT-OF-SAMPLE RESULTS")
        print("=" * 50)
        print(f"✅ Holdout accuracy: {oos_accuracy:.1%}")
        print(f"📊 Predictions made: {len(prediction_details)}")
        print(f"✅ Successful predictions: {sum(1 for p in prediction_details if p['correct'])}")

        # Detailed analysis
        print(f"\n📋 Classification Report:")
        print(classification_report(holdout_actuals, holdout_predictions,
                                  target_names=['Fighter B Wins', 'Fighter A Wins']))

        # Show some examples
        print(f"\n🔍 Sample Holdout Predictions:")
        for i, pred in enumerate(prediction_details[:10]):
            status = "✅" if pred['correct'] else "❌"
            print(f"  {status} {pred['date'].strftime('%Y-%m-%d')}: {pred['fighter_a']} vs {pred['fighter_b']}")
            print(f"      Predicted: {pred['predicted_winner']} ({pred['confidence']:.1%})")
            print(f"      Actual: {pred['actual_winner']}")

        if len(prediction_details) > 10:
            print(f"      ... and {len(prediction_details) - 10} more")

        # Performance assessment
        print(f"\n📊 OUT-OF-SAMPLE ASSESSMENT")
        print("=" * 50)

        if oos_accuracy > 0.75:
            print("🏆 EXCELLENT: Model generalizes very well to unseen data")
        elif oos_accuracy > 0.65:
            print("✅ GOOD: Model shows solid generalization")
        elif oos_accuracy > 0.55:
            print("⚠️  FAIR: Model has some predictive power but room for improvement")
        else:
            print("❌ POOR: Model may be overfitting or insufficient")

        return {
            'holdout_accuracy': oos_accuracy,
            'predictions': prediction_details,
            'actuals': holdout_actuals,
            'predicted': holdout_predictions
        }

    def run_complete_out_of_sample_test(self):
        """Run the complete out-of-sample testing pipeline."""
        print("🧪 COMPLETE OUT-OF-SAMPLE TESTING FRAMEWORK")
        print("=" * 60)

        # Step 1: Load and split data
        if not self.load_and_split_data():
            return False

        # Step 2: Train model only on training data
        if not self.train_model_on_training_data():
            return False

        # Step 3: Test on holdout data
        results = self.test_on_holdout_data()

        if results is None:
            return False

        # Step 4: Save results for future reference
        results_summary = {
            'holdout_months': self.holdout_months,
            'training_fights': len(self.training_data),
            'holdout_fights': len(self.holdout_data),
            'holdout_accuracy': results['holdout_accuracy'],
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open('out_of_sample_results.pkl', 'wb') as f:
            pickle.dump(results_summary, f)

        print(f"\n💾 Results saved to 'out_of_sample_results.pkl'")
        print(f"🎯 OUT-OF-SAMPLE ACCURACY: {results['holdout_accuracy']:.1%}")

        return True

def main():
    """Run out-of-sample testing."""
    tester = OutOfSampleTester(holdout_months=6)
    tester.run_complete_out_of_sample_test()

if __name__ == "__main__":
    main()