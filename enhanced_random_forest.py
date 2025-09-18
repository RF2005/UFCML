#!/usr/bin/env python3
"""
Enhanced Random Forest for UFC Predictions
==========================================

Improved random forest model using enhanced features for better accuracy.
Implements all the "easy wins" for accuracy improvement:
- Weighted recent performance
- Physical advantages
- Style matchup dynamics
- Temporal features
- Experience differentials

Expected improvement: +3-5% accuracy over baseline models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, brier_score_loss, log_loss
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
from datetime import datetime, timedelta
import sys
sys.path.append('/Users/ralphfrancolini/UFCML')

from .enhanced_feature_engineering import EnhancedFeatureEngineer

class EnhancedUFCRandomForest:
    """Enhanced Random Forest predictor with advanced features."""

    def __init__(self, n_estimators=200, max_depth=4, min_samples_split=60,
                 min_samples_leaf=25, max_features=0.5, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,  # Regularization: use subset of features
            max_samples=0.6,  # Smaller bootstrap to reduce variance and memorization
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        self.feature_engineer = None
        self.feature_columns = None
        self.label_encoders = {}
        self.is_trained = False
        self.permutation_importance_ = None
        self.calibrator_ = None
        self.calibration_info_ = None
        self.calibrator_type = None

    def prepare_features(self, df):
        """Prepare features for training/prediction."""
        # Identify feature columns (exclude metadata and datetime)
        exclude_cols = ['fighter_a', 'fighter_b', 'winner', 'target', 'weight_class', 'date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()

        # Handle categorical features
        categorical_cols = ['style_a', 'style_b']
        for col in categorical_cols:
            if col in X.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Fill missing values
        X = X.fillna(0)

        self.feature_columns = X.columns.tolist()
        return X

    def train(self, df, test_size=0.2, temporal_split=True, validation_size=0.1):
        """Train the enhanced random forest model with proper validation."""
        print("🌳 TRAINING REGULARIZED RANDOM FOREST")
        print("=" * 50)

        print(f"📊 Training data: {len(df)} fights")

        # Prepare features and target
        X = self.prepare_features(df)
        y = df['target']

        print(f"🔢 Features: {len(self.feature_columns)}")

        # Three-way split: train/validation/test
        if temporal_split and 'date' in df.columns:
            # Sort by date and split temporally
            df_sorted = df.sort_values('date')

            # Calculate split indices
            train_end = int(len(df_sorted) * (1 - test_size - validation_size))
            val_end = int(len(df_sorted) * (1 - test_size))

            train_idx = df_sorted.index[:train_end]
            val_idx = df_sorted.index[train_end:val_end]
            test_idx = df_sorted.index[val_end:]

            X_train = X.loc[train_idx]
            X_val = X.loc[val_idx]
            X_test = X.loc[test_idx]
            y_train = y.loc[train_idx]
            y_val = y.loc[val_idx]
            y_test = y.loc[test_idx]

            print("📅 Using temporal split (chronological)")
        else:
            # Random split with validation
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=validation_size/(1-test_size),
                random_state=42, stratify=y_temp
            )
            print("🔀 Using random split")

        print(f"📈 Training set: {len(X_train)} fights")
        print(f"📊 Validation set: {len(X_val)} fights")
        print(f"📉 Test set: {len(X_test)} fights")

        # Train model with regularization
        print("🔄 Training Regularized Random Forest...")
        self.model.fit(X_train, y_train)

        val_proba_raw = self.model.predict_proba(X_val)[:, 1]
        test_proba_raw = self.model.predict_proba(X_test)[:, 1]
        train_proba_raw = self.model.predict_proba(X_train)[:, 1]

        sigmoid_calibrator = LogisticRegression(max_iter=1000, class_weight='balanced')
        sigmoid_calibrator.fit(val_proba_raw.reshape(-1, 1), y_val)
        val_probs_sig = sigmoid_calibrator.predict_proba(val_proba_raw.reshape(-1, 1))[:, 1]
        test_probs_sig = sigmoid_calibrator.predict_proba(test_proba_raw.reshape(-1, 1))[:, 1]
        train_probs_sig = sigmoid_calibrator.predict_proba(train_proba_raw.reshape(-1, 1))[:, 1]

        iso_calibrator = IsotonicRegression(out_of_bounds='clip')
        iso_calibrator.fit(val_proba_raw, y_val)
        val_probs_iso = iso_calibrator.predict(val_proba_raw)
        test_probs_iso = iso_calibrator.predict(test_proba_raw)
        train_probs_iso = iso_calibrator.predict(train_proba_raw)

        def _cal_stats(probs):
            return np.clip(np.column_stack([1 - probs, probs]), 1e-7, 1 - 1e-7)

        stats = {}
        candidate_metrics = {}
        candidate_curves = {}
        for name, (train_probs, val_probs, test_probs) in {
            'sigmoid': (train_probs_sig, val_probs_sig, test_probs_sig),
            'isotonic': (train_probs_iso, val_probs_iso, test_probs_iso),
        }.items():
            stats[name] = {
                'train_proba': _cal_stats(train_probs),
                'val_proba': _cal_stats(val_probs),
                'test_proba': _cal_stats(test_probs),
            }
            candidate_metrics[name] = {
                'val_brier': brier_score_loss(y_val, val_probs),
                'test_brier': brier_score_loss(y_test, test_probs),
                'val_logloss': log_loss(y_val, _cal_stats(val_probs)),
                'test_logloss': log_loss(y_test, _cal_stats(test_probs)),
            }
            frac_pos, mean_pred = calibration_curve(y_val, val_probs, n_bins=10)
            candidate_curves[name] = {
                'frac_pos': frac_pos,
                'mean_pred': mean_pred,
            }

        best_name = min(candidate_metrics, key=lambda n: candidate_metrics[n]['val_brier'])
        self.calibrator_type = best_name
        if best_name == 'sigmoid':
            self.calibrator_ = sigmoid_calibrator
        else:
            self.calibrator_ = iso_calibrator

        print(f"🧮 Calibration method selected: {best_name}")

        chosen = stats[best_name]
        train_proba = chosen['train_proba']
        val_proba_cal = chosen['val_proba']
        test_proba = chosen['test_proba']

        train_pred = (train_proba[:, 1] >= 0.5).astype(int)
        val_pred = (val_proba_cal[:, 1] >= 0.5).astype(int)
        test_pred = (test_proba[:, 1] >= 0.5).astype(int)

        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        val_brier_raw = brier_score_loss(y_val, val_proba_raw)
        val_brier_cal = candidate_metrics[best_name]['val_brier']
        test_brier_cal = candidate_metrics[best_name]['test_brier']
        val_logloss_raw = log_loss(y_val, np.column_stack([1 - val_proba_raw, val_proba_raw]))
        val_logloss_cal = candidate_metrics[best_name]['val_logloss']
        test_logloss_cal = candidate_metrics[best_name]['test_logloss']

        bins = np.linspace(0.0, 1.0, 11)
        bin_counts, _ = np.histogram(val_proba_cal[:, 1], bins=bins)

        self.calibration_info_ = {
            'method': best_name,
            'val_brier_raw': val_brier_raw,
            'val_brier_cal': val_brier_cal,
            'test_brier_cal': test_brier_cal,
            'val_logloss_raw': val_logloss_raw,
            'val_logloss_cal': val_logloss_cal,
            'test_logloss_cal': test_logloss_cal,
            'calibration_curve_frac_pos': candidate_curves[best_name]['frac_pos'],
            'calibration_curve_mean_pred': candidate_curves[best_name]['mean_pred'],
            'calibration_bins': bins,
            'bin_counts': bin_counts,
            'candidate_metrics': candidate_metrics,
            'candidate_curves': candidate_curves,
        }

        print(f"✅ Training accuracy: {train_accuracy:.1%}")
        print(f"📊 Validation accuracy: {val_accuracy:.1%}")
        print(f"🎯 Test accuracy: {test_accuracy:.1%}")
        print(f"⚖️  Val Brier (raw→cal): {val_brier_raw:.3f} → {val_brier_cal:.3f}")
        print(f"⚖️  Test Brier (cal): {test_brier_cal:.3f}")
        print(f"📉 Val LogLoss (raw→cal): {val_logloss_raw:.3f} → {val_logloss_cal:.3f}")
        print(f"📉 Test LogLoss (cal): {test_logloss_cal:.3f}")

        # Detailed overfitting analysis
        train_val_gap = train_accuracy - val_accuracy
        val_test_gap = val_accuracy - test_accuracy
        train_test_gap = train_accuracy - test_accuracy

        print(f"📊 Train-Val gap: {train_val_gap:.1%}")
        print(f"📊 Val-Test gap: {val_test_gap:.1%}")
        print(f"📊 Train-Test gap: {train_test_gap:.1%}")

        # Overfitting detection with stricter thresholds
        if train_accuracy > 0.95:
            print("🚨 SEVERE OVERFITTING: Training accuracy > 95%")
        elif train_val_gap > 0.15:
            print("⚠️  HIGH OVERFITTING: Train-Val gap > 15%")
        elif train_val_gap > 0.08:
            print("⚡ MODERATE OVERFITTING: Train-Val gap > 8%")
        elif train_val_gap > 0.05:
            print("⚖️  SLIGHT OVERFITTING: Train-Val gap > 5%")
        else:
            print("✅ GOOD GENERALIZATION: Low overfitting detected")

        # Sanity check for realistic performance
        if test_accuracy > 0.80:
            print("🤔 WARNING: Test accuracy > 80% - verify no data leakage")
        elif test_accuracy > 0.75:
            print("🎯 EXCELLENT: Test accuracy > 75% - very good model")
        elif test_accuracy > 0.65:
            print("✅ GOOD: Test accuracy > 65% - realistic performance")
        else:
            print("📈 ROOM FOR IMPROVEMENT: Consider feature engineering")

        # Cross-validation
        print("\n🔄 Running 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        print(f"✅ CV Accuracy: {cv_mean:.1%} (±{cv_std:.1%})")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🏆 Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {i+1:2d}. {row['feature']:30} {row['importance']:.3f}")

        # Detailed classification report
        print(f"\n📋 Detailed Test Results:")
        print(classification_report(y_test, test_pred, target_names=['Fighter B Wins', 'Fighter A Wins']))

        # Permutation importance on validation split to confirm feature reliance
        print(f"\n♻️  Permutation Importance (validation set):")
        perm_result = permutation_importance(
            self.model,
            X_val,
            y_val,
            n_repeats=8,
            random_state=42,
            n_jobs=1,  # Single-threaded to avoid sandbox semaphore limits
            scoring='accuracy'
        )

        perm_sorted_idx = perm_result.importances_mean.argsort()[::-1]
        self.permutation_importance_ = {
            'feature': [self.feature_columns[i] for i in perm_sorted_idx],
            'mean': perm_result.importances_mean[perm_sorted_idx],
            'std': perm_result.importances_std[perm_sorted_idx]
        }

        for i in range(min(10, len(self.feature_columns))):
            idx = perm_sorted_idx[i]
            mean_drop = perm_result.importances_mean[idx]
            std_drop = perm_result.importances_std[idx]
            print(f"  {i+1:2d}. {self.feature_columns[idx]:30} Δacc={mean_drop:.3f} ± {std_drop:.3f}")

        self.is_trained = True

        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'cv_accuracy': cv_mean,
            'cv_std': cv_std,
            'feature_importance': feature_importance,
            'train_val_gap': train_val_gap,
            'val_test_gap': val_test_gap,
            'train_test_gap': train_test_gap,
            'permutation_importance': self.permutation_importance_,
            'calibration': self.calibration_info_,
        }

    def predict_fight(self, fighter_a, fighter_b, fight_date=None, title_fight=False, weight_class=None):
        """Predict outcome of a specific fight."""
        if not self.is_trained:
            print("❌ Model not trained yet")
            return None

        if not self.feature_engineer:
            print("❌ Feature engineer not available")
            return None

        # Extract features
        features = self.feature_engineer.extract_enhanced_features(
            fighter_a, fighter_b, fight_date, title_fight, weight_class
        )

        # Convert to DataFrame
        feature_df = pd.DataFrame([features])

        # Prepare features (same as training)
        X = self.prepare_features(feature_df)

        # Ensure feature order matches training
        if set(X.columns) != set(self.feature_columns):
            missing_cols = set(self.feature_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(self.feature_columns)

            # Add missing columns with default values
            for col in missing_cols:
                X[col] = 0

            # Remove extra columns
            X = X.drop(columns=extra_cols, errors='ignore')

        # Reorder columns to match training
        X = X[self.feature_columns]

        raw_prob = self.model.predict_proba(X)[:, 1]
        if self.calibrator_ is not None:
            if self.calibrator_type == 'sigmoid':
                calibrated_prob = self.calibrator_.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
            elif self.calibrator_type == 'isotonic':
                calibrated_prob = self.calibrator_.predict(raw_prob)
            else:
                calibrated_prob = raw_prob
            calibrated_prob = np.clip(calibrated_prob, 1e-7, 1 - 1e-7)
            probability = np.column_stack([1 - calibrated_prob, calibrated_prob])[0]
        else:
            probability = np.column_stack([1 - raw_prob, raw_prob])[0]

        prediction = 1 if probability[1] >= 0.5 else 0

        predicted_winner = fighter_a if prediction == 1 else fighter_b
        confidence = max(probability)

        return {
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'fighter_a_prob': probability[1],
            'fighter_b_prob': probability[0],
            'features_used': features
        }

    def save_model(self, filepath='enhanced_ufc_random_forest.pkl'):
        """Save the trained model."""
        if not self.is_trained:
            print("❌ Cannot save untrained model")
            return False

        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'feature_engineer': self.feature_engineer,
            'calibrator': self.calibrator_,
            'calibrator_type': self.calibrator_type,
            'calibration_info': self.calibration_info_,
            'permutation_importance': self.permutation_importance_,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Enhanced model saved to {filepath}")
        return True

    @classmethod
    def load_model(cls, filepath='enhanced_ufc_random_forest.pkl'):
        """Load a trained model."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            instance = cls()
            instance.model = model_data['model']
            instance.feature_columns = model_data['feature_columns']
            instance.label_encoders = model_data['label_encoders']
            instance.feature_engineer = model_data['feature_engineer']
            instance.calibrator_ = model_data.get('calibrator')
            instance.calibrator_type = model_data.get('calibrator_type')
            instance.calibration_info_ = model_data.get('calibration_info')
            instance.permutation_importance_ = model_data.get('permutation_importance')
            instance.is_trained = True

            print(f"✅ Enhanced model loaded from {filepath}")
            return instance

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None

def train_enhanced_model():
    """Train and save the enhanced UFC random forest model."""
    print("🥊 ENHANCED UFC RANDOM FOREST TRAINING")
    print("=" * 60)

    # Initialize feature engineer
    print("🔧 Initializing enhanced feature engineering...")
    engineer = EnhancedFeatureEngineer()

    if not engineer.load_and_prepare_data():
        print("❌ Failed to load data")
        return

    # Create enhanced dataset
    enhanced_df = engineer.create_enhanced_training_data()

    # Initialize and train model with optimized parameters
    model = EnhancedUFCRandomForest(
        n_estimators=200,   # Many trees to smooth predictions
        max_depth=4,        # Shallower trees for stronger regularization
        min_samples_split=60,  # Require substantial evidence before splitting
        min_samples_leaf=25,   # Larger leaves reduce variance and memorization
        max_features=0.5       # Restrict per-tree feature usage
    )

    # Attach feature engineer to model
    model.feature_engineer = engineer

    # Train model
    results = model.train(enhanced_df, temporal_split=True)

    # Save model
    model.save_model('models/enhanced_ufc_random_forest.pkl')

    # Test prediction
    print(f"\n🧪 TESTING SAMPLE PREDICTION")
    print("=" * 50)

    sample_result = model.predict_fight(
        "Jon Jones", "Stipe Miocic",
        title_fight=True, weight_class="Heavyweight"
    )

    if sample_result:
        print(f"🥊 Test Fight: Jon Jones vs Stipe Miocic")
        print(f"🏆 Prediction: {sample_result['predicted_winner']}")
        print(f"📊 Confidence: {sample_result['confidence']:.1%}")
        print(f"⚔️  Jon Jones probability: {sample_result['fighter_a_prob']:.1%}")
        print(f"⚔️  Stipe Miocic probability: {sample_result['fighter_b_prob']:.1%}")

    return model, results

def main():
    """Main training function."""
    model, results = train_enhanced_model()

    print(f"\n" + "="*60)
    print("🎯 ENHANCED RANDOM FOREST SUMMARY")
    print("="*60)
    print(f"✅ Model trained successfully!")
    print(f"📊 Test Accuracy: {results['test_accuracy']:.1%}")
    print(f"🎯 CV Accuracy: {results['cv_accuracy']:.1%} (±{results['cv_std']:.1%})")
    print(f"📈 Train-Test Gap: {results['train_test_gap']:.1%}")
    if results.get('calibration'):
        cal = results['calibration']
        print(f"🧮 Calibration: {cal.get('method', 'n/a')}")
        print(f"⚖️  Test Brier (calibrated): {cal['test_brier_cal']:.3f}")
        print(f"📉 Test LogLoss (calibrated): {cal['test_logloss_cal']:.3f}")
    print(f"💾 Model saved to: models/enhanced_ufc_random_forest.pkl")

    print(f"\n🚀 Ready for enhanced predictions!")
    print(f"Expected improvement: +3-5% over baseline models")

if __name__ == "__main__":
    main()
