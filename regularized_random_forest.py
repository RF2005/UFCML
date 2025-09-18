#!/usr/bin/env python3
"""
Regularized Random Forest for UFC Prediction
============================================

Implements feature importance regularization to prevent single features
from dominating the model and causing memorization.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
from datetime import datetime, timedelta
import sys
sys.path.append('/Users/ralphfrancolini/UFCML')

from .enhanced_feature_engineering import EnhancedFeatureEngineer

class RegularizedUFCRandomForest:
    """Random Forest with feature importance regularization to prevent memorization."""

    def __init__(self, n_estimators=200, max_depth=4, min_samples_split=50,
                 min_samples_leaf=20, max_features=0.6, random_state=42,
                 max_feature_importance=0.15):
        """
        Initialize with aggressive regularization parameters.

        Args:
            max_feature_importance: Maximum allowed importance for any single feature (0.20 = 20%)
        """
        self.max_feature_importance = max_feature_importance

        # More conservative parameters to reduce memorization
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,     # More trees for stability
            max_depth=max_depth,           # Shallower trees
            min_samples_split=min_samples_split,  # More samples required
            min_samples_leaf=min_samples_leaf,    # Larger leaves
            max_features=max_features,     # Use only 60% of features
            max_samples=0.6,              # Smaller bootstrap samples
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True
        )

        self.feature_engineer = None
        self.feature_columns = None
        self.label_encoders = {}
        self.is_trained = False

    def create_balanced_feature_subsets(self, X, y, n_subsets=5):
        """Create multiple feature subsets to distribute importance."""
        n_features = X.shape[1]
        subset_size = max(int(n_features * 0.6), min(12, n_features))  # Use ~60% of features per subset

        subsets = []
        for i in range(n_subsets):
            # Create different feature combinations
            np.random.seed(42 + i)
            feature_indices = np.random.choice(n_features, subset_size, replace=False)
            subsets.append(feature_indices)

        return subsets

    def train_with_feature_regularization(self, df, test_size=0.2, temporal_split=True, validation_size=0.1):
        """Train with feature importance regularization."""
        print("🛡️  TRAINING REGULARIZED RANDOM FOREST")
        print("=" * 50)
        print(f"🎯 Max feature importance limit: {self.max_feature_importance:.1%}")

        print(f"📊 Training data: {len(df)} fights")

        # Prepare features and target
        X = self.prepare_features(df)
        y = df['target']

        print(f"🔢 Features: {len(self.feature_columns)}")

        # Temporal split like before
        if temporal_split and 'date' in df.columns:
            df_sorted = df.sort_values('date')
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

        # Train multiple models with feature subsets
        print("🔄 Training with feature regularization...")

        # Create feature subsets
        feature_subsets = self.create_balanced_feature_subsets(X_train, y_train)

        # Train ensemble of models with different feature subsets
        models = []
        for i, feature_subset in enumerate(feature_subsets):
            print(f"  🌲 Training subset {i+1}/{len(feature_subsets)} ({len(feature_subset)} features)")

            subset_model = RandomForestClassifier(
                n_estimators=30,  # Fewer trees per subset
                max_depth=self.model.max_depth,
                min_samples_split=self.model.min_samples_split,
                min_samples_leaf=self.model.min_samples_leaf,
                max_features='sqrt',
                random_state=42 + i,
                n_jobs=-1,
                class_weight='balanced'
            )

            X_subset = X_train.iloc[:, feature_subset]
            subset_model.fit(X_subset, y_train)
            models.append((subset_model, feature_subset))

        # Store the ensemble
        self.ensemble_models = models

        # Also train main model for comparison
        self.model.fit(X_train, y_train)

        # Evaluate both approaches
        print("\n📊 EVALUATION COMPARISON:")

        # Main model
        train_acc_main = self.model.score(X_train, y_train)
        val_acc_main = self.model.score(X_val, y_val)
        test_acc_main = self.model.score(X_test, y_test)

        # Ensemble model
        ensemble_train_pred = self.predict_ensemble(X_train)
        ensemble_val_pred = self.predict_ensemble(X_val)
        ensemble_test_pred = self.predict_ensemble(X_test)

        train_acc_ensemble = accuracy_score(y_train, ensemble_train_pred)
        val_acc_ensemble = accuracy_score(y_val, ensemble_val_pred)
        test_acc_ensemble = accuracy_score(y_test, ensemble_test_pred)

        print(f"📈 Main Model    - Train: {train_acc_main:.3f} | Val: {val_acc_main:.3f} | Test: {test_acc_main:.3f}")
        print(f"🛡️  Ensemble     - Train: {train_acc_ensemble:.3f} | Val: {val_acc_ensemble:.3f} | Test: {test_acc_ensemble:.3f}")

        # Check feature importance distribution
        main_importances = self.model.feature_importances_
        max_importance = np.max(main_importances)

        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
        print(f"📊 Main model max feature importance: {max_importance:.1%}")

        if max_importance > self.max_feature_importance:
            print(f"🚨 USING ENSEMBLE: Max importance {max_importance:.1%} > {self.max_feature_importance:.1%}")
            self.use_ensemble = True
        else:
            print(f"✅ USING MAIN MODEL: Max importance {max_importance:.1%} ≤ {self.max_feature_importance:.1%}")
            self.use_ensemble = False

        self.is_trained = True

        # Return results
        if self.use_ensemble:
            return {
                'train_accuracy': train_acc_ensemble,
                'val_accuracy': val_acc_ensemble,
                'test_accuracy': test_acc_ensemble,
                'model_type': 'ensemble'
            }
        else:
            return {
                'train_accuracy': train_acc_main,
                'val_accuracy': val_acc_main,
                'test_accuracy': test_acc_main,
                'model_type': 'main'
            }

    def predict_ensemble(self, X):
        """Make predictions using ensemble of feature subsets."""
        if not hasattr(self, 'ensemble_models'):
            return self.model.predict(X)

        predictions = []
        for model, feature_subset in self.ensemble_models:
            X_subset = X.iloc[:, feature_subset]
            pred = model.predict(X_subset)
            predictions.append(pred)

        # Majority voting
        predictions = np.array(predictions)
        ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return ensemble_pred

    def predict_fight(self, fighter_a, fighter_b, fight_date=None, title_fight=False, weight_class=None):
        """Predict fight outcome using regularized model."""
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
        X = self.prepare_features(feature_df)

        # Ensure feature order matches training
        if set(X.columns) != set(self.feature_columns):
            missing_cols = set(self.feature_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(self.feature_columns)

            for col in missing_cols:
                X[col] = 0
            X = X.drop(columns=extra_cols, errors='ignore')

        X = X[self.feature_columns]

        # Make prediction using appropriate model
        if hasattr(self, 'use_ensemble') and self.use_ensemble:
            prediction = self.predict_ensemble(X)[0]

            # Get probabilities from ensemble average
            probs = []
            for model, feature_subset in self.ensemble_models:
                X_subset = X.iloc[:, feature_subset]
                prob = model.predict_proba(X_subset)[0]
                probs.append(prob)
            probability = np.mean(probs, axis=0)
        else:
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]

        predicted_winner = fighter_a if prediction == 1 else fighter_b
        confidence = max(probability)

        return {
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'fighter_a_prob': probability[1],
            'fighter_b_prob': probability[0],
            'features_used': features
        }

    def prepare_features(self, df):
        """Prepare features for training (same as enhanced model)."""
        feature_df = df.copy()

        # Remove metadata columns that should not be used as features
        drop_cols = ['fighter_a', 'fighter_b', 'winner', 'target', 'date']
        feature_df = feature_df.drop(columns=[col for col in drop_cols if col in feature_df.columns], errors='ignore')

        # Encode categorical features (style and weight class)
        categorical_columns = feature_df.select_dtypes(include=['object']).columns

        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                feature_df[column] = self.label_encoders[column].fit_transform(
                    feature_df[column].astype(str)
                )
            else:
                existing_classes = list(self.label_encoders[column].classes_)
                new_values = [val for val in feature_df[column].astype(str).unique() if val not in existing_classes]
                if new_values:
                    updated_classes = existing_classes + new_values
                    self.label_encoders[column].classes_ = np.array(updated_classes)
                feature_df[column] = self.label_encoders[column].transform(
                    feature_df[column].astype(str)
                )

        feature_df = feature_df.fillna(0)

        # Persist column order for downstream splits/predictions
        if self.feature_columns is None:
            self.feature_columns = feature_df.columns.tolist()
        else:
            missing_cols = [col for col in self.feature_columns if col not in feature_df.columns]
            for col in missing_cols:
                feature_df[col] = 0

            extra_cols = [col for col in feature_df.columns if col not in self.feature_columns]
            feature_df = feature_df.drop(columns=extra_cols, errors='ignore')
            feature_df = feature_df[self.feature_columns]

        return feature_df

    def save_model(self, filepath='regularized_ufc_random_forest.pkl'):
        """Save the regularized model."""
        if not self.is_trained:
            print("❌ Cannot save untrained model")
            return False

        model_data = {
            'model': self.model,
            'ensemble_models': getattr(self, 'ensemble_models', None),
            'use_ensemble': getattr(self, 'use_ensemble', False),
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'feature_engineer': self.feature_engineer,
            'max_feature_importance': self.max_feature_importance
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Regularized model saved to {filepath}")
        return True

    @classmethod
    def load_model(cls, filepath):
        """Load a saved regularized model."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            instance = cls()
            instance.model = model_data['model']
            instance.ensemble_models = model_data.get('ensemble_models')
            instance.use_ensemble = model_data.get('use_ensemble', False)
            instance.feature_columns = model_data['feature_columns']
            instance.label_encoders = model_data['label_encoders']
            instance.feature_engineer = model_data['feature_engineer']
            instance.max_feature_importance = model_data.get('max_feature_importance', 0.20)
            instance.is_trained = True

            print(f"✅ Regularized model loaded from {filepath}")
            return instance

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None

def train_regularized_model():
    """Train and save the regularized UFC random forest model."""
    print("🛡️  REGULARIZED UFC RANDOM FOREST TRAINING")
    print("=" * 60)

    # Initialize feature engineer
    print("🔧 Initializing enhanced feature engineering...")
    engineer = EnhancedFeatureEngineer()

    if not engineer.load_and_prepare_data():
        print("❌ Failed to load data")
        return

    # Create enhanced dataset
    enhanced_df = engineer.create_enhanced_training_data()

    # Initialize and train regularized model
    model = RegularizedUFCRandomForest(
        n_estimators=200,        # Many trees to stabilize subset voting
        max_depth=4,             # Keep trees shallow to curb variance
        min_samples_split=50,    # Require substantial evidence to split
        min_samples_leaf=20,     # Larger leaves improve generalization
        max_features=0.6,        # Use 60% of features per tree
        max_feature_importance=0.15  # Tighter importance ceiling
    )

    # Attach feature engineer to model
    model.feature_engineer = engineer

    # Train with regularization
    results = model.train_with_feature_regularization(enhanced_df)

    # Analyze final feature importance
    if hasattr(model, 'use_ensemble') and model.use_ensemble:
        print("\n🛡️  ENSEMBLE MODEL FEATURE ANALYSIS:")
        print("Feature importance distributed across multiple subsets")
    else:
        print("\n📊 MAIN MODEL FEATURE IMPORTANCE:")
        importances = model.model.feature_importances_
        feature_names = model.feature_columns
        sorted_idx = np.argsort(importances)[::-1]

        for i in range(min(10, len(importances))):
            idx = sorted_idx[i]
            print(f"  {i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.3f} ({importances[idx]*100:.1f}%)")

    # Save model
    model.save_model('models/regularized_ufc_random_forest.pkl')

    print(f"\n" + "="*60)
    print("🛡️  REGULARIZED RANDOM FOREST SUMMARY")
    print("="*60)
    print(f"✅ Model trained successfully!")
    print(f"📊 Test Accuracy: {results['test_accuracy']:.1%}")
    print(f"🛡️  Model Type: {results['model_type']}")
    print(f"🎯 Max Feature Importance: {model.max_feature_importance:.1%}")
    print(f"💾 Model saved to: models/regularized_ufc_random_forest.pkl")

    return model, results

if __name__ == "__main__":
    train_regularized_model()
