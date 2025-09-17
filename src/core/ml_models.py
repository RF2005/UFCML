"""
Machine Learning Models for UFC Fight Prediction
================================================

Implements decision trees and random forests using Elo-based features
to predict UFC fight outcomes.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle


def create_elo_features(fight_results):
    """
    Create machine learning features from fight results and Elo ratings.

    Args:
        fight_results (list): List of fight result dictionaries

    Returns:
        tuple: (features_array, labels_array, feature_names)
    """
    df = pd.DataFrame(fight_results)
    features = []
    labels = []

    for _, row in df.iterrows():
        # Elo-based features
        elo_diff = row['rating_a_before'] - row['rating_b_before']
        elo_a = row['rating_a_before']
        elo_b = row['rating_b_before']
        avg_elo = (elo_a + elo_b) / 2
        max_elo = max(elo_a, elo_b)
        min_elo = min(elo_a, elo_b)

        # Expected probability difference
        prob_diff = row['expected_prob_a'] - row['expected_prob_b']

        features.append([
            elo_diff,           # Elo rating difference
            elo_a,              # Fighter A's Elo
            elo_b,              # Fighter B's Elo
            avg_elo,            # Average Elo of both fighters
            max_elo,            # Higher Elo rating
            min_elo,            # Lower Elo rating
            prob_diff,          # Expected probability difference
            row['expected_prob_a']  # Expected probability for fighter A
        ])

        # Label: 1 if fighter A won, 0 if fighter B won
        if row['winner'] == row['fighter_a']:
            labels.append(1)
        else:
            labels.append(0)

    # Feature names for interpretability
    feature_names = [
        'Elo_Difference',
        'Fighter_A_Elo',
        'Fighter_B_Elo',
        'Average_Elo',
        'Max_Elo',
        'Min_Elo',
        'Probability_Difference',
        'Expected_Prob_A'
    ]

    return np.array(features), np.array(labels), feature_names


def create_elo_decision_tree(fight_results, max_depth=6, min_samples_split=20, min_samples_leaf=10, save_model=True):
    """
    Create and train a decision tree using Elo-based features.

    Args:
        fight_results (list): List of fight result dictionaries
        max_depth (int): Maximum depth of the decision tree
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required in a leaf node
        save_model (bool): Whether to save the trained model

    Returns:
        tuple: (trained_model, feature_names, accuracy, test_predictions)
    """
    # Create features and labels
    X, y, feature_names = create_elo_features(fight_results)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train decision tree
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    dt.fit(X_train, y_train)

    # Make predictions
    y_pred = dt.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nDecision Tree Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Feature importance
    importance = dt.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print(f"\nFeature Importance:")
    print(feature_importance_df.to_string(index=False))

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fighter B Wins', 'Fighter A Wins']))

    if save_model:
        # Save the decision tree model
        joblib.dump(dt, 'models/ufc_elo_decision_tree.pkl')
        print(f"\nDecision tree saved as 'models/ufc_elo_decision_tree.pkl'")

        # Save the feature names and data for later use
        model_data = {
            'decision_tree': dt,
            'feature_names': feature_names,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'accuracy': accuracy,
            'feature_importance': feature_importance_df
        }

        with open('models/ufc_model_data.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model data saved as 'models/ufc_model_data.pkl'")

    return dt, feature_names, accuracy, {'y_test': y_test, 'y_pred': y_pred}


def create_elo_random_forest(fight_results, n_estimators=100, max_depth=6, min_samples_split=20,
                            min_samples_leaf=10, save_model=True):
    """
    Create and train a random forest using Elo-based features.

    Args:
        fight_results (list): List of fight result dictionaries
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of each tree
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required in a leaf node
        save_model (bool): Whether to save the trained model

    Returns:
        tuple: (trained_model, feature_names, accuracy, test_predictions)
    """
    # Create features and labels
    X, y, feature_names = create_elo_features(fight_results)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nRandom Forest Results ({n_estimators} trees):")
    print(f"Accuracy: {accuracy:.3f}")

    # Feature importance
    importance = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print(f"\nRandom Forest Feature Importance:")
    print(feature_importance_df.to_string(index=False))

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fighter B Wins', 'Fighter A Wins']))

    if save_model:
        # Save Random Forest
        joblib.dump(rf, 'models/ufc_elo_random_forest.pkl')
        print(f"\nRandom Forest saved as 'models/ufc_elo_random_forest.pkl'")

    return rf, feature_names, accuracy, {'y_test': y_test, 'y_pred': y_pred}


def load_saved_model(model_path):
    """
    Load a previously saved model.

    Args:
        model_path (str): Path to the saved model file

    Returns:
        sklearn model: Loaded model
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def predict_fight_outcome(model, fighter_a_elo, fighter_b_elo, feature_names):
    """
    Predict the outcome of a fight given two fighters' Elo ratings.

    Args:
        model: Trained sklearn model
        fighter_a_elo (float): Fighter A's Elo rating
        fighter_b_elo (float): Fighter B's Elo rating
        feature_names (list): List of feature names used by the model

    Returns:
        dict: Prediction results including probability and winner
    """
    # Calculate expected probabilities
    expected_a = 1 / (1 + 10**((fighter_b_elo - fighter_a_elo) / 400))
    expected_b = 1 / (1 + 10**((fighter_a_elo - fighter_b_elo) / 400))

    # Create features
    elo_diff = fighter_a_elo - fighter_b_elo
    avg_elo = (fighter_a_elo + fighter_b_elo) / 2
    max_elo = max(fighter_a_elo, fighter_b_elo)
    min_elo = min(fighter_a_elo, fighter_b_elo)
    prob_diff = expected_a - expected_b

    features = np.array([[
        elo_diff, fighter_a_elo, fighter_b_elo, avg_elo,
        max_elo, min_elo, prob_diff, expected_a
    ]])

    # Make prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    result = {
        'predicted_winner': 'Fighter A' if prediction == 1 else 'Fighter B',
        'fighter_a_win_prob': probabilities[1],
        'fighter_b_win_prob': probabilities[0],
        'elo_expected_prob_a': expected_a,
        'elo_expected_prob_b': expected_b,
        'confidence': max(probabilities)
    }

    return result