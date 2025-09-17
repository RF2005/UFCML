"""
Advanced Machine Learning Models for UFC Fight Prediction
=========================================================

This module implements specialized decision trees for different aspects of UFC fights:
- Striking statistics trees
- Grappling statistics trees
- Fight context trees (title fights, method, timing)
- Positional statistics trees (head/body/leg, distance/clinch/ground)

These trees can be combined into custom ensemble models for improved predictions.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle


def load_enhanced_ufc_data(file_path='/Users/ralphfrancolini/Desktop/ufc_data.csv'):
    """
    Load UFC dataset with all the detailed fight statistics.

    Args:
        file_path (str): Path to the UFC dataset CSV file

    Returns:
        pandas.DataFrame or None: Loaded dataset with all fight statistics
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)

        # Clean critical columns first
        df = df.dropna(subset=['r_name', 'b_name', 'winner'])
        df = df[df['winner'].astype(str).str.lower() != 'nan']

        # Convert categorical columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Convert numeric columns
        numeric_columns = [
            'title_fight', 'finish_round', 'match_time_sec', 'total_rounds',
            'r_kd', 'r_sig_str_landed', 'r_sig_str_atmpted', 'r_sig_str_acc',
            'r_total_str_landed', 'r_total_str_atmpted', 'r_total_str_acc',
            'r_td_landed', 'r_td_atmpted', 'r_td_acc', 'r_sub_att', 'r_ctrl',
            'r_head_landed', 'r_head_atmpted', 'r_head_acc',
            'r_body_landed', 'r_body_atmpted', 'r_body_acc',
            'r_leg_landed', 'r_leg_atmpted', 'r_leg_acc',
            'r_dist_landed', 'r_dist_atmpted', 'r_dist_acc',
            'r_clinch_landed', 'r_clinch_atmpted', 'r_clinch_acc',
            'r_ground_landed', 'r_ground_atmpted', 'r_ground_acc',
            'r_landed_head_per', 'r_landed_body_per', 'r_landed_leg_per',
            'r_landed_dist_per', 'r_landed_clinch_per', 'r_landed_ground_per',
            'b_kd', 'b_sig_str_landed', 'b_sig_str_atmpted', 'b_sig_str_acc',
            'b_total_str_landed', 'b_total_str_atmpted', 'b_total_str_acc',
            'b_td_landed', 'b_td_atmpted', 'b_td_acc', 'b_sub_att', 'b_ctrl',
            'b_head_landed', 'b_head_atmpted', 'b_head_acc',
            'b_body_landed', 'b_body_atmpted', 'b_body_acc',
            'b_leg_landed', 'b_leg_atmpted', 'b_leg_acc',
            'b_dist_landed', 'b_dist_atmpted', 'b_dist_acc',
            'b_clinch_landed', 'b_clinch_atmpted', 'b_clinch_acc',
            'b_ground_landed', 'b_ground_atmpted', 'b_ground_acc',
            'b_landed_head_per', 'b_landed_body_per', 'b_landed_leg_per',
            'b_landed_dist_per', 'b_landed_clinch_per', 'b_landed_ground_per'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except FileNotFoundError:
        print(f"Could not find UFC dataset at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading UFC dataset: {e}")
        return None


def create_striking_features(df):
    """
    Create features focused on striking statistics.

    Args:
        df (pandas.DataFrame): UFC fight dataset

    Returns:
        tuple: (features_array, labels_array, feature_names)
    """
    features = []
    labels = []

    for _, row in df.iterrows():
        # Skip rows with missing critical data
        if pd.isna(row.get('r_name')) or pd.isna(row.get('b_name')) or pd.isna(row.get('winner')):
            continue

        # Striking accuracy and volume differences
        r_sig_acc = row.get('r_sig_str_acc', 0) or 0
        b_sig_acc = row.get('b_sig_str_acc', 0) or 0
        r_sig_landed = row.get('r_sig_str_landed', 0) or 0
        b_sig_landed = row.get('b_sig_str_landed', 0) or 0
        r_sig_attempted = row.get('r_sig_str_atmpted', 0) or 0
        b_sig_attempted = row.get('b_sig_str_atmpted', 0) or 0

        # Total striking statistics
        r_total_acc = row.get('r_total_str_acc', 0) or 0
        b_total_acc = row.get('b_total_str_acc', 0) or 0
        r_total_landed = row.get('r_total_str_landed', 0) or 0
        b_total_landed = row.get('b_total_str_landed', 0) or 0

        # Knockdowns
        r_kd = row.get('r_kd', 0) or 0
        b_kd = row.get('b_kd', 0) or 0

        striking_features = [
            # Accuracy differences
            r_sig_acc - b_sig_acc,
            r_total_acc - b_total_acc,

            # Volume differences
            r_sig_landed - b_sig_landed,
            r_total_landed - b_total_landed,
            r_sig_attempted - b_sig_attempted,

            # Knockdown difference
            r_kd - b_kd,

            # Individual stats
            r_sig_acc, b_sig_acc,
            r_sig_landed, b_sig_landed,
            r_total_acc, b_total_acc,
            r_kd, b_kd,

            # Ratios and efficiency
            r_sig_landed / max(r_sig_attempted, 1),  # Efficiency
            b_sig_landed / max(b_sig_attempted, 1),
            (r_sig_landed + r_total_landed) / 2,  # Combined volume
            (b_sig_landed + b_total_landed) / 2
        ]

        features.append(striking_features)

        # Label: 1 if red corner won, 0 if blue corner won
        labels.append(1 if row['winner'] == row['r_name'] else 0)

    feature_names = [
        'Sig_Accuracy_Diff', 'Total_Accuracy_Diff',
        'Sig_Landed_Diff', 'Total_Landed_Diff', 'Sig_Attempted_Diff',
        'Knockdown_Diff',
        'R_Sig_Acc', 'B_Sig_Acc',
        'R_Sig_Landed', 'B_Sig_Landed',
        'R_Total_Acc', 'B_Total_Acc',
        'R_Knockdowns', 'B_Knockdowns',
        'R_Striking_Efficiency', 'B_Striking_Efficiency',
        'R_Combined_Volume', 'B_Combined_Volume'
    ]

    return np.array(features), np.array(labels), feature_names


def create_grappling_features(df):
    """
    Create features focused on grappling and ground game statistics.

    Args:
        df (pandas.DataFrame): UFC fight dataset

    Returns:
        tuple: (features_array, labels_array, feature_names)
    """
    features = []
    labels = []

    for _, row in df.iterrows():
        # Skip rows with missing critical data
        if pd.isna(row.get('r_name')) or pd.isna(row.get('b_name')) or pd.isna(row.get('winner')):
            continue

        # Takedown statistics
        r_td_landed = row.get('r_td_landed', 0) or 0
        b_td_landed = row.get('b_td_landed', 0) or 0
        r_td_acc = row.get('r_td_acc', 0) or 0
        b_td_acc = row.get('b_td_acc', 0) or 0
        r_td_attempted = row.get('r_td_atmpted', 0) or 0
        b_td_attempted = row.get('b_td_atmpted', 0) or 0

        # Submission attempts
        r_sub_att = row.get('r_sub_att', 0) or 0
        b_sub_att = row.get('b_sub_att', 0) or 0

        # Control time
        r_ctrl = row.get('r_ctrl', 0) or 0
        b_ctrl = row.get('b_ctrl', 0) or 0

        grappling_features = [
            # Takedown differences
            r_td_landed - b_td_landed,
            r_td_acc - b_td_acc,
            r_td_attempted - b_td_attempted,

            # Submission attempts difference
            r_sub_att - b_sub_att,

            # Control time difference
            r_ctrl - b_ctrl,

            # Individual grappling stats
            r_td_landed, b_td_landed,
            r_td_acc, b_td_acc,
            r_sub_att, b_sub_att,
            r_ctrl, b_ctrl,

            # Grappling efficiency metrics
            r_td_landed / max(r_td_attempted, 1),  # TD success rate
            b_td_landed / max(b_td_attempted, 1),
            (r_ctrl + r_td_landed * 30) / max(300, 1),  # Ground control index
            (b_ctrl + b_td_landed * 30) / max(300, 1)
        ]

        features.append(grappling_features)
        labels.append(1 if row['winner'] == row['r_name'] else 0)

    feature_names = [
        'TD_Landed_Diff', 'TD_Acc_Diff', 'TD_Attempted_Diff',
        'Sub_Attempts_Diff', 'Control_Time_Diff',
        'R_TD_Landed', 'B_TD_Landed',
        'R_TD_Acc', 'B_TD_Acc',
        'R_Sub_Att', 'B_Sub_Att',
        'R_Control', 'B_Control',
        'R_TD_Success_Rate', 'B_TD_Success_Rate',
        'R_Ground_Control_Index', 'B_Ground_Control_Index'
    ]

    return np.array(features), np.array(labels), feature_names


def create_positional_features(df):
    """
    Create features focused on positional striking (head/body/leg, distance/clinch/ground).

    Args:
        df (pandas.DataFrame): UFC fight dataset

    Returns:
        tuple: (features_array, labels_array, feature_names)
    """
    features = []
    labels = []

    for _, row in df.iterrows():
        # Skip rows with missing critical data
        if pd.isna(row.get('r_name')) or pd.isna(row.get('b_name')) or pd.isna(row.get('winner')):
            continue

        # Target area statistics (head, body, leg)
        r_head_acc = row.get('r_head_acc', 0) or 0
        b_head_acc = row.get('b_head_acc', 0) or 0
        r_body_acc = row.get('r_body_acc', 0) or 0
        b_body_acc = row.get('b_body_acc', 0) or 0
        r_leg_acc = row.get('r_leg_acc', 0) or 0
        b_leg_acc = row.get('b_leg_acc', 0) or 0

        r_head_landed = row.get('r_head_landed', 0) or 0
        b_head_landed = row.get('b_head_landed', 0) or 0
        r_body_landed = row.get('r_body_landed', 0) or 0
        b_body_landed = row.get('b_body_landed', 0) or 0
        r_leg_landed = row.get('r_leg_landed', 0) or 0
        b_leg_landed = row.get('b_leg_landed', 0) or 0

        # Position statistics (distance, clinch, ground)
        r_dist_acc = row.get('r_dist_acc', 0) or 0
        b_dist_acc = row.get('b_dist_acc', 0) or 0
        r_clinch_acc = row.get('r_clinch_acc', 0) or 0
        b_clinch_acc = row.get('b_clinch_acc', 0) or 0
        r_ground_acc = row.get('r_ground_acc', 0) or 0
        b_ground_acc = row.get('b_ground_acc', 0) or 0

        r_dist_landed = row.get('r_dist_landed', 0) or 0
        b_dist_landed = row.get('b_dist_landed', 0) or 0
        r_clinch_landed = row.get('r_clinch_landed', 0) or 0
        b_clinch_landed = row.get('b_clinch_landed', 0) or 0
        r_ground_landed = row.get('r_ground_landed', 0) or 0
        b_ground_landed = row.get('b_ground_landed', 0) or 0

        positional_features = [
            # Target area accuracy differences
            r_head_acc - b_head_acc,
            r_body_acc - b_body_acc,
            r_leg_acc - b_leg_acc,

            # Target area volume differences
            r_head_landed - b_head_landed,
            r_body_landed - b_body_landed,
            r_leg_landed - b_leg_landed,

            # Position accuracy differences
            r_dist_acc - b_dist_acc,
            r_clinch_acc - b_clinch_acc,
            r_ground_acc - b_ground_acc,

            # Position volume differences
            r_dist_landed - b_dist_landed,
            r_clinch_landed - b_clinch_landed,
            r_ground_landed - b_ground_landed,

            # Individual accuracy stats
            r_head_acc, b_head_acc,
            r_body_acc, b_body_acc,
            r_leg_acc, b_leg_acc,
            r_dist_acc, b_dist_acc,
            r_clinch_acc, b_clinch_acc,
            r_ground_acc, b_ground_acc,

            # Volume distribution ratios
            r_head_landed / max(r_head_landed + r_body_landed + r_leg_landed, 1),  # Head focus
            b_head_landed / max(b_head_landed + b_body_landed + b_leg_landed, 1),
            r_dist_landed / max(r_dist_landed + r_clinch_landed + r_ground_landed, 1),  # Distance fighting
            b_dist_landed / max(b_dist_landed + b_clinch_landed + b_ground_landed, 1)
        ]

        features.append(positional_features)
        labels.append(1 if row['winner'] == row['r_name'] else 0)

    feature_names = [
        'Head_Acc_Diff', 'Body_Acc_Diff', 'Leg_Acc_Diff',
        'Head_Landed_Diff', 'Body_Landed_Diff', 'Leg_Landed_Diff',
        'Dist_Acc_Diff', 'Clinch_Acc_Diff', 'Ground_Acc_Diff',
        'Dist_Landed_Diff', 'Clinch_Landed_Diff', 'Ground_Landed_Diff',
        'R_Head_Acc', 'B_Head_Acc',
        'R_Body_Acc', 'B_Body_Acc',
        'R_Leg_Acc', 'B_Leg_Acc',
        'R_Dist_Acc', 'B_Dist_Acc',
        'R_Clinch_Acc', 'B_Clinch_Acc',
        'R_Ground_Acc', 'B_Ground_Acc',
        'R_Head_Focus', 'B_Head_Focus',
        'R_Distance_Focus', 'B_Distance_Focus'
    ]

    return np.array(features), np.array(labels), feature_names


def create_fight_context_features(df):
    """
    Create features focused on fight context (title fights, method, timing, etc.).

    Args:
        df (pandas.DataFrame): UFC fight dataset

    Returns:
        tuple: (features_array, labels_array, feature_names)
    """
    features = []
    labels = []

    # Clean the dataframe first
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=['r_name', 'b_name', 'winner'])
    df_clean = df_clean[df_clean['winner'].astype(str).str.lower() != 'nan']

    # Initialize label encoders
    method_encoder = LabelEncoder()
    referee_encoder = LabelEncoder()

    # Fit encoders on all data first
    method_values = df_clean['method'].fillna('Unknown').astype(str)
    referee_values = df_clean['referee'].fillna('Unknown').astype(str)

    # Replace 'nan' strings with 'Unknown'
    method_values = method_values.replace('nan', 'Unknown')
    referee_values = referee_values.replace('nan', 'Unknown')

    method_encoder.fit(method_values)
    referee_encoder.fit(referee_values)

    for _, row in df_clean.iterrows():
        # Additional safety check
        if (pd.isna(row.get('r_name')) or pd.isna(row.get('b_name')) or
            pd.isna(row.get('winner')) or str(row.get('winner')).lower() == 'nan'):
            continue

        # Fight context features
        title_fight = row.get('title_fight', 0) or 0
        finish_round = row.get('finish_round', 3) or 3
        match_time_sec = row.get('match_time_sec', 900) or 900  # Default to 15 minutes
        total_rounds = row.get('total_rounds', 3) or 3

        # Encode categorical variables
        method = str(row.get('method', 'Unknown'))
        referee = str(row.get('referee', 'Unknown'))

        # Handle 'nan' strings
        if method.lower() == 'nan' or pd.isna(method):
            method = 'Unknown'
        if referee.lower() == 'nan' or pd.isna(referee):
            referee = 'Unknown'

        method_encoded = method_encoder.transform([method])[0]
        referee_encoded = referee_encoder.transform([referee])[0]

        # Calculate fight pace and timing metrics
        rounds_completed = finish_round if finish_round < total_rounds else total_rounds
        fight_duration_ratio = match_time_sec / (total_rounds * 300)  # Ratio of actual to maximum time
        early_finish = 1 if rounds_completed < total_rounds else 0

        context_features = [
            title_fight,
            finish_round,
            match_time_sec,
            total_rounds,
            method_encoded,
            referee_encoded,
            rounds_completed,
            fight_duration_ratio,
            early_finish,

            # Derived timing features
            match_time_sec / 60,  # Duration in minutes
            rounds_completed / total_rounds,  # Completion ratio
            1 if match_time_sec < 300 else 0,  # Quick finish (under 5 minutes)
            1 if 'Decision' in method else 0,  # Went to decision
            1 if 'Submission' in method else 0,  # Submission finish
            1 if 'KO' in method or 'TKO' in method else 0  # Knockout finish
        ]

        features.append(context_features)
        labels.append(1 if row['winner'] == row['r_name'] else 0)

    feature_names = [
        'Title_Fight', 'Finish_Round', 'Match_Time_Sec', 'Total_Rounds',
        'Method_Encoded', 'Referee_Encoded', 'Rounds_Completed',
        'Duration_Ratio', 'Early_Finish', 'Duration_Minutes',
        'Completion_Ratio', 'Quick_Finish', 'Decision_Finish',
        'Submission_Finish', 'Knockout_Finish'
    ]

    return np.array(features), np.array(labels), feature_names


def create_striking_decision_tree(df, max_depth=8, min_samples_split=15, min_samples_leaf=8, save_model=True):
    """
    Create and train a decision tree specialized for striking statistics.

    Args:
        df (pandas.DataFrame): UFC fight dataset
        max_depth (int): Maximum depth of the decision tree
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required in a leaf node
        save_model (bool): Whether to save the trained model

    Returns:
        tuple: (trained_model, feature_names, accuracy, test_predictions)
    """
    # Create striking-specific features
    X, y, feature_names = create_striking_features(df)

    if len(X) == 0:
        print("No valid striking data found")
        return None, None, 0, None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train striking-focused decision tree
    dt_striking = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        criterion='gini'
    )

    dt_striking.fit(X_train, y_train)
    y_pred = dt_striking.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nStriking Decision Tree Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Feature importance
    importance = dt_striking.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print(f"\nStriking Feature Importance:")
    print(feature_importance_df.head(10).to_string(index=False))

    if save_model:
        joblib.dump(dt_striking, 'models/ufc_striking_decision_tree.pkl')
        print(f"\nStriking decision tree saved as 'models/ufc_striking_decision_tree.pkl'")

    return dt_striking, feature_names, accuracy, {'y_test': y_test, 'y_pred': y_pred}


def create_grappling_decision_tree(df, max_depth=6, min_samples_split=20, min_samples_leaf=10, save_model=True):
    """
    Create and train a decision tree specialized for grappling statistics.

    Args:
        df (pandas.DataFrame): UFC fight dataset
        max_depth (int): Maximum depth of the decision tree
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required in a leaf node
        save_model (bool): Whether to save the trained model

    Returns:
        tuple: (trained_model, feature_names, accuracy, test_predictions)
    """
    # Create grappling-specific features
    X, y, feature_names = create_grappling_features(df)

    if len(X) == 0:
        print("No valid grappling data found")
        return None, None, 0, None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train grappling-focused decision tree
    dt_grappling = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        criterion='gini'
    )

    dt_grappling.fit(X_train, y_train)
    y_pred = dt_grappling.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nGrappling Decision Tree Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Feature importance
    importance = dt_grappling.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print(f"\nGrappling Feature Importance:")
    print(feature_importance_df.head(10).to_string(index=False))

    if save_model:
        joblib.dump(dt_grappling, 'models/ufc_grappling_decision_tree.pkl')
        print(f"\nGrappling decision tree saved as 'models/ufc_grappling_decision_tree.pkl'")

    return dt_grappling, feature_names, accuracy, {'y_test': y_test, 'y_pred': y_pred}


def create_positional_decision_tree(df, max_depth=7, min_samples_split=15, min_samples_leaf=8, save_model=True):
    """
    Create and train a decision tree specialized for positional striking statistics.

    Args:
        df (pandas.DataFrame): UFC fight dataset
        max_depth (int): Maximum depth of the decision tree
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required in a leaf node
        save_model (bool): Whether to save the trained model

    Returns:
        tuple: (trained_model, feature_names, accuracy, test_predictions)
    """
    # Create positional-specific features
    X, y, feature_names = create_positional_features(df)

    if len(X) == 0:
        print("No valid positional data found")
        return None, None, 0, None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train positional-focused decision tree
    dt_positional = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        criterion='gini'
    )

    dt_positional.fit(X_train, y_train)
    y_pred = dt_positional.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nPositional Decision Tree Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Feature importance
    importance = dt_positional.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print(f"\nPositional Feature Importance:")
    print(feature_importance_df.head(10).to_string(index=False))

    if save_model:
        joblib.dump(dt_positional, 'models/ufc_positional_decision_tree.pkl')
        print(f"\nPositional decision tree saved as 'models/ufc_positional_decision_tree.pkl'")

    return dt_positional, feature_names, accuracy, {'y_test': y_test, 'y_pred': y_pred}


def create_context_decision_tree(df, max_depth=5, min_samples_split=25, min_samples_leaf=12, save_model=True):
    """
    Create and train a decision tree specialized for fight context features.

    Args:
        df (pandas.DataFrame): UFC fight dataset
        max_depth (int): Maximum depth of the decision tree
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required in a leaf node
        save_model (bool): Whether to save the trained model

    Returns:
        tuple: (trained_model, feature_names, accuracy, test_predictions)
    """
    # Create context-specific features
    X, y, feature_names = create_fight_context_features(df)

    if len(X) == 0:
        print("No valid context data found")
        return None, None, 0, None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train context-focused decision tree
    dt_context = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        criterion='gini'
    )

    dt_context.fit(X_train, y_train)
    y_pred = dt_context.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nContext Decision Tree Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Feature importance
    importance = dt_context.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print(f"\nContext Feature Importance:")
    print(feature_importance_df.head(10).to_string(index=False))

    if save_model:
        joblib.dump(dt_context, 'models/ufc_context_decision_tree.pkl')
        print(f"\nContext decision tree saved as 'models/ufc_context_decision_tree.pkl'")

    return dt_context, feature_names, accuracy, {'y_test': y_test, 'y_pred': y_pred}


class UFC_SpecializedEnsemble:
    """
    Custom ensemble model that combines specialized decision trees for UFC fight prediction.

    Each tree specializes in different aspects of fighting:
    - Striking tree: focuses on striking statistics
    - Grappling tree: focuses on takedowns, submissions, control
    - Positional tree: focuses on head/body/leg and distance/clinch/ground
    - Context tree: focuses on fight circumstances (title fights, timing, method)
    """

    def __init__(self, weights=None):
        """
        Initialize the specialized ensemble.

        Args:
            weights (dict): Custom weights for each tree type
                          Default: {'striking': 0.3, 'grappling': 0.25, 'positional': 0.25, 'context': 0.2}
        """
        self.trees = {}
        self.feature_names = {}
        self.weights = weights or {
            'striking': 0.3,
            'grappling': 0.25,
            'positional': 0.25,
            'context': 0.2
        }

    def train_all_trees(self, df, save_models=True):
        """
        Train all specialized decision trees.

        Args:
            df (pandas.DataFrame): UFC fight dataset
            save_models (bool): Whether to save trained models

        Returns:
            dict: Training results for each tree type
        """
        results = {}

        print("Training Specialized Decision Tree Ensemble")
        print("=" * 60)

        # Train striking tree
        print("\n1. Training Striking Decision Tree...")
        self.trees['striking'], self.feature_names['striking'], acc_striking, res_striking = \
            create_striking_decision_tree(df, save_model=save_models)
        results['striking'] = {'accuracy': acc_striking, 'results': res_striking}

        # Train grappling tree
        print("\n2. Training Grappling Decision Tree...")
        self.trees['grappling'], self.feature_names['grappling'], acc_grappling, res_grappling = \
            create_grappling_decision_tree(df, save_model=save_models)
        results['grappling'] = {'accuracy': acc_grappling, 'results': res_grappling}

        # Train positional tree
        print("\n3. Training Positional Decision Tree...")
        self.trees['positional'], self.feature_names['positional'], acc_positional, res_positional = \
            create_positional_decision_tree(df, save_model=save_models)
        results['positional'] = {'accuracy': acc_positional, 'results': res_positional}

        # Train context tree
        print("\n4. Training Context Decision Tree...")
        self.trees['context'], self.feature_names['context'], acc_context, res_context = \
            create_context_decision_tree(df, save_model=save_models)
        results['context'] = {'accuracy': acc_context, 'results': res_context}

        # Calculate ensemble accuracy
        ensemble_accuracy = self._calculate_ensemble_accuracy(results)
        results['ensemble'] = {'accuracy': ensemble_accuracy}

        print("\n" + "=" * 60)
        print("SPECIALIZED ENSEMBLE TRAINING COMPLETE")
        print("=" * 60)
        print(f"Striking Tree Accuracy:    {acc_striking:.3f}")
        print(f"Grappling Tree Accuracy:   {acc_grappling:.3f}")
        print(f"Positional Tree Accuracy:  {acc_positional:.3f}")
        print(f"Context Tree Accuracy:     {acc_context:.3f}")
        print(f"Ensemble Accuracy:         {ensemble_accuracy:.3f}")
        print("\nTree Weights:")
        for tree_type, weight in self.weights.items():
            print(f"  {tree_type.capitalize()}: {weight}")

        if save_models:
            self.save_ensemble('models/ufc_specialized_ensemble.pkl')

        return results

    def _calculate_ensemble_accuracy(self, results):
        """Calculate weighted ensemble accuracy from individual tree results."""
        if not all(results[tree_type]['results'] for tree_type in ['striking', 'grappling', 'positional', 'context']):
            return 0.0

        # Get test predictions from first tree (they should all use same test split)
        y_test = results['striking']['results']['y_test']

        # Get predictions from each tree
        ensemble_predictions = []
        for i in range(len(y_test)):
            weighted_prob = 0
            for tree_type in ['striking', 'grappling', 'positional', 'context']:
                if self.trees[tree_type] is not None:
                    tree_pred = results[tree_type]['results']['y_pred'][i]
                    weighted_prob += tree_pred * self.weights[tree_type]

            ensemble_predictions.append(1 if weighted_prob > 0.5 else 0)

        return accuracy_score(y_test, ensemble_predictions)

    def predict_fight(self, fight_data):
        """
        Predict fight outcome using the ensemble of specialized trees.

        Args:
            fight_data (dict): Fight statistics for both fighters

        Returns:
            dict: Prediction results with individual tree contributions
        """
        if not all(tree is not None for tree in self.trees.values()):
            raise ValueError("Not all trees have been trained. Call train_all_trees() first.")

        predictions = {}
        probabilities = {}

        # Create mini dataframe for feature extraction
        df_single = pd.DataFrame([fight_data])

        # Get predictions from each specialized tree
        for tree_type, tree in self.trees.items():
            if tree_type == 'striking':
                X, _, _ = create_striking_features(df_single)
            elif tree_type == 'grappling':
                X, _, _ = create_grappling_features(df_single)
            elif tree_type == 'positional':
                X, _, _ = create_positional_features(df_single)
            elif tree_type == 'context':
                X, _, _ = create_fight_context_features(df_single)

            if len(X) > 0:
                pred = tree.predict(X[0].reshape(1, -1))[0]
                prob = tree.predict_proba(X[0].reshape(1, -1))[0]

                predictions[tree_type] = pred
                probabilities[tree_type] = {'fighter_a_prob': prob[1], 'fighter_b_prob': prob[0]}

        # Calculate weighted ensemble prediction
        weighted_prob = sum(
            predictions[tree_type] * self.weights[tree_type]
            for tree_type in predictions.keys()
        )

        ensemble_prediction = 1 if weighted_prob > 0.5 else 0

        result = {
            'ensemble_prediction': 'Fighter A (Red Corner)' if ensemble_prediction == 1 else 'Fighter B (Blue Corner)',
            'ensemble_confidence': abs(weighted_prob - 0.5) * 2,  # Convert to 0-1 scale
            'weighted_probability': weighted_prob,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'tree_weights': self.weights
        }

        return result

    def save_ensemble(self, filename):
        """Save the trained ensemble to a file."""
        ensemble_data = {
            'trees': self.trees,
            'feature_names': self.feature_names,
            'weights': self.weights
        }

        with open(filename, 'wb') as f:
            pickle.dump(ensemble_data, f)
        print(f"\nSpecialized ensemble saved as '{filename}'")

    @classmethod
    def load_ensemble(cls, filename):
        """Load a previously saved ensemble."""
        with open(filename, 'rb') as f:
            ensemble_data = pickle.load(f)

        ensemble = cls(weights=ensemble_data['weights'])
        ensemble.trees = ensemble_data['trees']
        ensemble.feature_names = ensemble_data['feature_names']

        print(f"Specialized ensemble loaded from '{filename}'")
        return ensemble


def create_comprehensive_random_forest(df, n_estimators=200, max_depth=None, save_model=True):
    """
    Create a comprehensive random forest using ALL available features.

    Args:
        df (pandas.DataFrame): UFC fight dataset
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of trees (None for no limit)
        save_model (bool): Whether to save the trained model

    Returns:
        tuple: (trained_model, all_feature_names, accuracy, test_predictions)
    """
    # Combine all feature types
    X_striking, y_striking, names_striking = create_striking_features(df)
    X_grappling, y_grappling, names_grappling = create_grappling_features(df)
    X_positional, y_positional, names_positional = create_positional_features(df)
    X_context, y_context, names_context = create_fight_context_features(df)

    # Ensure all feature sets have the same number of samples
    min_samples = min(len(X_striking), len(X_grappling), len(X_positional), len(X_context))

    # Combine all features
    X_combined = np.hstack([
        X_striking[:min_samples],
        X_grappling[:min_samples],
        X_positional[:min_samples],
        X_context[:min_samples]
    ])

    y_combined = y_striking[:min_samples]  # They should all be the same

    all_feature_names = (names_striking + names_grappling +
                        names_positional + names_context)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42
    )

    # Create comprehensive random forest
    rf_comprehensive = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf_comprehensive.fit(X_train, y_train)
    y_pred = rf_comprehensive.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nComprehensive Random Forest Results ({n_estimators} trees):")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Total features: {len(all_feature_names)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Feature importance
    importance = rf_comprehensive.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 15 Most Important Features:")
    print(feature_importance_df.head(15).to_string(index=False))

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fighter B Wins', 'Fighter A Wins']))

    if save_model:
        joblib.dump(rf_comprehensive, 'models/ufc_comprehensive_random_forest.pkl')
        print(f"\nComprehensive random forest saved as 'models/ufc_comprehensive_random_forest.pkl'")

        # Save feature importance analysis
        feature_importance_df.to_csv('models/ufc_feature_importance_analysis.csv', index=False)
        print(f"Feature importance analysis saved as 'models/ufc_feature_importance_analysis.csv'")

    return rf_comprehensive, all_feature_names, accuracy, {'y_test': y_test, 'y_pred': y_pred}