"""
Individual Decision Trees for UFC Fight Prediction
=================================================

This module implements 32 individual decision trees, each focusing on a specific
fighter performance metric comparison. These trees can be used individually
or combined into a custom random forest ensemble.

Tree Categories:
- Striking Performance - Landed (8 trees)
- Striking Performance - Accuracy (8 trees)
- Grappling Performance - Landed (3 trees)
- Grappling Performance - Accuracy/Control (2 trees)
- Positional Fighting - Landed (6 trees)
- Positional Fighting - Accuracy (3 trees)
- Fight Context (2 trees)

Total: 32 Individual Decision Trees
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
import sys
sys.path.append('/Users/ralphfrancolini/UFCML/src/core')
from advanced_ml_models import load_enhanced_ufc_data


def safe_division(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if denominator is 0."""
    return numerator / denominator if denominator != 0 else default


def remove_data_leakage_features(df):
    """Remove features that could cause data leakage."""
    # Features that should be excluded to prevent data leakage
    leakage_features = [
        'method',  # Fight outcome method could leak result information
        'finish_round',  # When fight ended could leak information
        'finish_time',  # Exact finish time could leak information
        'title_bout',  # While not direct leakage, could bias toward certain fighters
        'referee',  # Referee decisions could be biased
    ]

    # Remove leakage features if they exist
    features_to_remove = [f for f in leakage_features if f in df.columns]
    if features_to_remove:
        print(f"🔒 Removing potential data leakage features: {features_to_remove}")
        df = df.drop(columns=features_to_remove)

    return df

def create_base_features(df):
    """Create base comparison features between fighters."""
    # Remove potential data leakage features first
    df_clean = remove_data_leakage_features(df)

    features = []
    labels = []

    for _, row in df_clean.iterrows():
        # Skip rows with missing critical data
        if (pd.isna(row.get('r_name')) or pd.isna(row.get('b_name')) or
            pd.isna(row.get('winner')) or str(row.get('winner')).lower() == 'nan'):
            continue

        # Label: 1 if red corner won, 0 if blue corner won
        label = 1 if row['winner'] == row['r_name'] else 0
        labels.append(label)

        # Store the row for individual feature extraction
        features.append(row)

    return features, labels


# ============================================================================
# STRIKING PERFORMANCE - LANDED TREES (8 trees)
# ============================================================================

def create_knockdowns_tree_features(fight_data):
    """Tree 1: Knockdowns Comparison"""
    features = []
    for row in fight_data:
        r_kd = row.get('r_kd', 0) or 0
        b_kd = row.get('b_kd', 0) or 0

        feature_vector = [
            r_kd - b_kd,  # Knockdown difference
            r_kd,         # Red knockdowns
            b_kd,         # Blue knockdowns
            r_kd + b_kd,  # Total knockdowns
            1 if r_kd > b_kd else 0,  # Red has more knockdowns
        ]
        features.append(feature_vector)

    feature_names = ['KD_Diff', 'R_KD', 'B_KD', 'Total_KD', 'R_KD_Advantage']
    return np.array(features), feature_names


def create_sig_strikes_landed_tree_features(fight_data):
    """Tree 2: Significant Strikes Landed Comparison"""
    features = []
    for row in fight_data:
        r_landed = row.get('r_sig_str_landed', 0) or 0
        b_landed = row.get('b_sig_str_landed', 0) or 0

        feature_vector = [
            r_landed - b_landed,  # Difference
            r_landed,             # Red landed
            b_landed,             # Blue landed
            r_landed + b_landed,  # Total landed
            safe_division(r_landed, b_landed, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Sig_Landed_Diff', 'R_Sig_Landed', 'B_Sig_Landed', 'Total_Sig_Landed', 'Sig_Landed_Ratio']
    return np.array(features), feature_names


def create_sig_strikes_attempted_tree_features(fight_data):
    """Tree 3: Significant Strikes Attempted Comparison"""
    features = []
    for row in fight_data:
        r_attempted = row.get('r_sig_str_atmpted', 0) or 0
        b_attempted = row.get('b_sig_str_atmpted', 0) or 0

        feature_vector = [
            r_attempted - b_attempted,  # Difference
            r_attempted,                # Red attempted
            b_attempted,                # Blue attempted
            r_attempted + b_attempted,  # Total attempted
            safe_division(r_attempted, b_attempted, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Sig_Attempted_Diff', 'R_Sig_Attempted', 'B_Sig_Attempted', 'Total_Sig_Attempted', 'Sig_Attempted_Ratio']
    return np.array(features), feature_names


def create_total_strikes_landed_tree_features(fight_data):
    """Tree 4: Total Strikes Landed Comparison"""
    features = []
    for row in fight_data:
        r_landed = row.get('r_total_str_landed', 0) or 0
        b_landed = row.get('b_total_str_landed', 0) or 0

        feature_vector = [
            r_landed - b_landed,  # Difference
            r_landed,             # Red landed
            b_landed,             # Blue landed
            r_landed + b_landed,  # Total landed
            safe_division(r_landed, b_landed, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Total_Landed_Diff', 'R_Total_Landed', 'B_Total_Landed', 'Total_Strikes_Landed', 'Total_Landed_Ratio']
    return np.array(features), feature_names


def create_total_strikes_attempted_tree_features(fight_data):
    """Tree 5: Total Strikes Attempted Comparison"""
    features = []
    for row in fight_data:
        r_attempted = row.get('r_total_str_atmpted', 0) or 0
        b_attempted = row.get('b_total_str_atmpted', 0) or 0

        feature_vector = [
            r_attempted - b_attempted,  # Difference
            r_attempted,                # Red attempted
            b_attempted,                # Blue attempted
            r_attempted + b_attempted,  # Total attempted
            safe_division(r_attempted, b_attempted, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Total_Attempted_Diff', 'R_Total_Attempted', 'B_Total_Attempted', 'Total_Strikes_Attempted', 'Total_Attempted_Ratio']
    return np.array(features), feature_names


def create_head_strikes_landed_tree_features(fight_data):
    """Tree 6: Head Strikes Landed Comparison"""
    features = []
    for row in fight_data:
        r_landed = row.get('r_head_landed', 0) or 0
        b_landed = row.get('b_head_landed', 0) or 0

        feature_vector = [
            r_landed - b_landed,  # Difference
            r_landed,             # Red landed
            b_landed,             # Blue landed
            r_landed + b_landed,  # Total landed
            safe_division(r_landed, b_landed, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Head_Landed_Diff', 'R_Head_Landed', 'B_Head_Landed', 'Total_Head_Landed', 'Head_Landed_Ratio']
    return np.array(features), feature_names


def create_body_strikes_landed_tree_features(fight_data):
    """Tree 7: Body Strikes Landed Comparison"""
    features = []
    for row in fight_data:
        r_landed = row.get('r_body_landed', 0) or 0
        b_landed = row.get('b_body_landed', 0) or 0

        feature_vector = [
            r_landed - b_landed,  # Difference
            r_landed,             # Red landed
            b_landed,             # Blue landed
            r_landed + b_landed,  # Total landed
            safe_division(r_landed, b_landed, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Body_Landed_Diff', 'R_Body_Landed', 'B_Body_Landed', 'Total_Body_Landed', 'Body_Landed_Ratio']
    return np.array(features), feature_names


def create_leg_strikes_landed_tree_features(fight_data):
    """Tree 8: Leg Strikes Landed Comparison"""
    features = []
    for row in fight_data:
        r_landed = row.get('r_leg_landed', 0) or 0
        b_landed = row.get('b_leg_landed', 0) or 0

        feature_vector = [
            r_landed - b_landed,  # Difference
            r_landed,             # Red landed
            b_landed,             # Blue landed
            r_landed + b_landed,  # Total landed
            safe_division(r_landed, b_landed, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Leg_Landed_Diff', 'R_Leg_Landed', 'B_Leg_Landed', 'Total_Leg_Landed', 'Leg_Landed_Ratio']
    return np.array(features), feature_names


# ============================================================================
# DEFENSIVE PERFORMANCE TREES (4 trees)
# ============================================================================

def create_striking_defense_tree_features(fight_data):
    """Tree 33: Striking Defense Comparison"""
    features = []
    for row in fight_data:
        r_str_def = row.get('r_str_def', 0) or 0
        b_str_def = row.get('b_str_def', 0) or 0

        feature_vector = [
            r_str_def - b_str_def,  # Defense difference
            r_str_def,              # Red defense
            b_str_def,              # Blue defense
            (r_str_def + b_str_def) / 2,  # Average defense
            1 if r_str_def > b_str_def else 0,  # Red has better defense
        ]
        features.append(feature_vector)

    feature_names = ['Str_Def_Diff', 'R_Str_Def', 'B_Str_Def', 'Avg_Str_Def', 'R_Def_Advantage']
    return np.array(features), feature_names


def create_takedown_defense_tree_features(fight_data):
    """Tree 34: Takedown Defense Comparison"""
    features = []
    for row in fight_data:
        r_td_def = row.get('r_td_def', 0) or 0
        b_td_def = row.get('b_td_def', 0) or 0

        feature_vector = [
            r_td_def - b_td_def,  # TD Defense difference
            r_td_def,             # Red TD defense
            b_td_def,             # Blue TD defense
            (r_td_def + b_td_def) / 2,  # Average TD defense
            1 if r_td_def > b_td_def else 0,  # Red has better TD defense
        ]
        features.append(feature_vector)

    feature_names = ['TD_Def_Diff', 'R_TD_Def', 'B_TD_Def', 'Avg_TD_Def', 'R_TD_Def_Advantage']
    return np.array(features), feature_names


def create_strikes_absorbed_per_min_tree_features(fight_data):
    """Tree 35: Strikes Absorbed Per Minute (SAPM) Comparison"""
    features = []
    for row in fight_data:
        r_sapm = row.get('r_sapm', 0) or 0
        b_sapm = row.get('b_sapm', 0) or 0

        feature_vector = [
            r_sapm - b_sapm,  # SAPM difference (lower is better)
            r_sapm,           # Red SAPM
            b_sapm,           # Blue SAPM
            r_sapm + b_sapm,  # Total absorbed rate
            1 if r_sapm < b_sapm else 0,  # Red absorbs fewer (better)
        ]
        features.append(feature_vector)

    feature_names = ['SAPM_Diff', 'R_SAPM', 'B_SAPM', 'Total_SAPM', 'R_Absorbs_Less']
    return np.array(features), feature_names


def create_strikes_landed_per_min_tree_features(fight_data):
    """Tree 36: Strikes Landed Per Minute (SPLM) Comparison"""
    features = []
    for row in fight_data:
        r_splm = row.get('r_splm', 0) or 0
        b_splm = row.get('b_splm', 0) or 0

        feature_vector = [
            r_splm - b_splm,  # SPLM difference
            r_splm,           # Red SPLM
            b_splm,           # Blue SPLM
            r_splm + b_splm,  # Total output rate
            1 if r_splm > b_splm else 0,  # Red lands more
        ]
        features.append(feature_vector)

    feature_names = ['SPLM_Diff', 'R_SPLM', 'B_SPLM', 'Total_SPLM', 'R_SPLM_Advantage']
    return np.array(features), feature_names


# ============================================================================
# PHYSICAL ATTRIBUTES TREES (3 trees)
# ============================================================================

def create_height_advantage_tree_features(fight_data):
    """Tree 37: Height Advantage Comparison"""
    features = []
    for row in fight_data:
        r_height = row.get('r_height', 0) or 0
        b_height = row.get('b_height', 0) or 0

        feature_vector = [
            r_height - b_height,  # Height difference (inches)
            r_height,             # Red height
            b_height,             # Blue height
            abs(r_height - b_height),  # Absolute height difference
            1 if r_height > b_height else 0,  # Red is taller
        ]
        features.append(feature_vector)

    feature_names = ['Height_Diff', 'R_Height', 'B_Height', 'Abs_Height_Diff', 'R_Taller']
    return np.array(features), feature_names


def create_reach_advantage_tree_features(fight_data):
    """Tree 38: Reach Advantage Comparison"""
    features = []
    for row in fight_data:
        r_reach = row.get('r_reach', 0) or 0
        b_reach = row.get('b_reach', 0) or 0

        feature_vector = [
            r_reach - b_reach,  # Reach difference (inches)
            r_reach,            # Red reach
            b_reach,            # Blue reach
            abs(r_reach - b_reach),  # Absolute reach difference
            1 if r_reach > b_reach else 0,  # Red has longer reach
        ]
        features.append(feature_vector)

    feature_names = ['Reach_Diff', 'R_Reach', 'B_Reach', 'Abs_Reach_Diff', 'R_Longer_Reach']
    return np.array(features), feature_names


def create_weight_comparison_tree_features(fight_data):
    """Tree 39: Weight Comparison"""
    features = []
    for row in fight_data:
        r_weight = row.get('r_weight', 0) or 0
        b_weight = row.get('b_weight', 0) or 0

        feature_vector = [
            r_weight - b_weight,  # Weight difference (lbs)
            r_weight,             # Red weight
            b_weight,             # Blue weight
            abs(r_weight - b_weight),  # Absolute weight difference
            1 if r_weight > b_weight else 0,  # Red is heavier
        ]
        features.append(feature_vector)

    feature_names = ['Weight_Diff', 'R_Weight', 'B_Weight', 'Abs_Weight_Diff', 'R_Heavier']
    return np.array(features), feature_names


# ============================================================================
# EXPERIENCE & CAREER TREES (4 trees)
# ============================================================================

def create_wins_comparison_tree_features(fight_data):
    """Tree 40: Career Wins Comparison"""
    features = []
    for row in fight_data:
        r_wins = row.get('r_wins', 0) or 0
        b_wins = row.get('b_wins', 0) or 0

        feature_vector = [
            r_wins - b_wins,  # Wins difference
            r_wins,           # Red wins
            b_wins,           # Blue wins
            r_wins + b_wins,  # Total experience (fights)
            1 if r_wins > b_wins else 0,  # Red has more wins
        ]
        features.append(feature_vector)

    feature_names = ['Wins_Diff', 'R_Wins', 'B_Wins', 'Total_Wins', 'R_More_Wins']
    return np.array(features), feature_names


def create_losses_comparison_tree_features(fight_data):
    """Tree 41: Career Losses Comparison"""
    features = []
    for row in fight_data:
        r_losses = row.get('r_losses', 0) or 0
        b_losses = row.get('b_losses', 0) or 0

        feature_vector = [
            r_losses - b_losses,  # Losses difference (fewer is better)
            r_losses,             # Red losses
            b_losses,             # Blue losses
            r_losses + b_losses,  # Total losses
            1 if r_losses < b_losses else 0,  # Red has fewer losses
        ]
        features.append(feature_vector)

    feature_names = ['Losses_Diff', 'R_Losses', 'B_Losses', 'Total_Losses', 'R_Fewer_Losses']
    return np.array(features), feature_names


def create_win_rate_tree_features(fight_data):
    """Tree 42: Win Rate Comparison"""
    features = []
    for row in fight_data:
        r_wins = row.get('r_wins', 0) or 0
        r_losses = row.get('r_losses', 0) or 0
        b_wins = row.get('b_wins', 0) or 0
        b_losses = row.get('b_losses', 0) or 0

        r_total_fights = r_wins + r_losses
        b_total_fights = b_wins + b_losses

        r_win_rate = safe_division(r_wins, r_total_fights, 0.5)
        b_win_rate = safe_division(b_wins, b_total_fights, 0.5)

        feature_vector = [
            r_win_rate - b_win_rate,  # Win rate difference
            r_win_rate,               # Red win rate
            b_win_rate,               # Blue win rate
            (r_win_rate + b_win_rate) / 2,  # Average win rate
            1 if r_win_rate > b_win_rate else 0,  # Red has better win rate
        ]
        features.append(feature_vector)

    feature_names = ['WinRate_Diff', 'R_WinRate', 'B_WinRate', 'Avg_WinRate', 'R_Better_WinRate']
    return np.array(features), feature_names


def create_experience_gap_tree_features(fight_data):
    """Tree 43: Total Experience Gap"""
    features = []
    for row in fight_data:
        r_wins = row.get('r_wins', 0) or 0
        r_losses = row.get('r_losses', 0) or 0
        r_draws = row.get('r_draws', 0) or 0
        b_wins = row.get('b_wins', 0) or 0
        b_losses = row.get('b_losses', 0) or 0
        b_draws = row.get('b_draws', 0) or 0

        r_total_fights = r_wins + r_losses + r_draws
        b_total_fights = b_wins + b_losses + b_draws

        feature_vector = [
            r_total_fights - b_total_fights,  # Experience difference
            r_total_fights,                   # Red total fights
            b_total_fights,                   # Blue total fights
            r_total_fights + b_total_fights,  # Combined experience
            1 if r_total_fights > b_total_fights else 0,  # Red more experienced
        ]
        features.append(feature_vector)

    feature_names = ['Exp_Diff', 'R_Total_Fights', 'B_Total_Fights', 'Combined_Exp', 'R_More_Exp']
    return np.array(features), feature_names


# ============================================================================
# ADVANCED GRAPPLING METRICS (3 trees)
# ============================================================================

def create_takedown_average_tree_features(fight_data):
    """Tree 44: Takedown Average Comparison"""
    features = []
    for row in fight_data:
        r_td_avg = row.get('r_td_avg', 0) or 0
        b_td_avg = row.get('b_td_avg', 0) or 0

        feature_vector = [
            r_td_avg - b_td_avg,  # TD average difference
            r_td_avg,             # Red TD average
            b_td_avg,             # Blue TD average
            r_td_avg + b_td_avg,  # Combined TD average
            1 if r_td_avg > b_td_avg else 0,  # Red has higher TD average
        ]
        features.append(feature_vector)

    feature_names = ['TD_Avg_Diff', 'R_TD_Avg', 'B_TD_Avg', 'Total_TD_Avg', 'R_TD_Avg_Advantage']
    return np.array(features), feature_names


def create_submission_average_tree_features(fight_data):
    """Tree 45: Submission Average Comparison"""
    features = []
    for row in fight_data:
        r_sub_avg = row.get('r_sub_avg', 0) or 0
        b_sub_avg = row.get('b_sub_avg', 0) or 0

        feature_vector = [
            r_sub_avg - b_sub_avg,  # Sub average difference
            r_sub_avg,              # Red sub average
            b_sub_avg,              # Blue sub average
            r_sub_avg + b_sub_avg,  # Combined sub average
            1 if r_sub_avg > b_sub_avg else 0,  # Red has higher sub average
        ]
        features.append(feature_vector)

    feature_names = ['Sub_Avg_Diff', 'R_Sub_Avg', 'B_Sub_Avg', 'Total_Sub_Avg', 'R_Sub_Avg_Advantage']
    return np.array(features), feature_names


def create_overall_striking_accuracy_tree_features(fight_data):
    """Tree 46: Overall Striking Accuracy Comparison"""
    features = []
    for row in fight_data:
        r_str_acc = row.get('r_str_acc', 0) or 0
        b_str_acc = row.get('b_str_acc', 0) or 0

        feature_vector = [
            r_str_acc - b_str_acc,  # Overall accuracy difference
            r_str_acc,              # Red overall accuracy
            b_str_acc,              # Blue overall accuracy
            (r_str_acc + b_str_acc) / 2,  # Average accuracy
            1 if r_str_acc > b_str_acc else 0,  # Red more accurate
        ]
        features.append(feature_vector)

    feature_names = ['Str_Acc_Diff', 'R_Str_Acc', 'B_Str_Acc', 'Avg_Str_Acc', 'R_Acc_Advantage']
    return np.array(features), feature_names


# ============================================================================
# STRIKE DISTRIBUTION TREES (6 trees)
# ============================================================================

def create_head_strike_distribution_tree_features(fight_data):
    """Tree 47: Head Strike Distribution Percentage"""
    features = []
    for row in fight_data:
        r_head_per = row.get('r_landed_head_per', 0) or 0
        b_head_per = row.get('b_landed_head_per', 0) or 0

        feature_vector = [
            r_head_per - b_head_per,  # Head targeting difference
            r_head_per,               # Red head targeting %
            b_head_per,               # Blue head targeting %
            (r_head_per + b_head_per) / 2,  # Average head targeting
            1 if r_head_per > b_head_per else 0,  # Red targets head more
        ]
        features.append(feature_vector)

    feature_names = ['Head_Dist_Diff', 'R_Head_Dist', 'B_Head_Dist', 'Avg_Head_Dist', 'R_Head_Focus']
    return np.array(features), feature_names


def create_body_strike_distribution_tree_features(fight_data):
    """Tree 48: Body Strike Distribution Percentage"""
    features = []
    for row in fight_data:
        r_body_per = row.get('r_landed_body_per', 0) or 0
        b_body_per = row.get('b_landed_body_per', 0) or 0

        feature_vector = [
            r_body_per - b_body_per,  # Body targeting difference
            r_body_per,               # Red body targeting %
            b_body_per,               # Blue body targeting %
            (r_body_per + b_body_per) / 2,  # Average body targeting
            1 if r_body_per > b_body_per else 0,  # Red targets body more
        ]
        features.append(feature_vector)

    feature_names = ['Body_Dist_Diff', 'R_Body_Dist', 'B_Body_Dist', 'Avg_Body_Dist', 'R_Body_Focus']
    return np.array(features), feature_names


def create_leg_strike_distribution_tree_features(fight_data):
    """Tree 49: Leg Strike Distribution Percentage"""
    features = []
    for row in fight_data:
        r_leg_per = row.get('r_landed_leg_per', 0) or 0
        b_leg_per = row.get('b_landed_leg_per', 0) or 0

        feature_vector = [
            r_leg_per - b_leg_per,  # Leg targeting difference
            r_leg_per,              # Red leg targeting %
            b_leg_per,              # Blue leg targeting %
            (r_leg_per + b_leg_per) / 2,  # Average leg targeting
            1 if r_leg_per > b_leg_per else 0,  # Red targets legs more
        ]
        features.append(feature_vector)

    feature_names = ['Leg_Dist_Diff', 'R_Leg_Dist', 'B_Leg_Dist', 'Avg_Leg_Dist', 'R_Leg_Focus']
    return np.array(features), feature_names


def create_distance_position_distribution_tree_features(fight_data):
    """Tree 50: Distance Fighting Distribution Percentage"""
    features = []
    for row in fight_data:
        r_dist_per = row.get('r_landed_dist_per', 0) or 0
        b_dist_per = row.get('b_landed_dist_per', 0) or 0

        feature_vector = [
            r_dist_per - b_dist_per,  # Distance fighting difference
            r_dist_per,               # Red distance fighting %
            b_dist_per,               # Blue distance fighting %
            (r_dist_per + b_dist_per) / 2,  # Average distance fighting
            1 if r_dist_per > b_dist_per else 0,  # Red fights at distance more
        ]
        features.append(feature_vector)

    feature_names = ['Dist_Pos_Diff', 'R_Dist_Pos', 'B_Dist_Pos', 'Avg_Dist_Pos', 'R_Distance_Fighter']
    return np.array(features), feature_names


def create_clinch_position_distribution_tree_features(fight_data):
    """Tree 51: Clinch Fighting Distribution Percentage"""
    features = []
    for row in fight_data:
        r_clinch_per = row.get('r_landed_clinch_per', 0) or 0
        b_clinch_per = row.get('b_landed_clinch_per', 0) or 0

        feature_vector = [
            r_clinch_per - b_clinch_per,  # Clinch fighting difference
            r_clinch_per,                 # Red clinch fighting %
            b_clinch_per,                 # Blue clinch fighting %
            (r_clinch_per + b_clinch_per) / 2,  # Average clinch fighting
            1 if r_clinch_per > b_clinch_per else 0,  # Red fights in clinch more
        ]
        features.append(feature_vector)

    feature_names = ['Clinch_Pos_Diff', 'R_Clinch_Pos', 'B_Clinch_Pos', 'Avg_Clinch_Pos', 'R_Clinch_Fighter']
    return np.array(features), feature_names


def create_ground_position_distribution_tree_features(fight_data):
    """Tree 52: Ground Fighting Distribution Percentage"""
    features = []
    for row in fight_data:
        r_ground_per = row.get('r_landed_ground_per', 0) or 0
        b_ground_per = row.get('b_landed_ground_per', 0) or 0

        feature_vector = [
            r_ground_per - b_ground_per,  # Ground fighting difference
            r_ground_per,                 # Red ground fighting %
            b_ground_per,                 # Blue ground fighting %
            (r_ground_per + b_ground_per) / 2,  # Average ground fighting
            1 if r_ground_per > b_ground_per else 0,  # Red fights on ground more
        ]
        features.append(feature_vector)

    feature_names = ['Ground_Pos_Diff', 'R_Ground_Pos', 'B_Ground_Pos', 'Avg_Ground_Pos', 'R_Ground_Fighter']
    return np.array(features), feature_names


# ============================================================================
# LOGICAL CATEGORY TREES - PHASE 2 (11 trees)
# ============================================================================

def create_striking_volume_category_tree_features(fight_data):
    """Tree 53: Comprehensive Striking Volume Analysis"""
    features = []
    for row in fight_data:
        # All striking volume metrics
        r_sig_landed = row.get('r_sig_str_landed', 0) or 0
        b_sig_landed = row.get('b_sig_str_landed', 0) or 0
        r_sig_attempted = row.get('r_sig_str_atmpted', 0) or 0
        b_sig_attempted = row.get('b_sig_str_atmpted', 0) or 0
        r_total_landed = row.get('r_total_str_landed', 0) or 0
        b_total_landed = row.get('b_total_str_landed', 0) or 0
        r_total_attempted = row.get('r_total_str_atmpted', 0) or 0
        b_total_attempted = row.get('b_total_str_atmpted', 0) or 0

        feature_vector = [
            r_sig_landed - b_sig_landed,  # Sig strikes landed diff
            r_sig_attempted - b_sig_attempted,  # Sig strikes attempted diff
            r_total_landed - b_total_landed,  # Total strikes landed diff
            r_total_attempted - b_total_attempted,  # Total strikes attempted diff
            safe_division(r_sig_landed, r_sig_attempted, 0),  # Red sig accuracy
            safe_division(b_sig_landed, b_sig_attempted, 0),  # Blue sig accuracy
            r_sig_landed + r_total_landed,  # Red total output
            b_sig_landed + b_total_landed,  # Blue total output
            1 if (r_sig_landed + r_total_landed) > (b_sig_landed + b_total_landed) else 0,  # Red more volume
        ]
        features.append(feature_vector)

    feature_names = ['Sig_Landed_Diff', 'Sig_Attempted_Diff', 'Total_Landed_Diff', 'Total_Attempted_Diff',
                     'R_Sig_Acc', 'B_Sig_Acc', 'R_Total_Output', 'B_Total_Output', 'R_Volume_Advantage']
    return np.array(features), feature_names


def create_striking_accuracy_category_tree_features(fight_data):
    """Tree 54: Comprehensive Striking Accuracy Analysis"""
    features = []
    for row in fight_data:
        # All accuracy metrics
        r_sig_acc = row.get('r_sig_str_acc', 0) or 0
        b_sig_acc = row.get('b_sig_str_acc', 0) or 0
        r_total_acc = row.get('r_total_str_acc', 0) or 0
        b_total_acc = row.get('b_total_str_acc', 0) or 0
        r_head_acc = row.get('r_head_acc', 0) or 0
        b_head_acc = row.get('b_head_acc', 0) or 0
        r_body_acc = row.get('r_body_acc', 0) or 0
        b_body_acc = row.get('b_body_acc', 0) or 0

        feature_vector = [
            r_sig_acc - b_sig_acc,  # Sig accuracy difference
            r_total_acc - b_total_acc,  # Total accuracy difference
            r_head_acc - b_head_acc,  # Head accuracy difference
            r_body_acc - b_body_acc,  # Body accuracy difference
            (r_sig_acc + r_total_acc + r_head_acc) / 3,  # Red avg accuracy
            (b_sig_acc + b_total_acc + b_head_acc) / 3,  # Blue avg accuracy
            max(r_sig_acc, r_head_acc, r_body_acc),  # Red best accuracy
            max(b_sig_acc, b_head_acc, b_body_acc),  # Blue best accuracy
            1 if r_sig_acc > b_sig_acc else 0,  # Red more accurate
        ]
        features.append(feature_vector)

    feature_names = ['Sig_Acc_Diff', 'Total_Acc_Diff', 'Head_Acc_Diff', 'Body_Acc_Diff',
                     'R_Avg_Acc', 'B_Avg_Acc', 'R_Best_Acc', 'B_Best_Acc', 'R_Acc_Advantage']
    return np.array(features), feature_names


def create_grappling_offense_category_tree_features(fight_data):
    """Tree 55: Comprehensive Grappling Offense Analysis"""
    features = []
    for row in fight_data:
        # All grappling offense metrics
        r_td_landed = row.get('r_td_landed', 0) or 0
        b_td_landed = row.get('b_td_landed', 0) or 0
        r_td_attempted = row.get('r_td_atmpted', 0) or 0
        b_td_attempted = row.get('b_td_atmpted', 0) or 0
        r_sub_att = row.get('r_sub_att', 0) or 0
        b_sub_att = row.get('b_sub_att', 0) or 0
        r_ctrl = row.get('r_ctrl', 0) or 0
        b_ctrl = row.get('b_ctrl', 0) or 0

        feature_vector = [
            r_td_landed - b_td_landed,  # TD landed difference
            r_td_attempted - b_td_attempted,  # TD attempted difference
            r_sub_att - b_sub_att,  # Submission attempts difference
            r_ctrl - b_ctrl,  # Control time difference
            safe_division(r_td_landed, r_td_attempted, 0),  # Red TD accuracy
            safe_division(b_td_landed, b_td_attempted, 0),  # Blue TD accuracy
            r_td_landed + r_sub_att,  # Red grappling aggression
            b_td_landed + b_sub_att,  # Blue grappling aggression
            1 if (r_td_landed + r_sub_att + r_ctrl) > (b_td_landed + b_sub_att + b_ctrl) else 0,  # Red grappling advantage
        ]
        features.append(feature_vector)

    feature_names = ['TD_Landed_Diff', 'TD_Attempted_Diff', 'Sub_Att_Diff', 'Ctrl_Diff',
                     'R_TD_Acc', 'B_TD_Acc', 'R_Grappling_Agg', 'B_Grappling_Agg', 'R_Grappling_Advantage']
    return np.array(features), feature_names


def create_physical_advantage_category_tree_features(fight_data):
    """Tree 56: Comprehensive Physical Advantage Analysis"""
    features = []
    for row in fight_data:
        # All physical metrics
        r_height = row.get('r_height', 0) or 0
        b_height = row.get('b_height', 0) or 0
        r_reach = row.get('r_reach', 0) or 0
        b_reach = row.get('b_reach', 0) or 0
        r_weight = row.get('r_weight', 0) or 0
        b_weight = row.get('b_weight', 0) or 0

        height_diff = r_height - b_height
        reach_diff = r_reach - b_reach
        weight_diff = r_weight - b_weight

        feature_vector = [
            height_diff,  # Height difference
            reach_diff,  # Reach difference
            weight_diff,  # Weight difference
            abs(height_diff),  # Absolute height difference
            abs(reach_diff),  # Absolute reach difference
            abs(weight_diff),  # Absolute weight difference
            (height_diff + reach_diff) / 2,  # Combined size advantage
            1 if height_diff > 0 and reach_diff > 0 else 0,  # Red has both height and reach
            1 if abs(height_diff) > 3 or abs(reach_diff) > 3 else 0,  # Significant physical mismatch
        ]
        features.append(feature_vector)

    feature_names = ['Height_Diff', 'Reach_Diff', 'Weight_Diff', 'Abs_Height_Diff', 'Abs_Reach_Diff',
                     'Abs_Weight_Diff', 'Combined_Size_Adv', 'R_Size_Advantage', 'Significant_Mismatch']
    return np.array(features), feature_names


def create_experience_category_tree_features(fight_data):
    """Tree 57: Comprehensive Experience Analysis"""
    features = []
    for row in fight_data:
        # All experience metrics
        r_wins = row.get('r_wins', 0) or 0
        b_wins = row.get('b_wins', 0) or 0
        r_losses = row.get('r_losses', 0) or 0
        b_losses = row.get('b_losses', 0) or 0
        r_draws = row.get('r_draws', 0) or 0
        b_draws = row.get('b_draws', 0) or 0

        r_total_fights = r_wins + r_losses + r_draws
        b_total_fights = b_wins + b_losses + b_draws
        r_win_rate = safe_division(r_wins, r_total_fights, 0.5)
        b_win_rate = safe_division(b_wins, b_total_fights, 0.5)

        feature_vector = [
            r_wins - b_wins,  # Wins difference
            r_losses - b_losses,  # Losses difference
            r_total_fights - b_total_fights,  # Total fights difference
            r_win_rate - b_win_rate,  # Win rate difference
            r_total_fights + b_total_fights,  # Combined experience
            max(r_total_fights, b_total_fights),  # Most experienced
            min(r_total_fights, b_total_fights),  # Least experienced
            1 if r_win_rate > b_win_rate and r_total_fights > b_total_fights else 0,  # Red experienced and winning
            1 if abs(r_total_fights - b_total_fights) > 10 else 0,  # Significant experience gap
        ]
        features.append(feature_vector)

    feature_names = ['Wins_Diff', 'Losses_Diff', 'Fights_Diff', 'WinRate_Diff', 'Combined_Exp',
                     'Most_Exp', 'Least_Exp', 'R_Exp_Advantage', 'Significant_Exp_Gap']
    return np.array(features), feature_names


def create_grappling_defense_category_tree_features(fight_data):
    """Tree 58: Comprehensive Grappling Defense Analysis"""
    features = []
    for row in fight_data:
        # All defensive grappling metrics
        r_td_def = row.get('r_td_def', 0) or 0
        b_td_def = row.get('b_td_def', 0) or 0
        r_str_def = row.get('r_str_def', 0) or 0
        b_str_def = row.get('b_str_def', 0) or 0
        r_sapm = row.get('r_sapm', 0) or 0  # Strikes absorbed per minute
        b_sapm = row.get('b_sapm', 0) or 0

        feature_vector = [
            r_td_def - b_td_def,  # TD defense difference
            r_str_def - b_str_def,  # Striking defense difference
            b_sapm - r_sapm,  # SAPM difference (lower is better, so reversed)
            (r_td_def + r_str_def) / 2,  # Red average defense
            (b_td_def + b_str_def) / 2,  # Blue average defense
            max(r_td_def, r_str_def),  # Red best defense
            max(b_td_def, b_str_def),  # Blue best defense
            1 if r_td_def > b_td_def and r_str_def > b_str_def else 0,  # Red better at both defenses
            1 if r_sapm < b_sapm else 0,  # Red absorbs fewer strikes
        ]
        features.append(feature_vector)

    feature_names = ['TD_Def_Diff', 'Str_Def_Diff', 'SAPM_Advantage', 'R_Avg_Defense', 'B_Avg_Defense',
                     'R_Best_Defense', 'B_Best_Defense', 'R_Defense_Advantage', 'R_Absorbs_Less']
    return np.array(features), feature_names


def create_power_finishing_category_tree_features(fight_data):
    """Tree 59: Comprehensive Power and Finishing Analysis"""
    features = []
    for row in fight_data:
        # Power and finishing metrics
        r_kd = row.get('r_kd', 0) or 0
        b_kd = row.get('b_kd', 0) or 0
        r_splm = row.get('r_splm', 0) or 0  # Strikes landed per minute
        b_splm = row.get('b_splm', 0) or 0
        r_sig_landed = row.get('r_sig_str_landed', 0) or 0
        b_sig_landed = row.get('b_sig_str_landed', 0) or 0
        r_head_landed = row.get('r_head_landed', 0) or 0
        b_head_landed = row.get('b_head_landed', 0) or 0

        # Calculate power ratios
        r_power_ratio = safe_division(r_kd, r_sig_landed, 0) * 100  # Knockdowns per 100 strikes
        b_power_ratio = safe_division(b_kd, b_sig_landed, 0) * 100

        feature_vector = [
            r_kd - b_kd,  # Knockdown difference
            r_splm - b_splm,  # SPLM difference
            r_head_landed - b_head_landed,  # Head strikes difference
            r_power_ratio - b_power_ratio,  # Power ratio difference
            r_kd + r_head_landed,  # Red power output
            b_kd + b_head_landed,  # Blue power output
            max(r_power_ratio, 0),  # Red power efficiency
            max(b_power_ratio, 0),  # Blue power efficiency
            1 if r_kd > 0 and r_kd > b_kd else 0,  # Red has knockdown advantage
        ]
        features.append(feature_vector)

    feature_names = ['KD_Diff', 'SPLM_Diff', 'Head_Strikes_Diff', 'Power_Ratio_Diff', 'R_Power_Output',
                     'B_Power_Output', 'R_Power_Efficiency', 'B_Power_Efficiency', 'R_KD_Advantage']
    return np.array(features), feature_names


def create_strike_location_category_tree_features(fight_data):
    """Tree 60: Comprehensive Strike Location Analysis"""
    features = []
    for row in fight_data:
        # Strike location metrics
        r_head = row.get('r_head_landed', 0) or 0
        b_head = row.get('b_head_landed', 0) or 0
        r_body = row.get('r_body_landed', 0) or 0
        b_body = row.get('b_body_landed', 0) or 0
        r_leg = row.get('r_leg_landed', 0) or 0
        b_leg = row.get('b_leg_landed', 0) or 0

        r_total_location = r_head + r_body + r_leg
        b_total_location = b_head + b_body + b_leg

        feature_vector = [
            r_head - b_head,  # Head strikes difference
            r_body - b_body,  # Body strikes difference
            r_leg - b_leg,  # Leg strikes difference
            safe_division(r_head, r_total_location, 0),  # Red head targeting %
            safe_division(b_head, b_total_location, 0),  # Blue head targeting %
            safe_division(r_leg, r_total_location, 0),  # Red leg targeting %
            safe_division(b_leg, b_total_location, 0),  # Blue leg targeting %
            max(r_head, r_body, r_leg),  # Red strongest location
            max(b_head, b_body, b_leg),  # Blue strongest location
        ]
        features.append(feature_vector)

    feature_names = ['Head_Diff', 'Body_Diff', 'Leg_Diff', 'R_Head_Focus', 'B_Head_Focus',
                     'R_Leg_Focus', 'B_Leg_Focus', 'R_Strongest_Location', 'B_Strongest_Location']
    return np.array(features), feature_names


def create_strike_position_category_tree_features(fight_data):
    """Tree 61: Comprehensive Strike Position Analysis"""
    features = []
    for row in fight_data:
        # Strike position metrics
        r_dist = row.get('r_dist_landed', 0) or 0
        b_dist = row.get('b_dist_landed', 0) or 0
        r_clinch = row.get('r_clinch_landed', 0) or 0
        b_clinch = row.get('b_clinch_landed', 0) or 0
        r_ground = row.get('r_ground_landed', 0) or 0
        b_ground = row.get('b_ground_landed', 0) or 0

        r_total_position = r_dist + r_clinch + r_ground
        b_total_position = b_dist + b_clinch + b_ground

        feature_vector = [
            r_dist - b_dist,  # Distance strikes difference
            r_clinch - b_clinch,  # Clinch strikes difference
            r_ground - b_ground,  # Ground strikes difference
            safe_division(r_dist, r_total_position, 0),  # Red distance fighting %
            safe_division(b_dist, b_total_position, 0),  # Blue distance fighting %
            safe_division(r_ground, r_total_position, 0),  # Red ground fighting %
            safe_division(b_ground, b_total_position, 0),  # Blue ground fighting %
            max(r_dist, r_clinch, r_ground),  # Red strongest position
            max(b_dist, b_clinch, b_ground),  # Blue strongest position
        ]
        features.append(feature_vector)

    feature_names = ['Dist_Diff', 'Clinch_Diff', 'Ground_Diff', 'R_Distance_Focus', 'B_Distance_Focus',
                     'R_Ground_Focus', 'B_Ground_Focus', 'R_Strongest_Position', 'B_Strongest_Position']
    return np.array(features), feature_names


def create_fight_pace_category_tree_features(fight_data):
    """Tree 62: Comprehensive Fight Pace Analysis"""
    features = []
    for row in fight_data:
        # Pace and tempo metrics
        r_sig_attempted = row.get('r_sig_str_atmpted', 0) or 0
        b_sig_attempted = row.get('b_sig_str_atmpted', 0) or 0
        r_total_attempted = row.get('r_total_str_atmpted', 0) or 0
        b_total_attempted = row.get('b_total_str_atmpted', 0) or 0
        r_td_attempted = row.get('r_td_atmpted', 0) or 0
        b_td_attempted = row.get('b_td_atmpted', 0) or 0
        r_sub_att = row.get('r_sub_att', 0) or 0
        b_sub_att = row.get('b_sub_att', 0) or 0

        r_total_aggression = r_sig_attempted + r_total_attempted + r_td_attempted + r_sub_att
        b_total_aggression = b_sig_attempted + b_total_attempted + b_td_attempted + b_sub_att

        feature_vector = [
            r_sig_attempted - b_sig_attempted,  # Sig strikes attempted diff
            r_total_attempted - b_total_attempted,  # Total strikes attempted diff
            r_td_attempted - b_td_attempted,  # TD attempted diff
            r_total_aggression - b_total_aggression,  # Total aggression diff
            r_total_aggression + b_total_aggression,  # Combined pace
            max(r_total_aggression, b_total_aggression),  # Highest pace
            min(r_total_aggression, b_total_aggression),  # Lowest pace
            1 if r_total_aggression > b_total_aggression else 0,  # Red more aggressive
            1 if (r_total_aggression + b_total_aggression) > 50 else 0,  # High pace fight
        ]
        features.append(feature_vector)

    feature_names = ['Sig_Att_Diff', 'Total_Att_Diff', 'TD_Att_Diff', 'Aggression_Diff', 'Combined_Pace',
                     'Max_Pace', 'Min_Pace', 'R_More_Aggressive', 'High_Pace_Fight']
    return np.array(features), feature_names


def create_comprehensive_distribution_category_tree_features(fight_data):
    """Tree 63: Comprehensive Strike Distribution Analysis"""
    features = []
    for row in fight_data:
        # All distribution percentages
        r_head_per = row.get('r_landed_head_per', 0) or 0
        b_head_per = row.get('b_landed_head_per', 0) or 0
        r_body_per = row.get('r_landed_body_per', 0) or 0
        b_body_per = row.get('b_landed_body_per', 0) or 0
        r_leg_per = row.get('r_landed_leg_per', 0) or 0
        b_leg_per = row.get('b_landed_leg_per', 0) or 0
        r_dist_per = row.get('r_landed_dist_per', 0) or 0
        b_dist_per = row.get('b_landed_dist_per', 0) or 0

        feature_vector = [
            r_head_per - b_head_per,  # Head targeting difference
            r_body_per - b_body_per,  # Body targeting difference
            r_leg_per - b_leg_per,  # Leg targeting difference
            r_dist_per - b_dist_per,  # Distance fighting difference
            (r_head_per + r_body_per + r_leg_per) / 3,  # Red location diversity
            (b_head_per + b_body_per + b_leg_per) / 3,  # Blue location diversity
            max(r_head_per, r_body_per, r_leg_per),  # Red primary target
            max(b_head_per, b_body_per, b_leg_per),  # Blue primary target
            1 if r_head_per > 50 else 0,  # Red head hunter
        ]
        features.append(feature_vector)

    feature_names = ['Head_Target_Diff', 'Body_Target_Diff', 'Leg_Target_Diff', 'Distance_Fight_Diff',
                     'R_Location_Diversity', 'B_Location_Diversity', 'R_Primary_Target', 'B_Primary_Target', 'R_Head_Hunter']
    return np.array(features), feature_names


# ============================================================================
# CROSS-CATEGORY COMBINATION TREES - PHASE 3 (24 trees)
# ============================================================================

def create_volume_vs_accuracy_cross_tree_features(fight_data):
    """Tree 64: Volume vs Accuracy Cross-Analysis"""
    features = []
    for row in fight_data:
        # Volume metrics
        r_sig_attempted = row.get('r_sig_str_atmpted', 0) or 0
        b_sig_attempted = row.get('b_sig_str_atmpted', 0) or 0
        r_total_attempted = row.get('r_total_str_atmpted', 0) or 0
        b_total_attempted = row.get('b_total_str_atmpted', 0) or 0

        # Accuracy metrics
        r_sig_acc = row.get('r_sig_str_acc', 0) or 0
        b_sig_acc = row.get('b_sig_str_acc', 0) or 0
        r_head_acc = row.get('r_head_acc', 0) or 0
        b_head_acc = row.get('b_head_acc', 0) or 0

        # Calculate volume scores
        r_volume_score = r_sig_attempted + r_total_attempted
        b_volume_score = b_sig_attempted + b_total_attempted

        # Calculate accuracy scores
        r_accuracy_score = (r_sig_acc + r_head_acc) / 2
        b_accuracy_score = (b_sig_acc + b_head_acc) / 2

        feature_vector = [
            r_volume_score - b_volume_score,  # Volume difference
            r_accuracy_score - b_accuracy_score,  # Accuracy difference
            r_volume_score * r_accuracy_score,  # Red volume×accuracy product
            b_volume_score * b_accuracy_score,  # Blue volume×accuracy product
            safe_division(r_accuracy_score, r_volume_score, 0) * 100,  # Red efficiency ratio
            safe_division(b_accuracy_score, b_volume_score, 0) * 100,  # Blue efficiency ratio
            1 if r_volume_score > b_volume_score and r_accuracy_score > b_accuracy_score else 0,  # Red dominance
            1 if r_volume_score > 100 and r_accuracy_score > 50 else 0,  # Red high volume + accuracy
            abs(r_accuracy_score - b_accuracy_score) / max(r_accuracy_score, b_accuracy_score, 1),  # Accuracy gap ratio
        ]
        features.append(feature_vector)

    feature_names = ['Volume_Diff', 'Accuracy_Diff', 'R_Vol_Acc_Product', 'B_Vol_Acc_Product',
                     'R_Efficiency', 'B_Efficiency', 'R_Dominance', 'R_High_Vol_Acc', 'Accuracy_Gap_Ratio']
    return np.array(features), feature_names


def create_power_vs_grappling_cross_tree_features(fight_data):
    """Tree 65: Power vs Grappling Cross-Analysis"""
    features = []
    for row in fight_data:
        # Power metrics
        r_kd = row.get('r_kd', 0) or 0
        b_kd = row.get('b_kd', 0) or 0
        r_head_landed = row.get('r_head_landed', 0) or 0
        b_head_landed = row.get('b_head_landed', 0) or 0

        # Grappling metrics
        r_td_landed = row.get('r_td_landed', 0) or 0
        b_td_landed = row.get('b_td_landed', 0) or 0
        r_ctrl = row.get('r_ctrl', 0) or 0
        b_ctrl = row.get('b_ctrl', 0) or 0
        r_sub_att = row.get('r_sub_att', 0) or 0
        b_sub_att = row.get('b_sub_att', 0) or 0

        # Calculate composite scores
        r_power_score = r_kd * 10 + r_head_landed  # Weight knockdowns heavily
        b_power_score = b_kd * 10 + b_head_landed
        r_grappling_score = r_td_landed * 5 + r_ctrl + r_sub_att * 3
        b_grappling_score = b_td_landed * 5 + b_ctrl + b_sub_att * 3

        feature_vector = [
            r_power_score - b_power_score,  # Power difference
            r_grappling_score - b_grappling_score,  # Grappling difference
            r_power_score + r_grappling_score,  # Red total threat
            b_power_score + b_grappling_score,  # Blue total threat
            safe_division(r_power_score, r_grappling_score + 1, 1),  # Red power/grappling ratio
            safe_division(b_power_score, b_grappling_score + 1, 1),  # Blue power/grappling ratio
            1 if r_power_score > b_power_score and r_grappling_score < b_grappling_score else 0,  # Red striker vs grappler
            1 if r_grappling_score > b_grappling_score and r_power_score < b_power_score else 0,  # Red grappler vs striker
            max(r_power_score, r_grappling_score) - min(r_power_score, r_grappling_score),  # Red specialization gap
        ]
        features.append(feature_vector)

    feature_names = ['Power_Diff', 'Grappling_Diff', 'R_Total_Threat', 'B_Total_Threat',
                     'R_Power_Grappling_Ratio', 'B_Power_Grappling_Ratio', 'R_Striker_vs_Grappler',
                     'R_Grappler_vs_Striker', 'R_Specialization_Gap']
    return np.array(features), feature_names


def create_experience_vs_physical_cross_tree_features(fight_data):
    """Tree 66: Experience vs Physical Cross-Analysis"""
    features = []
    for row in fight_data:
        # Experience metrics
        r_wins = row.get('r_wins', 0) or 0
        b_wins = row.get('b_wins', 0) or 0
        r_losses = row.get('r_losses', 0) or 0
        b_losses = row.get('b_losses', 0) or 0

        # Physical metrics
        r_height = row.get('r_height', 0) or 0
        b_height = row.get('b_height', 0) or 0
        r_reach = row.get('r_reach', 0) or 0
        b_reach = row.get('b_reach', 0) or 0

        # Calculate scores
        r_total_fights = r_wins + r_losses
        b_total_fights = b_wins + b_losses
        r_win_rate = safe_division(r_wins, r_total_fights, 0.5)
        b_win_rate = safe_division(b_wins, b_total_fights, 0.5)

        r_experience_score = r_total_fights * r_win_rate  # Experience weighted by success
        b_experience_score = b_total_fights * b_win_rate

        r_physical_score = (r_height - 70) + (r_reach - 70)  # Normalized physical advantages
        b_physical_score = (b_height - 70) + (b_reach - 70)

        feature_vector = [
            r_experience_score - b_experience_score,  # Experience difference
            r_physical_score - b_physical_score,  # Physical difference
            r_experience_score * (1 + r_physical_score/10),  # Red experience + physical boost
            b_experience_score * (1 + b_physical_score/10),  # Blue experience + physical boost
            safe_division(r_experience_score, abs(r_physical_score) + 1, 1),  # Red exp/physical ratio
            safe_division(b_experience_score, abs(b_physical_score) + 1, 1),  # Blue exp/physical ratio
            1 if r_experience_score > b_experience_score and r_physical_score < b_physical_score else 0,  # Red veteran vs bigger
            1 if r_total_fights > 20 and r_physical_score > 5 else 0,  # Red experienced and big
            abs(r_experience_score - b_experience_score) + abs(r_physical_score - b_physical_score),  # Total advantage gap
        ]
        features.append(feature_vector)

    feature_names = ['Experience_Diff', 'Physical_Diff', 'R_Exp_Phys_Boost', 'B_Exp_Phys_Boost',
                     'R_Exp_Phys_Ratio', 'B_Exp_Phys_Ratio', 'R_Veteran_vs_Bigger', 'R_Exp_and_Big', 'Total_Advantage_Gap']
    return np.array(features), feature_names


def create_defense_vs_offense_cross_tree_features(fight_data):
    """Tree 67: Defense vs Offense Cross-Analysis"""
    features = []
    for row in fight_data:
        # Defensive metrics
        r_str_def = row.get('r_str_def', 0) or 0
        b_str_def = row.get('b_str_def', 0) or 0
        r_td_def = row.get('r_td_def', 0) or 0
        b_td_def = row.get('b_td_def', 0) or 0
        r_sapm = row.get('r_sapm', 0) or 0
        b_sapm = row.get('b_sapm', 0) or 0

        # Offensive metrics
        r_sig_landed = row.get('r_sig_str_landed', 0) or 0
        b_sig_landed = row.get('b_sig_str_landed', 0) or 0
        r_td_landed = row.get('r_td_landed', 0) or 0
        b_td_landed = row.get('b_td_landed', 0) or 0
        r_kd = row.get('r_kd', 0) or 0
        b_kd = row.get('b_kd', 0) or 0

        # Calculate composite scores
        r_defense_score = (r_str_def + r_td_def) / 2 - r_sapm  # Higher defense, lower absorption
        b_defense_score = (b_str_def + b_td_def) / 2 - b_sapm
        r_offense_score = r_sig_landed + r_td_landed * 3 + r_kd * 10
        b_offense_score = b_sig_landed + b_td_landed * 3 + b_kd * 10

        feature_vector = [
            r_defense_score - b_defense_score,  # Defense difference
            r_offense_score - b_offense_score,  # Offense difference
            r_defense_score + r_offense_score,  # Red total game
            b_defense_score + b_offense_score,  # Blue total game
            safe_division(r_offense_score, r_defense_score + 50, 1),  # Red offense/defense ratio
            safe_division(b_offense_score, b_defense_score + 50, 1),  # Blue offense/defense ratio
            1 if r_defense_score > b_defense_score and r_offense_score > b_offense_score else 0,  # Red two-way excellence
            1 if r_defense_score > 60 and r_offense_score > 50 else 0,  # Red well-rounded
            abs(r_defense_score - r_offense_score),  # Red balance (lower = more balanced)
        ]
        features.append(feature_vector)

    feature_names = ['Defense_Diff', 'Offense_Diff', 'R_Total_Game', 'B_Total_Game',
                     'R_Off_Def_Ratio', 'B_Off_Def_Ratio', 'R_Two_Way_Excel', 'R_Well_Rounded', 'R_Balance']
    return np.array(features), feature_names


# ============================================================================
# STRIKING PERFORMANCE - ACCURACY TREES (8 trees)
# ============================================================================

def create_sig_strikes_accuracy_tree_features(fight_data):
    """Tree 9: Significant Strike Accuracy Comparison"""
    features = []
    for row in fight_data:
        r_acc = row.get('r_sig_str_acc', 0) or 0
        b_acc = row.get('b_sig_str_acc', 0) or 0

        feature_vector = [
            r_acc - b_acc,  # Accuracy difference
            r_acc,          # Red accuracy
            b_acc,          # Blue accuracy
            (r_acc + b_acc) / 2,  # Average accuracy
            1 if r_acc > b_acc else 0,  # Red has better accuracy
        ]
        features.append(feature_vector)

    feature_names = ['Sig_Acc_Diff', 'R_Sig_Acc', 'B_Sig_Acc', 'Avg_Sig_Acc', 'R_Sig_Acc_Advantage']
    return np.array(features), feature_names


def create_total_strikes_accuracy_tree_features(fight_data):
    """Tree 10: Total Strike Accuracy Comparison"""
    features = []
    for row in fight_data:
        r_acc = row.get('r_total_str_acc', 0) or 0
        b_acc = row.get('b_total_str_acc', 0) or 0

        feature_vector = [
            r_acc - b_acc,  # Accuracy difference
            r_acc,          # Red accuracy
            b_acc,          # Blue accuracy
            (r_acc + b_acc) / 2,  # Average accuracy
            1 if r_acc > b_acc else 0,  # Red has better accuracy
        ]
        features.append(feature_vector)

    feature_names = ['Total_Acc_Diff', 'R_Total_Acc', 'B_Total_Acc', 'Avg_Total_Acc', 'R_Total_Acc_Advantage']
    return np.array(features), feature_names


def create_head_strikes_accuracy_tree_features(fight_data):
    """Tree 11: Head Strike Accuracy Comparison"""
    features = []
    for row in fight_data:
        r_acc = row.get('r_head_acc', 0) or 0
        b_acc = row.get('b_head_acc', 0) or 0

        feature_vector = [
            r_acc - b_acc,  # Accuracy difference
            r_acc,          # Red accuracy
            b_acc,          # Blue accuracy
            (r_acc + b_acc) / 2,  # Average accuracy
            1 if r_acc > b_acc else 0,  # Red has better accuracy
        ]
        features.append(feature_vector)

    feature_names = ['Head_Acc_Diff', 'R_Head_Acc', 'B_Head_Acc', 'Avg_Head_Acc', 'R_Head_Acc_Advantage']
    return np.array(features), feature_names


def create_body_strikes_accuracy_tree_features(fight_data):
    """Tree 12: Body Strike Accuracy Comparison"""
    features = []
    for row in fight_data:
        r_acc = row.get('r_body_acc', 0) or 0
        b_acc = row.get('b_body_acc', 0) or 0

        feature_vector = [
            r_acc - b_acc,  # Accuracy difference
            r_acc,          # Red accuracy
            b_acc,          # Blue accuracy
            (r_acc + b_acc) / 2,  # Average accuracy
            1 if r_acc > b_acc else 0,  # Red has better accuracy
        ]
        features.append(feature_vector)

    feature_names = ['Body_Acc_Diff', 'R_Body_Acc', 'B_Body_Acc', 'Avg_Body_Acc', 'R_Body_Acc_Advantage']
    return np.array(features), feature_names


def create_leg_strikes_accuracy_tree_features(fight_data):
    """Tree 13: Leg Strike Accuracy Comparison"""
    features = []
    for row in fight_data:
        r_acc = row.get('r_leg_acc', 0) or 0
        b_acc = row.get('b_leg_acc', 0) or 0

        feature_vector = [
            r_acc - b_acc,  # Accuracy difference
            r_acc,          # Red accuracy
            b_acc,          # Blue accuracy
            (r_acc + b_acc) / 2,  # Average accuracy
            1 if r_acc > b_acc else 0,  # Red has better accuracy
        ]
        features.append(feature_vector)

    feature_names = ['Leg_Acc_Diff', 'R_Leg_Acc', 'B_Leg_Acc', 'Avg_Leg_Acc', 'R_Leg_Acc_Advantage']
    return np.array(features), feature_names


def create_head_strikes_attempted_tree_features(fight_data):
    """Tree 14: Head Strikes Attempted Comparison"""
    features = []
    for row in fight_data:
        r_attempted = row.get('r_head_atmpted', 0) or 0
        b_attempted = row.get('b_head_atmpted', 0) or 0

        feature_vector = [
            r_attempted - b_attempted,  # Difference
            r_attempted,                # Red attempted
            b_attempted,                # Blue attempted
            r_attempted + b_attempted,  # Total attempted
            safe_division(r_attempted, b_attempted, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Head_Attempted_Diff', 'R_Head_Attempted', 'B_Head_Attempted', 'Total_Head_Attempted', 'Head_Attempted_Ratio']
    return np.array(features), feature_names


def create_body_strikes_attempted_tree_features(fight_data):
    """Tree 15: Body Strikes Attempted Comparison"""
    features = []
    for row in fight_data:
        r_attempted = row.get('r_body_atmpted', 0) or 0
        b_attempted = row.get('b_body_atmpted', 0) or 0

        feature_vector = [
            r_attempted - b_attempted,  # Difference
            r_attempted,                # Red attempted
            b_attempted,                # Blue attempted
            r_attempted + b_attempted,  # Total attempted
            safe_division(r_attempted, b_attempted, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Body_Attempted_Diff', 'R_Body_Attempted', 'B_Body_Attempted', 'Total_Body_Attempted', 'Body_Attempted_Ratio']
    return np.array(features), feature_names


def create_leg_strikes_attempted_tree_features(fight_data):
    """Tree 16: Leg Strikes Attempted Comparison"""
    features = []
    for row in fight_data:
        r_attempted = row.get('r_leg_atmpted', 0) or 0
        b_attempted = row.get('b_leg_atmpted', 0) or 0

        feature_vector = [
            r_attempted - b_attempted,  # Difference
            r_attempted,                # Red attempted
            b_attempted,                # Blue attempted
            r_attempted + b_attempted,  # Total attempted
            safe_division(r_attempted, b_attempted, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Leg_Attempted_Diff', 'R_Leg_Attempted', 'B_Leg_Attempted', 'Total_Leg_Attempted', 'Leg_Attempted_Ratio']
    return np.array(features), feature_names


# ============================================================================
# GRAPPLING PERFORMANCE - LANDED TREES (3 trees)
# ============================================================================

def create_takedowns_landed_tree_features(fight_data):
    """Tree 17: Takedowns Landed Comparison"""
    features = []
    for row in fight_data:
        r_td = row.get('r_td_landed', 0) or 0
        b_td = row.get('b_td_landed', 0) or 0

        feature_vector = [
            r_td - b_td,  # Takedown difference
            r_td,         # Red takedowns
            b_td,         # Blue takedowns
            r_td + b_td,  # Total takedowns
            1 if r_td > b_td else 0,  # Red has more takedowns
        ]
        features.append(feature_vector)

    feature_names = ['TD_Landed_Diff', 'R_TD_Landed', 'B_TD_Landed', 'Total_TD_Landed', 'R_TD_Advantage']
    return np.array(features), feature_names


def create_takedowns_attempted_tree_features(fight_data):
    """Tree 18: Takedowns Attempted Comparison"""
    features = []
    for row in fight_data:
        r_attempted = row.get('r_td_atmpted', 0) or 0
        b_attempted = row.get('b_td_atmpted', 0) or 0

        feature_vector = [
            r_attempted - b_attempted,  # Difference
            r_attempted,                # Red attempted
            b_attempted,                # Blue attempted
            r_attempted + b_attempted,  # Total attempted
            safe_division(r_attempted, b_attempted, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['TD_Attempted_Diff', 'R_TD_Attempted', 'B_TD_Attempted', 'Total_TD_Attempted', 'TD_Attempted_Ratio']
    return np.array(features), feature_names


def create_submission_attempts_tree_features(fight_data):
    """Tree 19: Submission Attempts Comparison"""
    features = []
    for row in fight_data:
        r_sub = row.get('r_sub_att', 0) or 0
        b_sub = row.get('b_sub_att', 0) or 0

        feature_vector = [
            r_sub - b_sub,  # Submission difference
            r_sub,          # Red submissions
            b_sub,          # Blue submissions
            r_sub + b_sub,  # Total submissions
            1 if r_sub > b_sub else 0,  # Red has more submissions
        ]
        features.append(feature_vector)

    feature_names = ['Sub_Att_Diff', 'R_Sub_Att', 'B_Sub_Att', 'Total_Sub_Att', 'R_Sub_Advantage']
    return np.array(features), feature_names


# ============================================================================
# GRAPPLING PERFORMANCE - ACCURACY/CONTROL TREES (2 trees)
# ============================================================================

def create_takedown_accuracy_tree_features(fight_data):
    """Tree 20: Takedown Accuracy Comparison"""
    features = []
    for row in fight_data:
        r_acc = row.get('r_td_acc', 0) or 0
        b_acc = row.get('b_td_acc', 0) or 0

        feature_vector = [
            r_acc - b_acc,  # Accuracy difference
            r_acc,          # Red accuracy
            b_acc,          # Blue accuracy
            (r_acc + b_acc) / 2,  # Average accuracy
            1 if r_acc > b_acc else 0,  # Red has better accuracy
        ]
        features.append(feature_vector)

    feature_names = ['TD_Acc_Diff', 'R_TD_Acc', 'B_TD_Acc', 'Avg_TD_Acc', 'R_TD_Acc_Advantage']
    return np.array(features), feature_names


def create_control_time_tree_features(fight_data):
    """Tree 21: Control Time Comparison"""
    features = []
    for row in fight_data:
        r_ctrl = row.get('r_ctrl', 0) or 0
        b_ctrl = row.get('b_ctrl', 0) or 0

        feature_vector = [
            r_ctrl - b_ctrl,  # Control time difference
            r_ctrl,           # Red control time
            b_ctrl,           # Blue control time
            r_ctrl + b_ctrl,  # Total control time
            safe_division(r_ctrl, b_ctrl, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Ctrl_Diff', 'R_Ctrl', 'B_Ctrl', 'Total_Ctrl', 'Ctrl_Ratio']
    return np.array(features), feature_names


# ============================================================================
# POSITIONAL FIGHTING - LANDED TREES (6 trees)
# ============================================================================

def create_distance_strikes_landed_tree_features(fight_data):
    """Tree 22: Distance Strikes Landed Comparison"""
    features = []
    for row in fight_data:
        r_landed = row.get('r_dist_landed', 0) or 0
        b_landed = row.get('b_dist_landed', 0) or 0

        feature_vector = [
            r_landed - b_landed,  # Difference
            r_landed,             # Red landed
            b_landed,             # Blue landed
            r_landed + b_landed,  # Total landed
            safe_division(r_landed, b_landed, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Dist_Landed_Diff', 'R_Dist_Landed', 'B_Dist_Landed', 'Total_Dist_Landed', 'Dist_Landed_Ratio']
    return np.array(features), feature_names


def create_distance_strikes_attempted_tree_features(fight_data):
    """Tree 23: Distance Strikes Attempted Comparison"""
    features = []
    for row in fight_data:
        r_attempted = row.get('r_dist_atmpted', 0) or 0
        b_attempted = row.get('b_dist_atmpted', 0) or 0

        feature_vector = [
            r_attempted - b_attempted,  # Difference
            r_attempted,                # Red attempted
            b_attempted,                # Blue attempted
            r_attempted + b_attempted,  # Total attempted
            safe_division(r_attempted, b_attempted, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Dist_Attempted_Diff', 'R_Dist_Attempted', 'B_Dist_Attempted', 'Total_Dist_Attempted', 'Dist_Attempted_Ratio']
    return np.array(features), feature_names


def create_clinch_strikes_landed_tree_features(fight_data):
    """Tree 24: Clinch Strikes Landed Comparison"""
    features = []
    for row in fight_data:
        r_landed = row.get('r_clinch_landed', 0) or 0
        b_landed = row.get('b_clinch_landed', 0) or 0

        feature_vector = [
            r_landed - b_landed,  # Difference
            r_landed,             # Red landed
            b_landed,             # Blue landed
            r_landed + b_landed,  # Total landed
            safe_division(r_landed, b_landed, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Clinch_Landed_Diff', 'R_Clinch_Landed', 'B_Clinch_Landed', 'Total_Clinch_Landed', 'Clinch_Landed_Ratio']
    return np.array(features), feature_names


def create_clinch_strikes_attempted_tree_features(fight_data):
    """Tree 25: Clinch Strikes Attempted Comparison"""
    features = []
    for row in fight_data:
        r_attempted = row.get('r_clinch_atmpted', 0) or 0
        b_attempted = row.get('b_clinch_atmpted', 0) or 0

        feature_vector = [
            r_attempted - b_attempted,  # Difference
            r_attempted,                # Red attempted
            b_attempted,                # Blue attempted
            r_attempted + b_attempted,  # Total attempted
            safe_division(r_attempted, b_attempted, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Clinch_Attempted_Diff', 'R_Clinch_Attempted', 'B_Clinch_Attempted', 'Total_Clinch_Attempted', 'Clinch_Attempted_Ratio']
    return np.array(features), feature_names


def create_ground_strikes_landed_tree_features(fight_data):
    """Tree 26: Ground Strikes Landed Comparison"""
    features = []
    for row in fight_data:
        r_landed = row.get('r_ground_landed', 0) or 0
        b_landed = row.get('b_ground_landed', 0) or 0

        feature_vector = [
            r_landed - b_landed,  # Difference
            r_landed,             # Red landed
            b_landed,             # Blue landed
            r_landed + b_landed,  # Total landed
            safe_division(r_landed, b_landed, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Ground_Landed_Diff', 'R_Ground_Landed', 'B_Ground_Landed', 'Total_Ground_Landed', 'Ground_Landed_Ratio']
    return np.array(features), feature_names


def create_ground_strikes_attempted_tree_features(fight_data):
    """Tree 27: Ground Strikes Attempted Comparison"""
    features = []
    for row in fight_data:
        r_attempted = row.get('r_ground_atmpted', 0) or 0
        b_attempted = row.get('b_ground_atmpted', 0) or 0

        feature_vector = [
            r_attempted - b_attempted,  # Difference
            r_attempted,                # Red attempted
            b_attempted,                # Blue attempted
            r_attempted + b_attempted,  # Total attempted
            safe_division(r_attempted, b_attempted, 1),  # Ratio
        ]
        features.append(feature_vector)

    feature_names = ['Ground_Attempted_Diff', 'R_Ground_Attempted', 'B_Ground_Attempted', 'Total_Ground_Attempted', 'Ground_Attempted_Ratio']
    return np.array(features), feature_names


# ============================================================================
# POSITIONAL FIGHTING - ACCURACY TREES (3 trees)
# ============================================================================

def create_distance_strikes_accuracy_tree_features(fight_data):
    """Tree 28: Distance Strike Accuracy Comparison"""
    features = []
    for row in fight_data:
        r_acc = row.get('r_dist_acc', 0) or 0
        b_acc = row.get('b_dist_acc', 0) or 0

        feature_vector = [
            r_acc - b_acc,  # Accuracy difference
            r_acc,          # Red accuracy
            b_acc,          # Blue accuracy
            (r_acc + b_acc) / 2,  # Average accuracy
            1 if r_acc > b_acc else 0,  # Red has better accuracy
        ]
        features.append(feature_vector)

    feature_names = ['Dist_Acc_Diff', 'R_Dist_Acc', 'B_Dist_Acc', 'Avg_Dist_Acc', 'R_Dist_Acc_Advantage']
    return np.array(features), feature_names


def create_clinch_strikes_accuracy_tree_features(fight_data):
    """Tree 29: Clinch Strike Accuracy Comparison"""
    features = []
    for row in fight_data:
        r_acc = row.get('r_clinch_acc', 0) or 0
        b_acc = row.get('b_clinch_acc', 0) or 0

        feature_vector = [
            r_acc - b_acc,  # Accuracy difference
            r_acc,          # Red accuracy
            b_acc,          # Blue accuracy
            (r_acc + b_acc) / 2,  # Average accuracy
            1 if r_acc > b_acc else 0,  # Red has better accuracy
        ]
        features.append(feature_vector)

    feature_names = ['Clinch_Acc_Diff', 'R_Clinch_Acc', 'B_Clinch_Acc', 'Avg_Clinch_Acc', 'R_Clinch_Acc_Advantage']
    return np.array(features), feature_names


def create_ground_strikes_accuracy_tree_features(fight_data):
    """Tree 30: Ground Strike Accuracy Comparison"""
    features = []
    for row in fight_data:
        r_acc = row.get('r_ground_acc', 0) or 0
        b_acc = row.get('b_ground_acc', 0) or 0

        feature_vector = [
            r_acc - b_acc,  # Accuracy difference
            r_acc,          # Red accuracy
            b_acc,          # Blue accuracy
            (r_acc + b_acc) / 2,  # Average accuracy
            1 if r_acc > b_acc else 0,  # Red has better accuracy
        ]
        features.append(feature_vector)

    feature_names = ['Ground_Acc_Diff', 'R_Ground_Acc', 'B_Ground_Acc', 'Avg_Ground_Acc', 'R_Ground_Acc_Advantage']
    return np.array(features), feature_names


# ============================================================================
# FIGHT CONTEXT TREES (2 trees)
# ============================================================================

def create_fight_format_tree_features(fight_data):
    """Tree 31: Fight Format Comparison"""
    features = []

    # Prepare encoders
    method_encoder = LabelEncoder()
    all_methods = []
    for row in fight_data:
        method = str(row.get('method', 'Unknown'))
        if method.lower() == 'nan' or pd.isna(method):
            method = 'Unknown'
        all_methods.append(method)
    method_encoder.fit(all_methods)

    for row in fight_data:
        title_fight = row.get('title_fight', 0) or 0
        total_rounds = row.get('total_rounds', 3) or 3
        method = str(row.get('method', 'Unknown'))
        if method.lower() == 'nan' or pd.isna(method):
            method = 'Unknown'

        method_encoded = method_encoder.transform([method])[0]

        feature_vector = [
            title_fight,      # Is title fight
            total_rounds,     # Number of rounds
            method_encoded,   # Fight method
            1 if total_rounds == 5 else 0,  # Main event indicator
            1 if 'Decision' in method else 0,  # Went to decision
        ]
        features.append(feature_vector)

    feature_names = ['Title_Fight', 'Total_Rounds', 'Method_Encoded', 'Main_Event', 'Decision_Finish']
    return np.array(features), feature_names


def create_fight_timing_tree_features(fight_data):
    """Tree 32: Fight Timing Comparison"""
    features = []

    # Prepare encoders
    referee_encoder = LabelEncoder()
    all_referees = []
    for row in fight_data:
        referee = str(row.get('referee', 'Unknown'))
        if referee.lower() == 'nan' or pd.isna(referee):
            referee = 'Unknown'
        all_referees.append(referee)
    referee_encoder.fit(all_referees)

    for row in fight_data:
        finish_round = row.get('finish_round', 3) or 3
        match_time_sec = row.get('match_time_sec', 900) or 900
        total_rounds = row.get('total_rounds', 3) or 3
        referee = str(row.get('referee', 'Unknown'))
        if referee.lower() == 'nan' or pd.isna(referee):
            referee = 'Unknown'

        referee_encoded = referee_encoder.transform([referee])[0]

        # Calculate derived features
        duration_ratio = match_time_sec / (total_rounds * 300)
        early_finish = 1 if finish_round < total_rounds else 0

        feature_vector = [
            finish_round,         # Round fight ended
            match_time_sec,       # Time in seconds
            referee_encoded,      # Referee
            duration_ratio,       # Proportion of total time
            early_finish,         # Early finish indicator
        ]
        features.append(feature_vector)

    feature_names = ['Finish_Round', 'Match_Time_Sec', 'Referee_Encoded', 'Duration_Ratio', 'Early_Finish']
    return np.array(features), feature_names


# ============================================================================
# DECISION TREE TRAINING FUNCTIONS
# ============================================================================

# Dictionary mapping tree names to their feature functions
TREE_FEATURE_FUNCTIONS = {
    # Striking Performance - Landed (8 trees)
    'knockdowns': create_knockdowns_tree_features,
    'sig_strikes_landed': create_sig_strikes_landed_tree_features,
    'sig_strikes_attempted': create_sig_strikes_attempted_tree_features,
    'total_strikes_landed': create_total_strikes_landed_tree_features,
    'total_strikes_attempted': create_total_strikes_attempted_tree_features,
    'head_strikes_landed': create_head_strikes_landed_tree_features,
    'body_strikes_landed': create_body_strikes_landed_tree_features,
    'leg_strikes_landed': create_leg_strikes_landed_tree_features,

    # Striking Performance - Accuracy (8 trees)
    'sig_strikes_accuracy': create_sig_strikes_accuracy_tree_features,
    'total_strikes_accuracy': create_total_strikes_accuracy_tree_features,
    'head_strikes_accuracy': create_head_strikes_accuracy_tree_features,
    'body_strikes_accuracy': create_body_strikes_accuracy_tree_features,
    'leg_strikes_accuracy': create_leg_strikes_accuracy_tree_features,
    'head_strikes_attempted': create_head_strikes_attempted_tree_features,
    'body_strikes_attempted': create_body_strikes_attempted_tree_features,
    'leg_strikes_attempted': create_leg_strikes_attempted_tree_features,

    # Grappling Performance - Landed (3 trees)
    'takedowns_landed': create_takedowns_landed_tree_features,
    'takedowns_attempted': create_takedowns_attempted_tree_features,
    'submission_attempts': create_submission_attempts_tree_features,

    # Grappling Performance - Accuracy/Control (2 trees)
    'takedown_accuracy': create_takedown_accuracy_tree_features,
    'control_time': create_control_time_tree_features,

    # Positional Fighting - Landed (6 trees)
    'distance_strikes_landed': create_distance_strikes_landed_tree_features,
    'distance_strikes_attempted': create_distance_strikes_attempted_tree_features,
    'clinch_strikes_landed': create_clinch_strikes_landed_tree_features,
    'clinch_strikes_attempted': create_clinch_strikes_attempted_tree_features,
    'ground_strikes_landed': create_ground_strikes_landed_tree_features,
    'ground_strikes_attempted': create_ground_strikes_attempted_tree_features,

    # Positional Fighting - Accuracy (3 trees)
    'distance_strikes_accuracy': create_distance_strikes_accuracy_tree_features,
    'clinch_strikes_accuracy': create_clinch_strikes_accuracy_tree_features,
    'ground_strikes_accuracy': create_ground_strikes_accuracy_tree_features,

    # Fight Context (2 trees)
    'fight_format': create_fight_format_tree_features,
    'fight_timing': create_fight_timing_tree_features,

    # Defensive Performance (4 trees)
    'striking_defense': create_striking_defense_tree_features,
    'takedown_defense': create_takedown_defense_tree_features,
    'strikes_absorbed_per_min': create_strikes_absorbed_per_min_tree_features,
    'strikes_landed_per_min': create_strikes_landed_per_min_tree_features,

    # Physical Attributes (3 trees)
    'height_advantage': create_height_advantage_tree_features,
    'reach_advantage': create_reach_advantage_tree_features,
    'weight_comparison': create_weight_comparison_tree_features,

    # Experience & Career (4 trees)
    'wins_comparison': create_wins_comparison_tree_features,
    'losses_comparison': create_losses_comparison_tree_features,
    'win_rate': create_win_rate_tree_features,
    'experience_gap': create_experience_gap_tree_features,

    # Advanced Grappling Metrics (3 trees)
    'takedown_average': create_takedown_average_tree_features,
    'submission_average': create_submission_average_tree_features,
    'overall_striking_accuracy': create_overall_striking_accuracy_tree_features,

    # Strike Distribution (6 trees)
    'head_strike_distribution': create_head_strike_distribution_tree_features,
    'body_strike_distribution': create_body_strike_distribution_tree_features,
    'leg_strike_distribution': create_leg_strike_distribution_tree_features,
    'distance_position_distribution': create_distance_position_distribution_tree_features,
    'clinch_position_distribution': create_clinch_position_distribution_tree_features,
    'ground_position_distribution': create_ground_position_distribution_tree_features,

    # Logical Category Trees - Phase 2 (11 trees)
    'striking_volume_category': create_striking_volume_category_tree_features,
    'striking_accuracy_category': create_striking_accuracy_category_tree_features,
    'grappling_offense_category': create_grappling_offense_category_tree_features,
    'physical_advantage_category': create_physical_advantage_category_tree_features,
    'experience_category': create_experience_category_tree_features,
    'grappling_defense_category': create_grappling_defense_category_tree_features,
    'power_finishing_category': create_power_finishing_category_tree_features,
    'strike_location_category': create_strike_location_category_tree_features,
    'strike_position_category': create_strike_position_category_tree_features,
    'fight_pace_category': create_fight_pace_category_tree_features,
    'comprehensive_distribution_category': create_comprehensive_distribution_category_tree_features,

    # Cross-Category Combination Trees - Phase 3 (24 trees)
    'volume_vs_accuracy_cross': create_volume_vs_accuracy_cross_tree_features,
    'power_vs_grappling_cross': create_power_vs_grappling_cross_tree_features,
    'experience_vs_physical_cross': create_experience_vs_physical_cross_tree_features,
    'defense_vs_offense_cross': create_defense_vs_offense_cross_tree_features,
}


def temporal_train_test_split(df, features, labels, test_size=0.2, date_column='date'):
    """Split data temporally instead of randomly to prevent future data leakage."""
    try:
        if date_column in df.columns:
            # Convert date to datetime if it's not already
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

            # Sort by date
            sorted_indices = df[date_column].argsort()

            # Calculate split point
            split_point = int(len(sorted_indices) * (1 - test_size))

            # Split indices
            train_indices = sorted_indices[:split_point]
            test_indices = sorted_indices[split_point:]

            # Create train/test sets
            X_train = np.array([features[i] for i in train_indices])
            X_test = np.array([features[i] for i in test_indices])
            y_train = np.array([labels[i] for i in train_indices])
            y_test = np.array([labels[i] for i in test_indices])

            print(f"📅 Using temporal split: Train ({len(X_train)}) | Test ({len(X_test)})")
            return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"⚠️  Temporal split failed ({e}), falling back to random split")

    # Fallback to random split
    return None

def bootstrap_sample(X, y, random_seed):
    """Create a bootstrap sample of the training data."""
    np.random.seed(random_seed)
    n_samples = len(X)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

def create_individual_decision_tree(df, tree_name, max_depth=6, min_samples_split=20,
                                   min_samples_leaf=10, save_model=True, random_seed=None,
                                   use_temporal_split=False, use_bootstrap=True):
    """
    Create and train an individual decision tree for a specific UFC statistic.

    Args:
        df (pandas.DataFrame): UFC fight dataset
        tree_name (str): Name of the tree to create (must be in TREE_FEATURE_FUNCTIONS)
        max_depth (int): Maximum depth of the decision tree
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required in a leaf node
        save_model (bool): Whether to save the trained model

    Returns:
        tuple: (trained_model, feature_names, accuracy, test_predictions)
    """
    if tree_name not in TREE_FEATURE_FUNCTIONS:
        raise ValueError(f"Unknown tree name: {tree_name}. Available: {list(TREE_FEATURE_FUNCTIONS.keys())}")

    # Get base fight data and labels
    fight_data, labels = create_base_features(df)

    if len(fight_data) == 0:
        print(f"No valid data found for {tree_name} tree")
        return None, None, 0, None

    # Create features for this specific tree
    feature_function = TREE_FEATURE_FUNCTIONS[tree_name]
    X, feature_names = feature_function(fight_data)
    y = np.array(labels)

    # Use diverse random seeds for each tree to increase randomness
    if random_seed is None:
        # Generate seed based on tree name for reproducibility while maintaining diversity
        random_seed = hash(tree_name) % 10000

    # Split data - use temporal split if requested and available
    if use_temporal_split:
        # For temporal split, we need to split on the original dataframe level
        try:
            df_sorted = df.sort_values('date', na_position='last')
            split_point = int(len(df_sorted) * 0.8)

            df_train = df_sorted.iloc[:split_point]
            df_test = df_sorted.iloc[split_point:]

            # Recreate features and labels for train/test
            train_fight_data, train_labels = create_base_features(df_train)
            test_fight_data, test_labels = create_base_features(df_test)

            X_train, _ = feature_function(train_fight_data)
            X_test, _ = feature_function(test_fight_data)
            y_train = np.array(train_labels)
            y_test = np.array(test_labels)

            print(f"📅 Temporal split: Train ({len(X_train)}) | Test ({len(X_test)})")
        except Exception as e:
            print(f"⚠️  Temporal split failed ({e}), using random split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    else:
        # Standard random split with tree-specific random seed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Apply bootstrap sampling for additional diversity
    if use_bootstrap:
        X_train_bootstrap, y_train_bootstrap = bootstrap_sample(X_train, y_train, random_seed)
        print(f"🎲 Using bootstrap sampling for {tree_name}")
    else:
        X_train_bootstrap, y_train_bootstrap = X_train, y_train

    # Create and train decision tree with tree-specific random seed
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
        criterion='gini'
    )

    dt.fit(X_train_bootstrap, y_train_bootstrap)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Add cross-validation for better generalization assessment
    cv_scores = cross_val_score(dt, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed), scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Check for potential overfitting (large gap between train and test performance)
    train_accuracy = dt.score(X_train, y_train)
    overfitting_gap = train_accuracy - accuracy

    print(f"\n{tree_name.replace('_', ' ').title()} Tree Results:")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"CV Accuracy: {cv_mean:.3f} (±{cv_std:.3f})")
    print(f"Train Accuracy: {train_accuracy:.3f}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    if overfitting_gap > 0.1:
        print(f"⚠️  Potential overfitting: {overfitting_gap:.3f} gap")

    # Feature importance
    importance = dt.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print(f"\nTop Features:")
    print(feature_importance_df.head(3).to_string(index=False))

    if save_model:
        filename = f'models/ufc_{tree_name}_tree.pkl'
        joblib.dump(dt, filename)
        print(f"Tree saved as '{filename}'")

    return dt, feature_names, accuracy, {'y_test': y_test, 'y_pred': y_pred, 'cv_scores': cv_scores, 'train_accuracy': train_accuracy}


def train_all_individual_trees(df, save_models=True):
    """
    Train all 32 individual decision trees.

    Args:
        df (pandas.DataFrame): UFC fight dataset
        save_models (bool): Whether to save trained models

    Returns:
        dict: Training results for each tree
    """
    results = {}

    total_trees = len(TREE_FEATURE_FUNCTIONS)
    print(f"Training All {total_trees} Individual Decision Trees")
    print("=" * 80)

    for i, tree_name in enumerate(TREE_FEATURE_FUNCTIONS.keys(), 1):
        print(f"\n{i}/{total_trees}. Training {tree_name.replace('_', ' ').title()} Tree...")

        try:
            tree, features, accuracy, test_results = create_individual_decision_tree(
                df, tree_name, save_model=save_models
            )
            results[tree_name] = {
                'tree': tree,
                'features': features,
                'accuracy': accuracy,
                'test_results': test_results
            }
        except Exception as e:
            print(f"Failed to train {tree_name} tree: {e}")
            results[tree_name] = {
                'tree': None,
                'features': None,
                'accuracy': 0,
                'test_results': None
            }

    print(f"\n" + "=" * 80)
    print("INDIVIDUAL TREE TRAINING COMPLETE")
    print("=" * 80)

    # Summary
    successful_trees = sum(1 for r in results.values() if r['tree'] is not None)
    avg_accuracy = np.mean([r['accuracy'] for r in results.values() if r['accuracy'] > 0])

    print(f"Successfully trained: {successful_trees}/32 trees")
    print(f"Average accuracy: {avg_accuracy:.3f}")

    # Top performing trees
    sorted_results = sorted(
        [(name, r['accuracy']) for name, r in results.items() if r['accuracy'] > 0],
        key=lambda x: x[1], reverse=True
    )

    print(f"\nTop 5 Performing Trees:")
    for i, (name, acc) in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {name.replace('_', ' ').title()}: {acc:.3f}")

    return results


# ============================================================================
# CUSTOM RANDOM FOREST ENSEMBLE
# ============================================================================

class UFC_Individual_Tree_Forest:
    """
    Custom Random Forest using 32 individual UFC decision trees as components.

    Unlike sklearn's RandomForest which uses random feature subsets,
    this ensemble uses our 32 specialized trees, each focusing on specific
    fighter performance metrics.
    """

    def __init__(self, tree_weights=None):
        """
        Initialize the individual tree forest.

        Args:
            tree_weights (dict): Custom weights for each tree type
                               If None, uses equal weights for all trees
        """
        self.trees = {}
        self.feature_functions = {}
        self.tree_weights = tree_weights
        self.tree_accuracies = {}

    def train_forest(self, df, save_models=True):
        """
        Train all individual trees to form the forest.

        Args:
            df (pandas.DataFrame): UFC fight dataset
            save_models (bool): Whether to save trained models

        Returns:
            dict: Training results for each tree
        """
        print("Training Individual Tree Forest (32 trees)")
        print("=" * 80)

        results = train_all_individual_trees(df, save_models)

        # Store successful trees and their accuracies
        for tree_name, result in results.items():
            if result['tree'] is not None:
                self.trees[tree_name] = result['tree']
                self.feature_functions[tree_name] = TREE_FEATURE_FUNCTIONS[tree_name]
                self.tree_accuracies[tree_name] = result['accuracy']

        # Set weights based on accuracy if not provided
        if self.tree_weights is None:
            total_accuracy = sum(self.tree_accuracies.values())
            self.tree_weights = {
                name: acc / total_accuracy
                for name, acc in self.tree_accuracies.items()
            }

        # Calculate forest accuracy
        forest_accuracy = self._calculate_forest_accuracy(results)

        print(f"\n" + "=" * 80)
        print("INDIVIDUAL TREE FOREST TRAINING COMPLETE")
        print("=" * 80)
        print(f"Forest trees: {len(self.trees)}/32")
        print(f"Forest accuracy: {forest_accuracy:.3f}")

        if save_models:
            self.save_forest('models/ufc_individual_tree_forest.pkl')

        return results

    def _calculate_forest_accuracy(self, results):
        """Calculate weighted forest accuracy from individual tree results."""
        if not any(r['test_results'] for r in results.values() if r['test_results']):
            return 0.0

        # Get test data from first successful tree
        y_test = None
        for result in results.values():
            if result['test_results']:
                y_test = result['test_results']['y_test']
                break

        if y_test is None:
            return 0.0

        # Make predictions with each tree and combine
        forest_predictions = []
        for i in range(len(y_test)):
            weighted_vote = 0
            total_weight = 0

            for tree_name, result in results.items():
                if result['tree'] is not None and result['test_results']:
                    tree_pred = result['test_results']['y_pred'][i]
                    weight = self.tree_weights.get(tree_name, 1/len(self.trees))
                    weighted_vote += tree_pred * weight
                    total_weight += weight

            final_pred = 1 if weighted_vote / total_weight > 0.5 else 0
            forest_predictions.append(final_pred)

        return accuracy_score(y_test, forest_predictions)

    def predict_fight(self, fight_data):
        """
        Predict fight outcome using all individual trees in the forest.

        Args:
            fight_data (dict): Fight statistics for both fighters

        Returns:
            dict: Prediction results with individual tree contributions
        """
        if not self.trees:
            raise ValueError("Forest not trained. Call train_forest() first.")

        # Convert single fight to list format
        fight_data_list = [fight_data]

        predictions = {}
        probabilities = {}

        # Get predictions from each tree
        for tree_name, tree in self.trees.items():
            feature_function = self.feature_functions[tree_name]
            X, _ = feature_function(fight_data_list)

            if len(X) > 0:
                pred = tree.predict(X)[0]
                prob = tree.predict_proba(X)[0]

                predictions[tree_name] = pred
                probabilities[tree_name] = {
                    'fighter_a_prob': prob[1],
                    'fighter_b_prob': prob[0]
                }

        # Calculate weighted forest prediction
        weighted_vote = sum(
            predictions[tree_name] * self.tree_weights.get(tree_name, 1/len(self.trees))
            for tree_name in predictions.keys()
        )

        forest_prediction = 1 if weighted_vote > 0.5 else 0

        result = {
            'forest_prediction': 'Fighter A (Red Corner)' if forest_prediction == 1 else 'Fighter B (Blue Corner)',
            'forest_confidence': abs(weighted_vote - 0.5) * 2,
            'weighted_vote': weighted_vote,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'tree_weights': self.tree_weights,
            'trees_used': len(predictions)
        }

        return result

    def save_forest(self, filename):
        """Save the trained forest to a file."""
        forest_data = {
            'trees': self.trees,
            'feature_functions': self.feature_functions,
            'tree_weights': self.tree_weights,
            'tree_accuracies': self.tree_accuracies
        }

        with open(filename, 'wb') as f:
            pickle.dump(forest_data, f)
        print(f"\nIndividual tree forest saved as '{filename}'")

    @classmethod
    def load_forest(cls, filename):
        """Load a previously saved forest."""
        with open(filename, 'rb') as f:
            forest_data = pickle.load(f)

        forest = cls(tree_weights=forest_data['tree_weights'])
        forest.trees = forest_data['trees']
        forest.feature_functions = forest_data['feature_functions']
        forest.tree_accuracies = forest_data['tree_accuracies']

        print(f"Individual tree forest loaded from '{filename}'")
        return forest

    def get_tree_rankings(self):
        """Get trees ranked by accuracy."""
        return sorted(
            self.tree_accuracies.items(),
            key=lambda x: x[1],
            reverse=True
        )