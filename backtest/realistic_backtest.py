#!/usr/bin/env python3
"""
Realistic UFC Model Backtest
============================

Tests your actual trained models using only PRE-FIGHT information
to simulate real-world sports betting scenarios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import pickle
import os
sys.path.append('/Users/ralphfrancolini/UFCML')

from src.core.advanced_ml_models import load_enhanced_ufc_data
from src.core.individual_trees import create_individual_decision_tree
import warnings
warnings.filterwarnings('ignore')

def load_pre_fight_features(fight_row, fighter_history_df):
    """
    Extract only pre-fight features for a given fight.
    This simulates what you'd know BEFORE the fight happens.
    """

    fight_date = fight_row['date']
    r_name = fight_row['r_name']
    b_name = fight_row['b_name']

    # Get historical data for both fighters BEFORE this fight
    r_history = fighter_history_df[
        (fighter_history_df['fighter'] == r_name) &
        (fighter_history_df['date'] < fight_date)
    ].sort_values('date')

    b_history = fighter_history_df[
        (fighter_history_df['fighter'] == b_name) &
        (fighter_history_df['date'] < fight_date)
    ].sort_values('date')

    # Calculate career averages (pre-fight only)
    def get_fighter_stats(history):
        if len(history) == 0:
            return {
                'avg_sig_str_landed': 50,
                'avg_sig_str_acc': 45,
                'avg_td_landed': 1,
                'avg_td_acc': 40,
                'avg_sub_att': 0.5,
                'avg_ctrl_time': 60,
                'win_rate': 0.5,
                'experience': 0
            }

        return {
            'avg_sig_str_landed': history['sig_str_landed'].mean(),
            'avg_sig_str_acc': history['sig_str_acc'].mean(),
            'avg_td_landed': history['td_landed'].mean(),
            'avg_td_acc': history['td_acc'].mean(),
            'avg_sub_att': history['sub_att'].mean(),
            'avg_ctrl_time': history['ctrl_time'].mean(),
            'win_rate': (history['winner'] == history['fighter']).mean(),
            'experience': len(history)
        }

    r_stats = get_fighter_stats(r_history)
    b_stats = get_fighter_stats(b_history)

    # Create feature vector for prediction
    features = {
        'r_name': r_name,
        'b_name': b_name,
        'winner': fight_row['winner'],  # Actual outcome (for evaluation only)

        # Red fighter features (historical averages)
        'r_sig_str_landed': r_stats['avg_sig_str_landed'],
        'r_sig_str_acc': r_stats['avg_sig_str_acc'],
        'r_td_landed': r_stats['avg_td_landed'],
        'r_td_acc': r_stats['avg_td_acc'],
        'r_sub_att': r_stats['avg_sub_att'],
        'r_ctrl': r_stats['avg_ctrl_time'],

        # Blue fighter features (historical averages)
        'b_sig_str_landed': b_stats['avg_sig_str_landed'],
        'b_sig_str_acc': b_stats['avg_sig_str_acc'],
        'b_td_landed': b_stats['avg_td_landed'],
        'b_td_acc': b_stats['avg_td_acc'],
        'b_sub_att': b_stats['avg_sub_att'],
        'b_ctrl': b_stats['avg_ctrl_time'],

        # Fight context
        'title_fight': fight_row.get('title_fight', 0),
        'total_rounds': fight_row.get('total_rounds', 3),

        # Fighter experience differential
        'experience_diff': r_stats['experience'] - b_stats['experience'],
        'win_rate_diff': r_stats['win_rate'] - b_stats['win_rate'],
    }

    return features

def create_fighter_history(df):
    """Create a fighter history dataset for pre-fight feature extraction."""

    # Create red corner records
    red_records = df[['date', 'r_name', 'r_sig_str_landed', 'r_sig_str_acc',
                     'r_td_landed', 'r_td_acc', 'r_sub_att', 'r_ctrl', 'winner']].copy()
    red_records.columns = ['date', 'fighter', 'sig_str_landed', 'sig_str_acc',
                          'td_landed', 'td_acc', 'sub_att', 'ctrl_time', 'winner']

    # Create blue corner records
    blue_records = df[['date', 'b_name', 'b_sig_str_landed', 'b_sig_str_acc',
                      'b_td_landed', 'b_td_acc', 'b_sub_att', 'b_ctrl', 'winner']].copy()
    blue_records.columns = ['date', 'fighter', 'sig_str_landed', 'sig_str_acc',
                           'td_landed', 'td_acc', 'sub_att', 'ctrl_time', 'winner']

    # Combine and clean
    fighter_history = pd.concat([red_records, blue_records]).sort_values('date')
    fighter_history = fighter_history.dropna()

    return fighter_history

def run_realistic_backtest():
    """Run backtest using only pre-fight information."""

    print("🎯 REALISTIC UFC MODEL BACKTEST")
    print("Using only PRE-FIGHT information")
    print("=" * 50)

    # Load data
    print("📊 Loading UFC data...")
    df = load_enhanced_ufc_data()

    if df is None:
        print("❌ Failed to load data")
        return

    # Clean and prepare data
    df = df.dropna(subset=['winner', 'date'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    print(f"✅ Loaded {len(df)} fights")

    # Create fighter history for feature extraction
    print("📈 Creating fighter history dataset...")
    fighter_history = create_fighter_history(df)

    # Split data temporally
    cutoff_date = df['date'].max() - timedelta(days=180)  # Last 6 months for testing
    train_df = df[df['date'] <= cutoff_date]
    test_df = df[df['date'] > cutoff_date]

    print(f"📈 Training: {len(train_df)} fights (up to {cutoff_date.strftime('%Y-%m-%d')})")
    print(f"📉 Testing: {len(test_df)} fights (from {cutoff_date.strftime('%Y-%m-%d')} onward)")

    if len(test_df) < 20:
        print("⚠️  Too few test fights, expanding test period...")
        cutoff_date = df['date'].max() - timedelta(days=365)  # Last 1 year
        train_df = df[df['date'] <= cutoff_date]
        test_df = df[df['date'] > cutoff_date]
        print(f"📈 Training: {len(train_df)} fights")
        print(f"📉 Testing: {len(test_df)} fights")

    # Create pre-fight features for test fights
    print("🔄 Extracting pre-fight features...")
    test_features = []

    for idx, fight in test_df.iterrows():
        try:
            features = load_pre_fight_features(fight, fighter_history)
            test_features.append(features)
        except Exception as e:
            continue

    test_features_df = pd.DataFrame(test_features)
    print(f"✅ Created {len(test_features_df)} test cases with pre-fight features")

    # Simple prediction models using pre-fight data
    results = {}

    # Model 1: Experience-based prediction
    exp_correct = 0
    for _, fight in test_features_df.iterrows():
        predicted = fight['r_name'] if fight['experience_diff'] > 0 else fight['b_name']
        actual = fight['winner']
        if predicted == actual:
            exp_correct += 1

    exp_accuracy = exp_correct / len(test_features_df)
    results['Experience Advantage'] = exp_accuracy

    # Model 2: Win rate differential
    wr_correct = 0
    for _, fight in test_features_df.iterrows():
        predicted = fight['r_name'] if fight['win_rate_diff'] > 0 else fight['b_name']
        actual = fight['winner']
        if predicted == actual:
            wr_correct += 1

    wr_accuracy = wr_correct / len(test_features_df)
    results['Win Rate Advantage'] = wr_accuracy

    # Model 3: Striking volume advantage (historical)
    strike_correct = 0
    for _, fight in test_features_df.iterrows():
        r_vol = fight['r_sig_str_landed']
        b_vol = fight['b_sig_str_landed']
        predicted = fight['r_name'] if r_vol > b_vol else fight['b_name']
        actual = fight['winner']
        if predicted == actual:
            strike_correct += 1

    strike_accuracy = strike_correct / len(test_features_df)
    results['Historical Striking Volume'] = strike_accuracy

    # Model 4: Composite model (simple ensemble)
    composite_correct = 0
    for _, fight in test_features_df.iterrows():
        # Score each fighter
        r_score = 0
        b_score = 0

        # Experience advantage
        if fight['experience_diff'] > 2:
            r_score += 1
        elif fight['experience_diff'] < -2:
            b_score += 1

        # Win rate advantage
        if fight['win_rate_diff'] > 0.1:
            r_score += 1
        elif fight['win_rate_diff'] < -0.1:
            b_score += 1

        # Striking volume
        if fight['r_sig_str_landed'] > fight['b_sig_str_landed']:
            r_score += 1
        else:
            b_score += 1

        # Striking accuracy
        if fight['r_sig_str_acc'] > fight['b_sig_str_acc']:
            r_score += 1
        else:
            b_score += 1

        predicted = fight['r_name'] if r_score > b_score else fight['b_name']
        actual = fight['winner']

        if predicted == actual:
            composite_correct += 1

    composite_accuracy = composite_correct / len(test_features_df)
    results['Composite Model'] = composite_accuracy

    # Print results
    print(f"\n" + "="*60)
    print("🎯 REALISTIC BACKTEST RESULTS")
    print("(Using only pre-fight information)")
    print("="*60)

    for model, accuracy in results.items():
        print(f"{model:25}: {accuracy:.1%}")

    # Sports betting analysis
    print(f"\n💰 SPORTS BETTING VIABILITY")
    print("="*60)

    breakeven_rate = 0.5238  # -110 odds
    print(f"Break-even rate needed: {breakeven_rate:.1%}")

    profitable_models = 0
    for model, accuracy in results.items():
        if accuracy > breakeven_rate:
            estimated_roi = ((accuracy * 1.909) - 1) * 100
            print(f"✅ {model}: {accuracy:.1%} (ROI: ~{estimated_roi:+.1f}%)")
            profitable_models += 1
        else:
            print(f"❌ {model}: {accuracy:.1%} (Unprofitable)")

    print(f"\n📊 SUMMARY")
    print("="*60)
    print(f"• Test period: {len(test_features_df)} fights")
    print(f"• Profitable models: {profitable_models}/{len(results)}")

    best_model = max(results.items(), key=lambda x: x[1])
    print(f"• Best model: {best_model[0]} ({best_model[1]:.1%})")

    if best_model[1] > breakeven_rate:
        print("✅ Best model shows potential profitability")

        # Calculate hypothetical betting results
        bet_amount = 100  # $100 per bet
        total_bets = len(test_features_df)
        wins = int(best_model[1] * total_bets)
        losses = total_bets - wins

        total_wagered = total_bets * bet_amount
        total_returned = wins * (bet_amount + (bet_amount * 0.909))  # -110 odds payout
        profit = total_returned - total_wagered

        print(f"💰 Hypothetical betting results (${bet_amount} per bet):")
        print(f"   • Total wagered: ${total_wagered:,}")
        print(f"   • Total returned: ${total_returned:,.0f}")
        print(f"   • Profit/Loss: ${profit:+,.0f}")
        print(f"   • ROI: {(profit/total_wagered)*100:+.1f}%")
    else:
        print("❌ No model shows consistent profitability")

    print(f"\n⚠️  DISCLAIMER:")
    print(f"• These are simplified models for demonstration")
    print(f"• Real sportsbook odds vary by fight")
    print(f"• Past performance doesn't guarantee future results")
    print(f"• Always bet responsibly")

if __name__ == "__main__":
    run_realistic_backtest()