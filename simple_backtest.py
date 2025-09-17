#!/usr/bin/env python3
"""
Simple UFC Model Backtest
=========================

A focused backtesting script that evaluates model performance
on recent UFC fights using proper temporal splitting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('/Users/ralphfrancolini/UFCML')

from src.core.advanced_ml_models import load_enhanced_ufc_data
import warnings
warnings.filterwarnings('ignore')

def run_simple_backtest():
    """Run a simple backtest with clear methodology."""

    print("🥊 UFC MODEL SIMPLE BACKTEST")
    print("=" * 50)

    # Load data
    print("📊 Loading UFC data...")
    df = load_enhanced_ufc_data()

    if df is None:
        print("❌ Failed to load data")
        return

    # Clean data
    df = df.dropna(subset=['winner', 'date'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    print(f"✅ Loaded {len(df)} fights from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # Split data temporally (last 1 year for testing)
    cutoff_date = df['date'].max() - timedelta(days=365)
    train_df = df[df['date'] <= cutoff_date]
    test_df = df[df['date'] > cutoff_date]

    print(f"📈 Training: {len(train_df)} fights (up to {cutoff_date.strftime('%Y-%m-%d')})")
    print(f"📉 Testing: {len(test_df)} fights (from {cutoff_date.strftime('%Y-%m-%d')} onward)")

    if len(test_df) < 10:
        print("⚠️  Too few test fights for meaningful results")
        return

    # Simple baseline strategies
    results = {}

    # Strategy 1: Random (50% baseline)
    random_accuracy = 0.5
    results['Random Baseline'] = random_accuracy

    # Strategy 2: Always pick the favorite (red corner bias)
    # Assume red corner is slight favorite historically
    red_wins = (test_df['winner'] == test_df['r_name']).sum()
    red_accuracy = red_wins / len(test_df)
    results['Always Pick Red Corner'] = red_accuracy

    # Strategy 3: Pick based on higher Elo (if available)
    if 'r_elo' in test_df.columns and 'b_elo' in test_df.columns:
        elo_correct = 0
        elo_total = 0

        for _, fight in test_df.iterrows():
            r_elo = fight.get('r_elo', 1500)
            b_elo = fight.get('b_elo', 1500)

            if pd.notna(r_elo) and pd.notna(b_elo):
                predicted = fight['r_name'] if r_elo > b_elo else fight['b_name']
                actual = fight['winner']

                if predicted == actual:
                    elo_correct += 1
                elo_total += 1

        if elo_total > 0:
            elo_accuracy = elo_correct / elo_total
            results['Elo-Based Picks'] = elo_accuracy
            print(f"📊 Elo-based predictions: {elo_correct}/{elo_total} = {elo_accuracy:.1%}")

    # Strategy 4: Pick based on significant strikes differential
    if all(col in test_df.columns for col in ['r_sig_str_landed', 'b_sig_str_landed']):
        strikes_correct = 0
        strikes_total = 0

        for _, fight in test_df.iterrows():
            r_strikes = fight.get('r_sig_str_landed', 0)
            b_strikes = fight.get('b_sig_str_landed', 0)

            if pd.notna(r_strikes) and pd.notna(b_strikes):
                predicted = fight['r_name'] if r_strikes > b_strikes else fight['b_name']
                actual = fight['winner']

                if predicted == actual:
                    strikes_correct += 1
                strikes_total += 1

        if strikes_total > 0:
            strikes_accuracy = strikes_correct / strikes_total
            results['Strikes-Based (Post-hoc)'] = strikes_accuracy

    # Strategy 5: Pick based on takedowns
    if all(col in test_df.columns for col in ['r_td_landed', 'b_td_landed']):
        td_correct = 0
        td_total = 0

        for _, fight in test_df.iterrows():
            r_td = fight.get('r_td_landed', 0)
            b_td = fight.get('b_td_landed', 0)

            if pd.notna(r_td) and pd.notna(b_td):
                # If equal takedowns, fall back to strikes
                if r_td == b_td:
                    r_strikes = fight.get('r_sig_str_landed', 0)
                    b_strikes = fight.get('b_sig_str_landed', 0)
                    predicted = fight['r_name'] if r_strikes > b_strikes else fight['b_name']
                else:
                    predicted = fight['r_name'] if r_td > b_td else fight['b_name']

                actual = fight['winner']

                if predicted == actual:
                    td_correct += 1
                td_total += 1

        if td_total > 0:
            td_accuracy = td_correct / td_total
            results['Takedown-Based (Post-hoc)'] = td_accuracy

    # Print results
    print(f"\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)

    for strategy, accuracy in results.items():
        print(f"{strategy:25}: {accuracy:.1%}")

    # Sports betting analysis
    print(f"\n💰 SPORTS BETTING ANALYSIS")
    print("="*60)

    # Assuming typical sportsbook odds of -110 (need 52.38% to break even)
    breakeven_rate = 0.5238

    print(f"Break-even rate (typical -110 odds): {breakeven_rate:.1%}")
    print(f"\nProfitable strategies:")

    profitable_count = 0
    for strategy, accuracy in results.items():
        if accuracy > breakeven_rate:
            roi_estimate = ((accuracy * 1.909) - 1) * 100  # Rough ROI calculation
            print(f"✅ {strategy}: {accuracy:.1%} (Est. ROI: {roi_estimate:+.1f}%)")
            profitable_count += 1
        else:
            print(f"❌ {strategy}: {accuracy:.1%} (Below break-even)")

    print(f"\n📈 Summary: {profitable_count}/{len(results)} strategies profitable")

    # Additional insights
    print(f"\n💡 KEY INSIGHTS")
    print("="*60)
    print(f"• Test period: {len(test_df)} fights over 1 year")
    print(f"• Red corner wins: {red_accuracy:.1%} (shows slight favorite bias)")

    best_strategy = max(results.items(), key=lambda x: x[1])
    print(f"• Best strategy: {best_strategy[0]} ({best_strategy[1]:.1%})")

    if best_strategy[1] > breakeven_rate:
        print(f"✅ Best strategy beats sportsbook break-even rate")
    else:
        print(f"❌ No strategy beats sportsbook break-even rate consistently")

    print(f"\n⚠️  IMPORTANT NOTES:")
    print(f"• Post-hoc strategies use fight outcome data (not realistic)")
    print(f"• Real predictions need pre-fight data only")
    print(f"• This backtest establishes performance baselines")
    print(f"• Your trained models should be tested with pre-fight features only")

if __name__ == "__main__":
    run_simple_backtest()