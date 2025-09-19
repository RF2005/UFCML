#!/usr/bin/env python3
"""
UFC Model Backtesting System
============================

Comprehensive backtesting framework for UFC prediction models.
Tests models on historical data to evaluate real-world performance.

Features:
- Historical data splitting (train/test by date)
- Multiple model evaluation
- Sports betting simulation
- Performance metrics (accuracy, precision, recall, ROI)
- Detailed analysis reports

Usage:
    python backtest_ufc_models.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('/Users/ralphfrancolini/UFCML')

from src.core.individual_trees import UFC_Individual_Tree_Forest, load_enhanced_ufc_data
from src.core.advanced_ml_models import UFC_SpecializedEnsemble
from src.core.fighter_matchup_predictor import FighterMatchupPredictor
import warnings
warnings.filterwarnings('ignore')

class UFCModelBacktester:
    """Comprehensive backtesting system for UFC prediction models."""

    def __init__(self):
        self.df = None
        self.train_data = None
        self.test_data = None
        self.results = {}
        self.betting_results = {}

    def load_data(self):
        """Load and prepare UFC data for backtesting."""
        print("📊 Loading UFC dataset for backtesting...")
        self.df = load_enhanced_ufc_data()

        if self.df is None:
            print("❌ Failed to load UFC dataset")
            return False

        print(f"✅ Loaded {len(self.df)} fights")

        # Clean and prepare data
        self.df = self.df.dropna(subset=['winner'])
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df = self.df.dropna(subset=['date'])
        self.df = self.df.sort_values('date')

        print(f"✅ Prepared {len(self.df)} fights with valid dates")
        date_range = f"{self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}"
        print(f"📅 Date range: {date_range}")

        return True

    def split_data_by_date(self, test_years=2):
        """Split data into training and testing periods by date."""
        print(f"\n🔄 Splitting data for backtesting...")

        # Get the latest date and work backwards
        max_date = self.df['date'].max()
        split_date = max_date - timedelta(days=365 * test_years)

        # Split data
        self.train_data = self.df[self.df['date'] <= split_date].copy()
        self.test_data = self.df[self.df['date'] > split_date].copy()

        print(f"📈 Training data: {len(self.train_data)} fights (up to {split_date.strftime('%Y-%m-%d')})")
        print(f"📉 Test data: {len(self.test_data)} fights (from {split_date.strftime('%Y-%m-%d')} onwards)")

        if len(self.test_data) < 50:
            print("⚠️  Warning: Very few test fights available")

        return len(self.test_data) > 0

    def backtest_fighter_matchup_predictor(self):
        """Backtest the fighter matchup predictor."""
        print(f"\n🥊 Backtesting Fighter Matchup Predictor...")

        try:
            predictor = FighterMatchupPredictor()
            predictions = []
            confidences = []
            actual_results = []

            test_sample = self.test_data.sample(min(100, len(self.test_data)), random_state=42)

            for idx, fight in test_sample.iterrows():
                try:
                    fighter_a = fight.get('r_name', '').strip()
                    fighter_b = fight.get('b_name', '').strip()
                    actual_winner = fight.get('winner', '').strip()

                    if not fighter_a or not fighter_b or not actual_winner:
                        continue

                    # Make prediction
                    result = predictor.predict_matchup(fighter_a, fighter_b)

                    if result and result.get('final_prediction'):
                        predicted_winner = result['final_prediction']
                        confidence = result.get('final_confidence', 0.5)

                        predictions.append(predicted_winner)
                        confidences.append(confidence)
                        actual_results.append(actual_winner)

                except Exception as e:
                    continue

            if len(predictions) > 0:
                accuracy = sum(1 for p, a in zip(predictions, actual_results) if p == a) / len(predictions)
                avg_confidence = np.mean(confidences)

                self.results['fighter_matchup_predictor'] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'total_predictions': len(predictions),
                    'correct_predictions': sum(1 for p, a in zip(predictions, actual_results) if p == a)
                }

                print(f"✅ Fighter Matchup Predictor: {accuracy:.1%} accuracy on {len(predictions)} fights")
                print(f"📊 Average confidence: {avg_confidence:.1%}")
            else:
                print("❌ No successful predictions from Fighter Matchup Predictor")

        except Exception as e:
            print(f"❌ Error testing Fighter Matchup Predictor: {e}")

    def backtest_individual_trees(self):
        """Backtest individual decision trees."""
        print(f"\n🌳 Backtesting Individual Trees...")

        try:
            # Load pre-trained forest
            forest = UFC_Individual_Tree_Forest.load_forest('models/ufc_individual_tree_forest.pkl')

            predictions = []
            actual_results = []

            test_sample = self.test_data.sample(min(100, len(self.test_data)), random_state=42)

            for idx, fight in test_sample.iterrows():
                try:
                    # Make prediction
                    result = forest.predict_fight(fight.to_dict())

                    if result and result.get('forest_prediction'):
                        predicted_winner = result['forest_prediction']
                        actual_winner = fight.get('winner', '').strip()

                        predictions.append(predicted_winner)
                        actual_results.append(actual_winner)

                except Exception as e:
                    continue

            if len(predictions) > 0:
                accuracy = sum(1 for p, a in zip(predictions, actual_results) if p == a) / len(predictions)

                self.results['individual_trees'] = {
                    'accuracy': accuracy,
                    'total_predictions': len(predictions),
                    'correct_predictions': sum(1 for p, a in zip(predictions, actual_results) if p == a)
                }

                print(f"✅ Individual Trees: {accuracy:.1%} accuracy on {len(predictions)} fights")
            else:
                print("❌ No successful predictions from Individual Trees")

        except Exception as e:
            print(f"❌ Error testing Individual Trees: {e}")

    def simulate_betting_strategy(self):
        """Simulate various betting strategies."""
        print(f"\n💰 Simulating Betting Strategies...")

        # Get predictions with confidence scores
        betting_data = []

        for model_name, model_results in self.results.items():
            if 'predictions' in model_results:
                for pred, actual, conf in zip(model_results['predictions'],
                                            model_results['actual'],
                                            model_results.get('confidences', [])):
                    betting_data.append({
                        'model': model_name,
                        'predicted': pred,
                        'actual': actual,
                        'confidence': conf,
                        'correct': pred == actual
                    })

        if not betting_data:
            print("❌ No betting data available")
            return

        df_bets = pd.DataFrame(betting_data)

        # Strategy 1: Bet on all predictions
        strategy_1_roi = self.calculate_betting_roi(df_bets, min_confidence=0.0)

        # Strategy 2: Only bet on high confidence (>60%)
        strategy_2_roi = self.calculate_betting_roi(df_bets, min_confidence=0.6)

        # Strategy 3: Only bet on very high confidence (>70%)
        strategy_3_roi = self.calculate_betting_roi(df_bets, min_confidence=0.7)

        self.betting_results = {
            'bet_all': strategy_1_roi,
            'high_confidence': strategy_2_roi,
            'very_high_confidence': strategy_3_roi
        }

        print(f"📈 Betting Strategy Results:")
        print(f"  • Bet All: {strategy_1_roi['roi']:.1%} ROI ({strategy_1_roi['total_bets']} bets)")
        print(f"  • High Confidence (>60%): {strategy_2_roi['roi']:.1%} ROI ({strategy_2_roi['total_bets']} bets)")
        print(f"  • Very High Confidence (>70%): {strategy_3_roi['roi']:.1%} ROI ({strategy_3_roi['total_bets']} bets)")

    def calculate_betting_roi(self, df_bets, min_confidence=0.0):
        """Calculate ROI for a betting strategy."""
        # Filter by confidence threshold
        strategy_bets = df_bets[df_bets['confidence'] >= min_confidence]

        if len(strategy_bets) == 0:
            return {'roi': 0.0, 'total_bets': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}

        # Assume even money bets (1:1 payout) for simplicity
        # In reality, odds would vary by fight
        total_wagered = len(strategy_bets) * 100  # $100 per bet
        total_winnings = strategy_bets['correct'].sum() * 100  # $100 profit per win + $100 stake back
        total_returned = strategy_bets['correct'].sum() * 200  # $200 total return per win

        roi = (total_returned - total_wagered) / total_wagered if total_wagered > 0 else 0
        win_rate = strategy_bets['correct'].mean()

        return {
            'roi': roi,
            'total_bets': len(strategy_bets),
            'wins': strategy_bets['correct'].sum(),
            'losses': len(strategy_bets) - strategy_bets['correct'].sum(),
            'win_rate': win_rate,
            'total_wagered': total_wagered,
            'total_returned': total_returned
        }

    def generate_report(self):
        """Generate comprehensive backtesting report."""
        print(f"\n" + "="*80)
        print("📊 UFC MODEL BACKTESTING REPORT")
        print("="*80)

        # Data summary
        print(f"\n📈 DATA SUMMARY:")
        print(f"Training period: {self.train_data['date'].min().strftime('%Y-%m-%d')} to {self.train_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"Testing period: {self.test_data['date'].min().strftime('%Y-%m-%d')} to {self.test_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"Training fights: {len(self.train_data)}")
        print(f"Testing fights: {len(self.test_data)}")

        # Model performance
        print(f"\n🎯 MODEL PERFORMANCE:")
        for model_name, results in self.results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  • Accuracy: {results['accuracy']:.1%}")
            print(f"  • Correct predictions: {results['correct_predictions']}/{results['total_predictions']}")
            if 'avg_confidence' in results:
                print(f"  • Average confidence: {results['avg_confidence']:.1%}")

        # Betting simulation
        if self.betting_results:
            print(f"\n💰 BETTING SIMULATION (Even Money Bets):")
            for strategy_name, results in self.betting_results.items():
                print(f"\n{strategy_name.upper().replace('_', ' ')}:")
                print(f"  • ROI: {results['roi']:.1%}")
                print(f"  • Win Rate: {results['win_rate']:.1%}")
                print(f"  • Total Bets: {results['total_bets']}")
                print(f"  • Wins/Losses: {results['wins']}/{results['losses']}")
                if results['total_wagered'] > 0:
                    print(f"  • Total Wagered: ${results['total_wagered']:,}")
                    print(f"  • Total Returned: ${results['total_returned']:,}")
                    print(f"  • Profit/Loss: ${results['total_returned'] - results['total_wagered']:,}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"• Best performing model: {best_model[0]} ({best_model[1]['accuracy']:.1%} accuracy)")

        if self.betting_results:
            best_strategy = max(self.betting_results.items(), key=lambda x: x[1]['roi'])
            print(f"• Best betting strategy: {best_strategy[0]} ({best_strategy[1]['roi']:.1%} ROI)")

            # Break-even analysis
            breakeven_rate = 0.5238  # ~52.38% needed to break even with -110 odds
            for model_name, results in self.results.items():
                if results['accuracy'] > breakeven_rate:
                    print(f"• {model_name} shows potential profitability (>{breakeven_rate:.1%} accuracy needed)")

        print("="*80)

    def run_full_backtest(self):
        """Run complete backtesting suite."""
        print("🚀 STARTING UFC MODEL BACKTESTING")
        print("="*50)

        # Load and prepare data
        if not self.load_data():
            return False

        # Split data temporally
        if not self.split_data_by_date():
            return False

        # Run model backtests
        self.backtest_fighter_matchup_predictor()
        self.backtest_individual_trees()

        # Simulate betting
        self.simulate_betting_strategy()

        # Generate final report
        self.generate_report()

        return True

def main():
    """Main backtesting function."""
    backtester = UFCModelBacktester()

    try:
        success = backtester.run_full_backtest()
        if success:
            print("\n✅ Backtesting completed successfully!")
        else:
            print("\n❌ Backtesting failed!")
    except Exception as e:
        print(f"\n❌ Backtesting error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()