#!/usr/bin/env python3
"""
Comprehensive Backtesting for Unified UFC Prediction System
===========================================================

Validates the unified enhanced random forest system across different time periods
and compares performance to ensure the consolidation didn't introduce regressions.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/Users/ralphfrancolini/UFCML')
from enhanced_random_forest import EnhancedUFCRandomForest
from enhanced_feature_engineering import EnhancedFeatureEngineer

class UFCBacktester:
    """Comprehensive backtesting system for UFC predictions."""

    def __init__(self):
        self.engineer = None
        self.results = []

    def load_data(self):
        """Load and prepare data for backtesting."""
        print("📊 Loading UFC data for backtesting...")

        self.engineer = EnhancedFeatureEngineer()
        if not self.engineer.load_and_prepare_data():
            print("❌ Failed to load data")
            return False

        self.enhanced_df = self.engineer.create_enhanced_training_data()

        # Ensure we have date information
        if 'date' not in self.enhanced_df.columns:
            print("❌ No date information available for temporal backtesting")
            return False

        self.enhanced_df['date'] = pd.to_datetime(self.enhanced_df['date'], errors='coerce')
        self.enhanced_df = self.enhanced_df.dropna(subset=['date']).sort_values('date')

        print(f"✅ Loaded {len(self.enhanced_df)} fights with dates")
        print(f"📅 Date range: {self.enhanced_df['date'].min().strftime('%Y-%m-%d')} to {self.enhanced_df['date'].max().strftime('%Y-%m-%d')}")

        return True

    def temporal_backtest(self, start_date=None, end_date=None, window_months=6, step_months=3):
        """
        Perform temporal backtesting by walking forward through time.

        Args:
            start_date: Start date for backtesting (defaults to data start + 2 years)
            end_date: End date for backtesting (defaults to data end - 6 months)
            window_months: Training window size in months
            step_months: Step size between backtest periods
        """
        print(f"\n🕐 TEMPORAL BACKTESTING")
        print("=" * 50)

        if start_date is None:
            start_date = self.enhanced_df['date'].min() + pd.DateOffset(years=2)
        if end_date is None:
            end_date = self.enhanced_df['date'].max() - pd.DateOffset(months=6)

        print(f"📅 Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"🪟 Training window: {window_months} months")
        print(f"📏 Step size: {step_months} months")

        current_date = start_date
        backtest_results = []

        while current_date <= end_date:
            # Define training and testing periods
            train_end = current_date
            train_start = train_end - pd.DateOffset(months=window_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=step_months)

            print(f"\n📊 Period: {train_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")

            # Get data for this period
            train_data = self.enhanced_df[
                (self.enhanced_df['date'] >= train_start) &
                (self.enhanced_df['date'] < train_end)
            ]

            test_data = self.enhanced_df[
                (self.enhanced_df['date'] >= test_start) &
                (self.enhanced_df['date'] < test_end)
            ]

            if len(train_data) < 100 or len(test_data) < 20:
                print(f"⚠️  Insufficient data: {len(train_data)} train, {len(test_data)} test")
                current_date += pd.DateOffset(months=step_months)
                continue

            print(f"📈 Training: {len(train_data)} fights")
            print(f"📉 Testing: {len(test_data)} fights")

            # Train model for this period
            model = EnhancedUFCRandomForest(
                n_estimators=100,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10
            )
            model.feature_engineer = self.engineer

            try:
                # Train on the period data
                exclude_cols = ['fighter_a', 'fighter_b', 'winner', 'target', 'weight_class', 'date', 'style_a', 'style_b']
                feature_cols = [col for col in train_data.columns if col not in exclude_cols]

                X_train = train_data[feature_cols].fillna(0)
                X_test = test_data[feature_cols].fillna(0)
                y_train = train_data['target']
                y_test = test_data['target']

                # Train the model
                model.model.fit(X_train, y_train)

                # Evaluate
                train_acc = model.model.score(X_train, y_train)
                test_acc = model.model.score(X_test, y_test)

                # Get predictions for analysis
                test_pred = model.model.predict(X_test)
                test_proba = model.model.predict_proba(X_test)[:, 1]

                # Calculate additional metrics
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

                precision = precision_score(y_test, test_pred)
                recall = recall_score(y_test, test_pred)
                f1 = f1_score(y_test, test_pred)
                auc = roc_auc_score(y_test, test_proba)

                result = {
                    'period_start': train_start,
                    'period_end': test_end,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'overfitting_gap': train_acc - test_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_score': auc
                }

                backtest_results.append(result)

                print(f"   Train: {train_acc:.1%}")
                print(f"   Test:  {test_acc:.1%}")
                print(f"   Gap:   {train_acc - test_acc:.1%}")
                print(f"   AUC:   {auc:.3f}")

            except Exception as e:
                print(f"❌ Error in period: {e}")

            current_date += pd.DateOffset(months=step_months)

        self.backtest_results = backtest_results
        return backtest_results

    def analyze_results(self):
        """Analyze backtesting results."""
        if not hasattr(self, 'backtest_results') or not self.backtest_results:
            print("❌ No backtest results to analyze")
            return

        results_df = pd.DataFrame(self.backtest_results)

        print(f"\n📊 BACKTESTING ANALYSIS")
        print("=" * 50)

        # Overall statistics
        print(f"📈 Overall Performance:")
        print(f"   Mean test accuracy: {results_df['test_accuracy'].mean():.1%}")
        print(f"   Std test accuracy:  {results_df['test_accuracy'].std():.1%}")
        print(f"   Min test accuracy:  {results_df['test_accuracy'].min():.1%}")
        print(f"   Max test accuracy:  {results_df['test_accuracy'].max():.1%}")

        print(f"\n🎯 Consistency Metrics:")
        print(f"   Mean overfitting gap: {results_df['overfitting_gap'].mean():.1%}")
        print(f"   Std overfitting gap:  {results_df['overfitting_gap'].std():.1%}")
        print(f"   Mean AUC score:       {results_df['auc_score'].mean():.3f}")
        print(f"   Mean F1 score:        {results_df['f1_score'].mean():.3f}")

        # Performance trends
        results_df['period'] = results_df['period_start'].dt.strftime('%Y-%m')

        print(f"\n📅 Performance by Period:")
        for _, row in results_df.iterrows():
            print(f"   {row['period']}: {row['test_accuracy']:.1%} (gap: {row['overfitting_gap']:.1%})")

        # Check for consistent performance
        accuracy_cv = results_df['test_accuracy'].std() / results_df['test_accuracy'].mean()
        print(f"\n📊 Performance Stability:")
        print(f"   Coefficient of variation: {accuracy_cv:.3f}")

        if accuracy_cv < 0.05:
            print("   ✅ Very stable performance")
        elif accuracy_cv < 0.10:
            print("   ✅ Stable performance")
        else:
            print("   ⚠️  Variable performance - investigate further")

        return results_df

    def plot_results(self, results_df):
        """Create visualizations of backtesting results."""
        print(f"\n📊 Creating backtest visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('UFC Model Backtesting Results', fontsize=16, fontweight='bold')

        # 1. Accuracy over time
        axes[0, 0].plot(results_df.index, results_df['test_accuracy'], 'b-o', label='Test Accuracy')
        axes[0, 0].plot(results_df.index, results_df['train_accuracy'], 'r--', alpha=0.7, label='Train Accuracy')
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Overfitting gap
        axes[0, 1].plot(results_df.index, results_df['overfitting_gap'], 'g-o')
        axes[0, 1].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
        axes[0, 1].axhline(y=0.10, color='red', linestyle='--', alpha=0.7, label='10% threshold')
        axes[0, 1].set_title('Overfitting Gap Over Time')
        axes[0, 1].set_ylabel('Overfitting Gap')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. AUC score
        axes[1, 0].plot(results_df.index, results_df['auc_score'], 'purple', marker='o')
        axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        axes[1, 0].set_title('AUC Score Over Time')
        axes[1, 0].set_ylabel('AUC Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Accuracy distribution
        axes[1, 1].hist(results_df['test_accuracy'], bins=10, alpha=0.7, color='blue')
        axes[1, 1].axvline(results_df['test_accuracy'].mean(), color='red', linestyle='--',
                          label=f'Mean: {results_df["test_accuracy"].mean():.1%}')
        axes[1, 1].set_title('Test Accuracy Distribution')
        axes[1, 1].set_xlabel('Test Accuracy')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('ufc_backtest_results.png', dpi=300, bbox_inches='tight')
        print("✅ Backtest visualizations saved as 'ufc_backtest_results.png'")

    def run_comprehensive_backtest(self):
        """Run the complete backtesting suite."""
        print("🧪 COMPREHENSIVE UFC MODEL BACKTESTING")
        print("=" * 60)

        if not self.load_data():
            return False

        # Run temporal backtesting
        results = self.temporal_backtest()

        if not results:
            print("❌ No backtest results generated")
            return False

        # Analyze results
        results_df = self.analyze_results()

        # Create visualizations
        self.plot_results(results_df)

        # Summary assessment
        print(f"\n🎯 BACKTESTING SUMMARY")
        print("=" * 50)

        mean_acc = results_df['test_accuracy'].mean()
        std_acc = results_df['test_accuracy'].std()
        mean_gap = results_df['overfitting_gap'].mean()

        print(f"✅ Backtesting completed successfully!")
        print(f"📊 {len(results)} periods tested")
        print(f"🎯 Average accuracy: {mean_acc:.1%} (±{std_acc:.1%})")
        print(f"📈 Average overfitting gap: {mean_gap:.1%}")

        if mean_acc > 0.70 and mean_gap < 0.08:
            print("🏆 EXCELLENT: Model shows strong and consistent performance")
        elif mean_acc > 0.65 and mean_gap < 0.12:
            print("✅ GOOD: Model performance is acceptable")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Consider model adjustments")

        return True

def main():
    """Run comprehensive backtesting."""
    backtester = UFCBacktester()
    backtester.run_comprehensive_backtest()

if __name__ == "__main__":
    main()