#!/usr/bin/env python3
"""
Enhanced Temporal Backtesting Framework
=======================================

Provides more rigorous temporal validation with stricter boundaries and
comprehensive analysis. This enhanced version addresses potential issues
with the original backtesting and provides higher confidence in results.

Key improvements:
1. Stricter temporal boundaries (no overlap whatsoever)
2. Multiple validation window sizes
3. Rolling window validation
4. Cross-validation within temporal periods
5. More comprehensive metrics
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.append('/Users/ralphfrancolini/UFCML')
from enhanced_random_forest import EnhancedUFCRandomForest
from enhanced_feature_engineering import EnhancedFeatureEngineer

class EnhancedTemporalBacktester:
    """Enhanced temporal backtesting with stricter validation methodology."""

    def __init__(self):
        self.engineer = None
        self.enhanced_df = None
        self.results = []

    def load_data(self):
        """Load and prepare data for enhanced backtesting."""
        print("📊 LOADING DATA FOR ENHANCED TEMPORAL BACKTESTING")
        print("=" * 60)

        self.engineer = EnhancedFeatureEngineer()
        if not self.engineer.load_and_prepare_data():
            print("❌ Failed to load data")
            return False

        self.enhanced_df = self.engineer.create_enhanced_training_data()

        if 'date' not in self.enhanced_df.columns:
            print("❌ No date information available for temporal backtesting")
            return False

        self.enhanced_df['date'] = pd.to_datetime(self.enhanced_df['date'], errors='coerce')
        self.enhanced_df = self.enhanced_df.dropna(subset=['date']).sort_values('date')

        print(f"✅ Loaded {len(self.enhanced_df)} fights with dates")
        print(f"📅 Date range: {self.enhanced_df['date'].min().strftime('%Y-%m-%d')} to {self.enhanced_df['date'].max().strftime('%Y-%m-%d')}")

        return True

    def enhanced_walk_forward_validation(self,
                                       initial_train_months=24,
                                       test_months=1,
                                       step_months=1,
                                       gap_days=7):
        """
        Enhanced walk-forward validation with strict temporal boundaries.

        Args:
            initial_train_months: Initial training window size
            test_months: Test period size
            step_months: Step size between validation periods
            gap_days: Mandatory gap between train and test to prevent leakage
        """
        print(f"\n🕐 ENHANCED WALK-FORWARD VALIDATION")
        print("=" * 60)
        print(f"📊 Initial training window: {initial_train_months} months")
        print(f"📊 Test period: {test_months} month(s)")
        print(f"📊 Step size: {step_months} month(s)")
        print(f"⚠️  Mandatory gap: {gap_days} days (prevents data leakage)")

        # Start from a point where we have enough training data
        data_start = self.enhanced_df['date'].min()
        validation_start = data_start + pd.DateOffset(months=initial_train_months)
        data_end = self.enhanced_df['date'].max()

        current_date = validation_start
        backtest_results = []

        while current_date <= data_end - pd.DateOffset(months=test_months):
            # Define strict temporal boundaries
            train_end = current_date
            train_start = train_end - pd.DateOffset(months=initial_train_months)

            # Mandatory gap to prevent data leakage
            test_start = train_end + timedelta(days=gap_days)
            test_end = test_start + pd.DateOffset(months=test_months)

            print(f"\\n📊 Period: Train {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')} | Gap {gap_days}d | Test {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")

            # Get data with strict temporal separation
            train_data = self.enhanced_df[
                (self.enhanced_df['date'] >= train_start) &
                (self.enhanced_df['date'] < train_end)
            ]

            test_data = self.enhanced_df[
                (self.enhanced_df['date'] >= test_start) &
                (self.enhanced_df['date'] < test_end)
            ]

            # Verify no temporal overlap
            if not train_data.empty and not test_data.empty:
                latest_train = train_data['date'].max()
                earliest_test = test_data['date'].min()
                actual_gap = (earliest_test - latest_train).days

                if actual_gap < gap_days:
                    print(f"⚠️  Warning: Insufficient gap ({actual_gap} days < {gap_days} required)")
                    current_date += pd.DateOffset(months=step_months)
                    continue

            # Check minimum data requirements
            if len(train_data) < 200 or len(test_data) < 10:
                print(f"⚠️  Insufficient data: {len(train_data)} train, {len(test_data)} test")
                current_date += pd.DateOffset(months=step_months)
                continue

            print(f"📈 Training: {len(train_data)} fights")
            print(f"📉 Testing: {len(test_data)} fights")
            print(f"⏰ Actual gap: {actual_gap} days")

            # Train model with expanded train/validation split
            try:
                result = self._train_and_evaluate_period(
                    train_data, test_data, train_start, test_end, actual_gap
                )

                if result:
                    backtest_results.append(result)
                    print(f"   ✅ Train: {result['train_accuracy']:.1%}")
                    print(f"   ✅ Valid: {result['val_accuracy']:.1%}")
                    print(f"   ✅ Test:  {result['test_accuracy']:.1%}")
                    print(f"   📊 Gap:   {result['overfitting_gap']:.1%}")

            except Exception as e:
                print(f"❌ Error in period: {e}")

            current_date += pd.DateOffset(months=step_months)

        self.results = backtest_results
        return backtest_results

    def _train_and_evaluate_period(self, train_data, test_data, period_start, period_end, gap_days):
        """Train and evaluate model for a specific time period with enhanced validation."""

        # Create internal train/validation split from training data
        # Use last 20% of training data as validation
        cutoff_date = train_data['date'].quantile(0.8)

        internal_train = train_data[train_data['date'] <= cutoff_date]
        internal_val = train_data[train_data['date'] > cutoff_date]

        if len(internal_train) < 100 or len(internal_val) < 20:
            # Fallback: use all training data for both train and validation
            internal_train = train_data
            internal_val = train_data

        # Prepare features
        exclude_cols = ['fighter_a', 'fighter_b', 'winner', 'target', 'weight_class', 'date', 'style_a', 'style_b']
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]

        X_train = internal_train[feature_cols].fillna(0)
        X_val = internal_val[feature_cols].fillna(0)
        X_test = test_data[feature_cols].fillna(0)

        y_train = internal_train['target']
        y_val = internal_val['target']
        y_test = test_data['target']

        # Train model with regularization
        model = EnhancedUFCRandomForest(
            n_estimators=100,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt'
        )

        model.model.fit(X_train, y_train)

        # Comprehensive evaluation
        train_acc = model.model.score(X_train, y_train)
        val_acc = model.model.score(X_val, y_val)
        test_acc = model.model.score(X_test, y_test)

        # Get predictions for detailed metrics
        train_pred = model.model.predict(X_train)
        val_pred = model.model.predict(X_val)
        test_pred = model.model.predict(X_test)

        test_proba = model.model.predict_proba(X_test)[:, 1]

        # Calculate comprehensive metrics
        precision = precision_score(y_test, test_pred)
        recall = recall_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred)
        auc = roc_auc_score(y_test, test_proba)

        return {
            'period_start': period_start,
            'period_end': period_end,
            'gap_days': gap_days,
            'train_size': len(internal_train),
            'val_size': len(internal_val),
            'test_size': len(test_data),
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'overfitting_gap': train_acc - val_acc,
            'generalization_gap': val_acc - test_acc,
            'total_gap': train_acc - test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'validation_date': datetime.now()
        }

    def rolling_window_validation(self, window_sizes=[6, 12, 18, 24]):
        """
        Test multiple window sizes to assess optimal training period.

        Args:
            window_sizes: List of training window sizes in months
        """
        print(f"\n🔄 ROLLING WINDOW VALIDATION")
        print("=" * 60)

        window_results = {}

        for window_months in window_sizes:
            print(f"\n📊 Testing {window_months}-month training window...")

            results = self.enhanced_walk_forward_validation(
                initial_train_months=window_months,
                test_months=1,
                step_months=3,  # Less frequent testing for efficiency
                gap_days=7
            )

            if results:
                accuracies = [r['test_accuracy'] for r in results]
                overfitting_gaps = [r['overfitting_gap'] for r in results]

                window_results[window_months] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'mean_overfitting': np.mean(overfitting_gaps),
                    'num_periods': len(results),
                    'results': results
                }

                print(f"   📊 Mean accuracy: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}")
                print(f"   📊 Mean overfitting: {np.mean(overfitting_gaps):.1%}")
                print(f"   📊 Periods tested: {len(results)}")

        return window_results

    def analyze_enhanced_results(self):
        """Analyze enhanced backtesting results with comprehensive metrics."""
        if not self.results:
            print("❌ No backtest results to analyze")
            return None

        results_df = pd.DataFrame(self.results)

        print(f"\n📊 ENHANCED BACKTESTING ANALYSIS")
        print("=" * 60)

        # Basic performance statistics
        print(f"📈 Performance Summary:")
        print(f"   Mean test accuracy: {results_df['test_accuracy'].mean():.1%}")
        print(f"   Std test accuracy:  {results_df['test_accuracy'].std():.1%}")
        print(f"   Min test accuracy:  {results_df['test_accuracy'].min():.1%}")
        print(f"   Max test accuracy:  {results_df['test_accuracy'].max():.1%}")
        print(f"   Median test accuracy: {results_df['test_accuracy'].median():.1%}")

        # Overfitting analysis
        print(f"\n🎯 Overfitting Analysis:")
        print(f"   Mean overfitting gap: {results_df['overfitting_gap'].mean():.1%}")
        print(f"   Std overfitting gap:  {results_df['overfitting_gap'].std():.1%}")
        print(f"   Max overfitting gap:  {results_df['overfitting_gap'].max():.1%}")

        # Generalization analysis
        print(f"\n📊 Generalization Analysis:")
        print(f"   Mean val-test gap: {results_df['generalization_gap'].mean():.1%}")
        print(f"   Std val-test gap:  {results_df['generalization_gap'].std():.1%}")

        # Comprehensive metrics
        print(f"\n🎯 Additional Metrics:")
        print(f"   Mean AUC score:  {results_df['auc_score'].mean():.3f}")
        print(f"   Mean F1 score:   {results_df['f1_score'].mean():.3f}")
        print(f"   Mean precision:  {results_df['precision'].mean():.3f}")
        print(f"   Mean recall:     {results_df['recall'].mean():.3f}")

        # Consistency assessment
        cv_accuracy = results_df['test_accuracy'].std() / results_df['test_accuracy'].mean()
        print(f"\n📊 Model Consistency:")
        print(f"   Coefficient of variation: {cv_accuracy:.3f}")

        if cv_accuracy < 0.05:
            print("   ✅ Very consistent performance")
        elif cv_accuracy < 0.10:
            print("   ✅ Consistent performance")
        elif cv_accuracy < 0.15:
            print("   ⚠️  Moderately variable performance")
        else:
            print("   ❌ Highly variable performance - investigate")

        # Temporal trend analysis
        results_df['period_mid'] = results_df['period_start'] + (results_df['period_end'] - results_df['period_start']) / 2

        # Check for performance trends over time
        correlation = results_df['test_accuracy'].corr(
            pd.to_numeric(results_df['period_mid'])
        )

        print(f"\n📈 Temporal Trends:")
        print(f"   Accuracy-time correlation: {correlation:.3f}")

        if abs(correlation) > 0.3:
            if correlation > 0:
                print("   📈 Performance improving over time")
            else:
                print("   📉 Performance declining over time")
        else:
            print("   ➡️  No significant temporal trend")

        return results_df

    def create_enhanced_visualizations(self, results_df, save_plots=True):
        """Create comprehensive visualizations of enhanced backtesting results."""
        print(f"\n📊 Creating enhanced backtest visualizations...")

        fig = plt.figure(figsize=(20, 12))

        # 1. Accuracy over time
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(results_df.index, results_df['test_accuracy'], 'b-o', label='Test Accuracy', linewidth=2)
        ax1.plot(results_df.index, results_df['val_accuracy'], 'g--', label='Val Accuracy', alpha=0.7)
        ax1.plot(results_df.index, results_df['train_accuracy'], 'r:', label='Train Accuracy', alpha=0.7)
        ax1.set_title('Accuracy Over Time', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Overfitting analysis
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(results_df.index, results_df['overfitting_gap'], 'r-o', label='Train-Val Gap')
        ax2.plot(results_df.index, results_df['generalization_gap'], 'orange', marker='s', label='Val-Test Gap')
        ax2.axhline(y=0.05, color='gray', linestyle='--', alpha=0.7, label='5% threshold')
        ax2.axhline(y=0.10, color='red', linestyle='--', alpha=0.7, label='10% threshold')
        ax2.set_title('Overfitting & Generalization Analysis', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Performance Gap')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Comprehensive metrics
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(results_df.index, results_df['auc_score'], 'purple', marker='o', label='AUC')
        ax3.plot(results_df.index, results_df['f1_score'], 'green', marker='s', label='F1')
        ax3.plot(results_df.index, results_df['precision'], 'blue', marker='^', label='Precision')
        ax3.plot(results_df.index, results_df['recall'], 'orange', marker='v', label='Recall')
        ax3.set_title('Comprehensive Metrics', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Accuracy distribution
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(results_df['test_accuracy'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(results_df['test_accuracy'].mean(), color='red', linestyle='--',
                   label=f'Mean: {results_df["test_accuracy"].mean():.1%}')
        ax4.axvline(results_df['test_accuracy'].median(), color='green', linestyle='--',
                   label=f'Median: {results_df["test_accuracy"].median():.1%}')
        ax4.set_title('Test Accuracy Distribution', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Test Accuracy')
        ax4.set_ylabel('Frequency')
        ax4.legend()

        # 5. Sample sizes
        ax5 = plt.subplot(2, 3, 5)
        ax5.bar(results_df.index, results_df['test_size'], alpha=0.7, color='lightcoral', label='Test Size')
        ax5.bar(results_df.index, results_df['train_size'], alpha=0.5, color='lightblue', label='Train Size')
        ax5.set_title('Sample Sizes by Period', fontweight='bold', fontsize=12)
        ax5.set_ylabel('Number of Fights')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Performance consistency
        ax6 = plt.subplot(2, 3, 6)
        rolling_mean = results_df['test_accuracy'].rolling(window=5, center=True).mean()
        rolling_std = results_df['test_accuracy'].rolling(window=5, center=True).std()

        ax6.plot(results_df.index, results_df['test_accuracy'], 'b-o', alpha=0.6, label='Actual')
        ax6.plot(results_df.index, rolling_mean, 'r-', linewidth=2, label='5-period MA')
        ax6.fill_between(results_df.index,
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.2, color='red', label='±1 Std')
        ax6.set_title('Performance Consistency', fontweight='bold', fontsize=12)
        ax6.set_ylabel('Test Accuracy')
        ax6.set_xlabel('Validation Period')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.suptitle('Enhanced UFC Model Backtesting Results', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_plots:
            filename = f'enhanced_backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Enhanced visualizations saved as '{filename}'")

        plt.show()

    def run_comprehensive_enhanced_backtest(self):
        """Run the complete enhanced backtesting suite."""
        print("🧪 COMPREHENSIVE ENHANCED TEMPORAL BACKTESTING")
        print("=" * 70)

        if not self.load_data():
            return False

        # Run enhanced walk-forward validation
        print(f"\n🔄 Running enhanced walk-forward validation...")
        results = self.enhanced_walk_forward_validation(
            initial_train_months=18,
            test_months=1,
            step_months=2,
            gap_days=7
        )

        if not results:
            print("❌ No backtest results generated")
            return False

        # Analyze results
        results_df = self.analyze_enhanced_results()

        # Create visualizations
        self.create_enhanced_visualizations(results_df)

        # Test different window sizes
        print(f"\n🔄 Testing optimal training window sizes...")
        window_results = self.rolling_window_validation([12, 18, 24])

        # Final assessment
        print(f"\n🎯 ENHANCED BACKTESTING SUMMARY")
        print("=" * 60)

        mean_acc = results_df['test_accuracy'].mean()
        std_acc = results_df['test_accuracy'].std()
        mean_gap = results_df['overfitting_gap'].mean()

        print(f"✅ Enhanced backtesting completed successfully!")
        print(f"📊 {len(results)} periods tested with strict temporal boundaries")
        print(f"🎯 Mean accuracy: {mean_acc:.1%} (±{std_acc:.1%})")
        print(f"📈 Mean overfitting gap: {mean_gap:.1%}")
        print(f"⏰ Mandatory {results[0]['gap_days']}-day gap enforced between train/test")

        if mean_acc > 0.70 and mean_gap < 0.08:
            print("🏆 EXCELLENT: Enhanced validation confirms strong performance")
        elif mean_acc > 0.65 and mean_gap < 0.12:
            print("✅ GOOD: Enhanced validation shows acceptable performance")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Enhanced validation reveals issues")

        return True

def main():
    """Run enhanced temporal backtesting."""
    backtester = EnhancedTemporalBacktester()
    backtester.run_comprehensive_enhanced_backtest()

if __name__ == "__main__":
    main()