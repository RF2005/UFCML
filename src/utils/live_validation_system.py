#!/usr/bin/env python3
"""
Live Validation System for UFC Predictions
==========================================

Continuously monitors and validates model performance on new fights as they occur.
Provides real-time feedback on model accuracy and alerts for performance degradation.

Features:
1. Automatic validation when new fight results become available
2. Performance monitoring and alerting
3. Model drift detection
4. Prediction confidence calibration analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import warnings

sys.path.append('/Users/ralphfrancolini/UFCML')
from future_fight_predictor import FutureFightPredictor
from tests.validation.proper_temporal_holdout_test import ProperTemporalHoldoutTester

class LiveValidationMonitor:
    """Continuously monitor and validate UFC prediction model performance."""

    def __init__(self):
        self.predictor = FutureFightPredictor()
        self.validation_log = 'live_validation_log.json'
        self.performance_history = 'performance_history.json'
        self.alert_thresholds = {
            'accuracy_drop': 0.10,  # Alert if accuracy drops by 10%
            'min_sample_size': 20,  # Minimum fights before calculating accuracy
            'confidence_miscalibration': 0.15  # Alert if confidence is miscalibrated
        }

    def validate_recent_predictions(self, days_back=30):
        """
        Validate predictions from the last N days against actual results.

        Args:
            days_back: Number of days to look back for validation
        """
        print(f"🔍 VALIDATING RECENT PREDICTIONS ({days_back} days)")
        print("=" * 60)

        # Load predictions
        verified_predictions = self._get_verified_predictions(days_back)

        if not verified_predictions:
            print(f"📊 No verified predictions found in the last {days_back} days")
            return None

        # Calculate performance metrics
        results = self._calculate_performance_metrics(verified_predictions)

        # Update performance history
        self._update_performance_history(results)

        # Check for alerts
        self._check_performance_alerts(results)

        # Display results
        self._display_validation_results(results)

        return results

    def _get_verified_predictions(self, days_back):
        """Get verified predictions from the specified time period."""
        try:
            with open(self.predictor.predictions_log, 'r') as f:
                all_predictions = json.load(f)

            cutoff_date = datetime.now() - timedelta(days=days_back)

            verified_predictions = []
            for pred in all_predictions:
                if (pred['result_verified'] and
                    datetime.strptime(pred['fight_date'], '%Y-%m-%d') >= cutoff_date):
                    verified_predictions.append(pred)

            return verified_predictions

        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return []

    def _calculate_performance_metrics(self, predictions):
        """Calculate comprehensive performance metrics."""
        if not predictions:
            return None

        total_predictions = len(predictions)
        correct_predictions = sum(1 for p in predictions if p['correct'])
        accuracy = correct_predictions / total_predictions

        # Confidence calibration analysis
        confidence_ranges = {
            'low': [p for p in predictions if p['confidence'] < 0.6],
            'medium': [p for p in predictions if 0.6 <= p['confidence'] < 0.8],
            'high': [p for p in predictions if p['confidence'] >= 0.8]
        }

        calibration_analysis = {}
        for range_name, range_preds in confidence_ranges.items():
            if range_preds:
                range_accuracy = sum(1 for p in range_preds if p['correct']) / len(range_preds)
                avg_confidence = np.mean([p['confidence'] for p in range_preds])
                calibration_error = abs(avg_confidence - range_accuracy)

                calibration_analysis[range_name] = {
                    'count': len(range_preds),
                    'accuracy': range_accuracy,
                    'avg_confidence': avg_confidence,
                    'calibration_error': calibration_error
                }

        # Performance by weight class
        weight_class_performance = defaultdict(list)
        for pred in predictions:
            if pred.get('weight_class'):
                weight_class_performance[pred['weight_class']].append(pred['correct'])

        weight_class_accuracy = {}
        for weight_class, results in weight_class_performance.items():
            if len(results) >= 3:  # Only include classes with enough data
                weight_class_accuracy[weight_class] = sum(results) / len(results)

        # Recent trend analysis (if we have enough data)
        trend_analysis = None
        if len(predictions) >= 10:
            # Sort by date and split into two halves
            sorted_preds = sorted(predictions, key=lambda x: x['fight_date'])
            half_point = len(sorted_preds) // 2
            early_half = sorted_preds[:half_point]
            recent_half = sorted_preds[half_point:]

            early_accuracy = sum(1 for p in early_half if p['correct']) / len(early_half)
            recent_accuracy = sum(1 for p in recent_half if p['correct']) / len(recent_half)

            trend_analysis = {
                'early_period_accuracy': early_accuracy,
                'recent_period_accuracy': recent_accuracy,
                'trend_direction': 'improving' if recent_accuracy > early_accuracy else 'declining',
                'trend_magnitude': abs(recent_accuracy - early_accuracy)
            }

        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'calibration_analysis': calibration_analysis,
            'weight_class_accuracy': weight_class_accuracy,
            'trend_analysis': trend_analysis,
            'predictions': predictions
        }

    def _update_performance_history(self, results):
        """Update the performance history log."""
        try:
            # Load existing history
            try:
                with open(self.performance_history, 'r') as f:
                    history = json.load(f)
            except FileNotFoundError:
                history = []

            # Add current results (without full predictions to save space)
            history_entry = results.copy()
            history_entry.pop('predictions', None)  # Remove detailed predictions
            history.append(history_entry)

            # Keep only last 100 entries
            if len(history) > 100:
                history = history[-100:]

            # Save updated history
            with open(self.performance_history, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            print(f"⚠️  Warning: Could not update performance history: {e}")

    def _check_performance_alerts(self, results):
        """Check for performance issues and generate alerts."""
        alerts = []

        # Check accuracy drop
        try:
            with open(self.performance_history, 'r') as f:
                history = json.load(f)

            if len(history) >= 2:
                previous_accuracy = history[-2]['accuracy']
                current_accuracy = results['accuracy']
                accuracy_change = previous_accuracy - current_accuracy

                if accuracy_change > self.alert_thresholds['accuracy_drop']:
                    alerts.append({
                        'type': 'accuracy_drop',
                        'severity': 'high',
                        'message': f"Accuracy dropped by {accuracy_change:.1%} from {previous_accuracy:.1%} to {current_accuracy:.1%}",
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

        except (FileNotFoundError, IndexError, KeyError):
            pass  # No previous data available

        # Check sample size
        if results['total_predictions'] < self.alert_thresholds['min_sample_size']:
            alerts.append({
                'type': 'small_sample',
                'severity': 'low',
                'message': f"Small sample size: only {results['total_predictions']} predictions validated",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        # Check confidence calibration
        for range_name, calibration in results['calibration_analysis'].items():
            if calibration['calibration_error'] > self.alert_thresholds['confidence_miscalibration']:
                alerts.append({
                    'type': 'miscalibration',
                    'severity': 'medium',
                    'message': f"Confidence miscalibration in {range_name} range: {calibration['calibration_error']:.1%} error",
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        # Display alerts
        if alerts:
            print(f"\n🚨 PERFORMANCE ALERTS ({len(alerts)})")
            print("=" * 50)
            for alert in alerts:
                severity_emoji = {'low': '🟡', 'medium': '🟠', 'high': '🔴'}
                print(f"{severity_emoji[alert['severity']]} {alert['type'].upper()}: {alert['message']}")

            # Log alerts
            self._log_alerts(alerts)
        else:
            print(f"\n✅ No performance alerts")

        return alerts

    def _log_alerts(self, alerts):
        """Log alerts for future reference."""
        try:
            alert_log = 'performance_alerts.json'

            # Load existing alerts
            try:
                with open(alert_log, 'r') as f:
                    existing_alerts = json.load(f)
            except FileNotFoundError:
                existing_alerts = []

            # Add new alerts
            existing_alerts.extend(alerts)

            # Keep only last 200 alerts
            if len(existing_alerts) > 200:
                existing_alerts = existing_alerts[-200:]

            # Save updated alerts
            with open(alert_log, 'w') as f:
                json.dump(existing_alerts, f, indent=2)

        except Exception as e:
            print(f"⚠️  Warning: Could not log alerts: {e}")

    def _display_validation_results(self, results):
        """Display comprehensive validation results."""
        print(f"\n📊 VALIDATION RESULTS")
        print("=" * 50)
        print(f"🎯 Overall Accuracy: {results['accuracy']:.1%} ({results['correct_predictions']}/{results['total_predictions']})")

        # Confidence calibration
        print(f"\n🎛️  Confidence Calibration:")
        for range_name, calibration in results['calibration_analysis'].items():
            if calibration['count'] > 0:
                print(f"  {range_name.title()} confidence ({calibration['count']} fights):")
                print(f"    Average confidence: {calibration['avg_confidence']:.1%}")
                print(f"    Actual accuracy: {calibration['accuracy']:.1%}")
                print(f"    Calibration error: {calibration['calibration_error']:.1%}")

        # Weight class performance
        if results['weight_class_accuracy']:
            print(f"\n⚖️  Performance by Weight Class:")
            for weight_class, accuracy in results['weight_class_accuracy'].items():
                print(f"  {weight_class}: {accuracy:.1%}")

        # Trend analysis
        if results['trend_analysis']:
            trend = results['trend_analysis']
            print(f"\n📈 Trend Analysis:")
            print(f"  Early period: {trend['early_period_accuracy']:.1%}")
            print(f"  Recent period: {trend['recent_period_accuracy']:.1%}")
            print(f"  Trend: {trend['trend_direction']} ({trend['trend_magnitude']:.1%})")

    def generate_performance_report(self, save_plot=True):
        """Generate a comprehensive performance report with visualizations."""
        print(f"📊 GENERATING PERFORMANCE REPORT")
        print("=" * 50)

        try:
            with open(self.performance_history, 'r') as f:
                history = json.load(f)

            if len(history) < 2:
                print("📊 Insufficient data for performance report")
                return

            # Extract data for plotting
            timestamps = [datetime.strptime(h['timestamp'], '%Y-%m-%d %H:%M:%S') for h in history]
            accuracies = [h['accuracy'] for h in history]
            sample_sizes = [h['total_predictions'] for h in history]

            # Create performance plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Accuracy over time
            ax1.plot(timestamps, accuracies, 'b-o', linewidth=2, markersize=6)
            ax1.axhline(y=0.70, color='green', linestyle='--', alpha=0.7, label='Good threshold (70%)')
            ax1.axhline(y=0.60, color='orange', linestyle='--', alpha=0.7, label='Warning threshold (60%)')
            ax1.axhline(y=0.50, color='red', linestyle='--', alpha=0.7, label='Poor threshold (50%)')
            ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0.4, 1.0)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Sample sizes
            ax2.bar(timestamps, sample_sizes, alpha=0.7, color='skyblue', width=1)
            ax2.set_title('Validation Sample Sizes', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Predictions')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_plot:
                plot_filename = f"live_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"📊 Performance report saved as {plot_filename}")

            plt.show()

        except Exception as e:
            print(f"❌ Error generating performance report: {e}")

    def run_daily_validation(self):
        """Run daily validation check (can be automated)."""
        print(f"🔄 DAILY VALIDATION CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Validate last 7 days
        results = self.validate_recent_predictions(days_back=7)

        if results:
            # Generate weekly report if enough data
            if results['total_predictions'] >= 5:
                print(f"\n📊 Weekly validation complete")
            else:
                print(f"\n📊 Limited validation data this week")

        return results

def main():
    """Demonstrate the live validation system."""
    monitor = LiveValidationMonitor()

    # Run validation on recent predictions
    results = monitor.validate_recent_predictions(days_back=30)

    if results:
        # Generate performance report
        monitor.generate_performance_report()

        # Demonstrate daily validation
        monitor.run_daily_validation()
    else:
        print("📊 No verified predictions available for validation")
        print("💡 Use the FutureFightPredictor to make predictions, then update with actual results")

if __name__ == "__main__":
    main()