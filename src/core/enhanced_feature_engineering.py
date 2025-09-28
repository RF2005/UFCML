#!/usr/bin/env python3
"""
Enhanced Feature Engineering for UFC Predictions
=================================================

Builds leakage-free matchup features by rolling fighter statistics forward
chronologically before computing fight-level inputs for tree-based models.
"""

import pandas as pd
from datetime import datetime
from collections import deque
import math
import sys
sys.path.append('/Users/ralphfrancolini/UFCML')

from src.core.advanced_ml_models import load_enhanced_ufc_data


class EnhancedFeatureEngineer:
    """Advanced feature engineering for UFC fight predictions."""

    def __init__(self, recent_window=3):
        self.df = None
        self.recent_window = recent_window
        self.style_matchup_matrix = self._create_style_matchup_matrix()
        self.stat_keys = [
            'sig_str_landed',
            'sig_str_acc',
            'td_landed',
            'td_acc',
            'sub_att',
            'ctrl_time',
        ]
        self._reset_state()

    def _reset_state(self):
        self.fighter_history = {}
        self.fighter_profiles = {}

    def _create_style_matchup_matrix(self):
        """Return fighting style matchup multipliers."""
        return {
            ('striker', 'striker'): 1.0,
            ('striker', 'grappler'): 0.85,
            ('striker', 'wrestler'): 0.90,
            ('striker', 'bjj'): 0.95,
            ('grappler', 'striker'): 1.15,
            ('grappler', 'grappler'): 1.0,
            ('grappler', 'wrestler'): 0.95,
            ('grappler', 'bjj'): 1.05,
            ('wrestler', 'striker'): 1.10,
            ('wrestler', 'grappler'): 1.05,
            ('wrestler', 'wrestler'): 1.0,
            ('wrestler', 'bjj'): 1.08,
            ('bjj', 'striker'): 1.05,
            ('bjj', 'grappler'): 0.95,
            ('bjj', 'wrestler'): 0.92,
            ('bjj', 'bjj'): 1.0,
        }

    def load_and_prepare_data(self):
        """Load UFC data and prepare it for feature generation."""
        print("📊 Loading UFC data for enhanced feature engineering...")
        self.df = load_enhanced_ufc_data()

        if self.df is None:
            print("❌ Failed to load UFC data")
            return False

        self.df = self.df.dropna(subset=['winner', 'date'])
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df = self.df.dropna(subset=['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)

        print(f"✅ Loaded {len(self.df)} fights for feature engineering")
        self._reset_state()
        return True

    def _empty_state(self):
        return {
            'total_fights': 0,
            'wins': 0,
            'losses': 0,
            'recent_results': deque(maxlen=self.recent_window),
            'recent_finishes': deque(maxlen=self.recent_window),
            'recent_sig_str_landed': deque(maxlen=self.recent_window),
            'recent_ctrl_time': deque(maxlen=self.recent_window),
            'recent_td_landed': deque(maxlen=self.recent_window),
            'recent_sub_att': deque(maxlen=self.recent_window),
            'recent_fight_times': deque(maxlen=self.recent_window),
            'stats_totals': {key: 0.0 for key in self.stat_keys},
            'stats_counts': {key: 0 for key in self.stat_keys},
            'finish_count': 0,
            'total_fight_time': 0.0,
            'last_result': 0,
            'last_fight_date': None,
            'first_fight_date': None,
            'date_history': [],
        }

    def _get_or_create_state(self, fighter_name):
        state = self.fighter_history.get(fighter_name)
        if state is None:
            state = self._empty_state()
            self.fighter_history[fighter_name] = state
        return state

    def _extract_fight_stats(self, fight_row, corner):
        prefix = 'r_' if corner == 'red' else 'b_'
        return {
            'sig_str_landed': fight_row.get(f'{prefix}sig_str_landed', 0),
            'sig_str_acc': fight_row.get(f'{prefix}sig_str_acc', 0),
            'td_landed': fight_row.get(f'{prefix}td_landed', 0),
            'td_acc': fight_row.get(f'{prefix}td_acc', 0),
            'sub_att': fight_row.get(f'{prefix}sub_att', 0),
            'ctrl_time': fight_row.get(f'{prefix}ctrl', 0),
        }

    def _update_fighter_state(self, fighter_name, fight_row, corner, fight_date, is_win, fight_winner):
        state = self._get_or_create_state(fighter_name)
        state['total_fights'] += 1
        if is_win:
            state['wins'] += 1
        elif isinstance(fight_winner, str) and fight_winner.lower() in {'draw', 'no contest', 'nc'}:
            pass
        else:
            state['losses'] += 1

        state['recent_results'].append(1 if is_win else 0)

        if state['first_fight_date'] is None:
            state['first_fight_date'] = fight_date
        state['last_fight_date'] = fight_date
        state['date_history'].append(fight_date)

        fight_duration = fight_row.get('match_time_sec')
        if pd.isna(fight_duration) or not fight_duration:
            total_rounds = fight_row.get('total_rounds', None)
            try:
                total_rounds = float(total_rounds)
            except (TypeError, ValueError):
                total_rounds = None
            if total_rounds and total_rounds > 0:
                fight_duration = total_rounds * 5 * 60
            else:
                fight_duration = 15 * 60  # Default to 3 rounds if unknown
        fight_duration = float(fight_duration)
        state['total_fight_time'] += fight_duration
        state['recent_fight_times'].append(fight_duration)

        stats = self._extract_fight_stats(fight_row, corner)
        for key, value in stats.items():
            if pd.isna(value):
                continue
            state['stats_totals'][key] += float(value)
            state['stats_counts'][key] += 1

        state['recent_sig_str_landed'].append(float(stats.get('sig_str_landed') or 0.0))
        state['recent_ctrl_time'].append(float(stats.get('ctrl_time') or 0.0))
        state['recent_td_landed'].append(float(stats.get('td_landed') or 0.0))
        state['recent_sub_att'].append(float(stats.get('sub_att') or 0.0))

        method = fight_row.get('method', '')
        finish_detected = False
        if is_win and isinstance(method, str):
            method_lower = method.lower()
            finish_detected = any(term in method_lower for term in ['ko', 'tko', 'submission', 'sub'])

        if finish_detected:
            state['finish_count'] += 1
        state['recent_finishes'].append(1 if finish_detected else 0)
        state['last_result'] = 1 if is_win else 0

    def _compute_activity_rate(self, date_history, current_date):
        historical_dates = [d for d in date_history if d <= current_date]
        if not historical_dates:
            return 0.0

        span_days = max((current_date - historical_dates[0]).days, 1)
        span_days = max(span_days, 365)
        fights = len(historical_dates)
        return fights * 365.0 / span_days

    def _classify_style_from_averages(self, averages):
        avg_td = averages.get('td_landed', 0) or 0
        avg_sub = averages.get('sub_att', 0) or 0
        avg_ctrl = averages.get('ctrl_time', 0) or 0
        avg_strikes = averages.get('sig_str_landed', 0) or 0

        if avg_td > 2.0 and avg_ctrl > 120:
            return 'wrestler'
        if avg_sub > 1.0 or (avg_td > 1.0 and avg_sub > 0.5):
            return 'bjj'
        if avg_td > 1.5 or avg_ctrl > 90:
            return 'grappler'
        if avg_strikes > 0:
            return 'striker'
        return 'striker'

    def _default_snapshot(self):
        return {
            'total_fights': 0,
            'wins': 0,
            'win_rate': 0.0,
            'recent_form': {
                'weighted_win_rate': 0.0,
                'performance_trend': 0.0,
                'recent_win_rate': 0.0,
            },
            'style': 'striker',
            'avg_stats': {key: 0.0 for key in self.stat_keys},
            'activity_rate': 0.0,
            'days_since_last': 365.0,
            'finish_rate': 0.0,
            'last_fight_date': None,
            'recent_finish_rate': 0.0,
            'average_fight_time': 0.0,
            'recent_average_fight_time': 0.0,
            'striking_pace': 0.0,
            'recent_striking_pace': 0.0,
            'control_per_minute': 0.0,
            'recent_control_per_minute': 0.0,
            'takedown_pace': 0.0,
            'recent_takedown_pace': 0.0,
            'submission_pace': 0.0,
            'recent_submission_pace': 0.0,
            'experience_score': 0.0,
            'last_result': 0,
        }

    def _snapshot_from_state(self, state, current_date):
        if state['total_fights'] == 0:
            return self._default_snapshot()

        current_date = pd.Timestamp(current_date)
        win_rate = state['wins'] / state['total_fights'] if state['total_fights'] else 0.0

        recent_results = list(state['recent_results'])
        recent_win_rate = sum(recent_results) / len(recent_results) if recent_results else win_rate
        weighted_win_rate = recent_win_rate * 0.6 + win_rate * 0.4
        performance_trend = recent_win_rate - win_rate

        last_fight_date = state['last_fight_date']
        if last_fight_date is None:
            days_since_last = 365.0
        else:
            days_since_last = (current_date - last_fight_date).days
            days_since_last = float(max(days_since_last, 0))

        averages = {}
        for key in self.stat_keys:
            count = state['stats_counts'][key]
            averages[key] = state['stats_totals'][key] / count if count else 0.0

        style = self._classify_style_from_averages(averages)
        finish_rate = state['finish_count'] / state['wins'] if state['wins'] else 0.0
        activity_rate = self._compute_activity_rate(state['date_history'], current_date)

        total_fight_time = state.get('total_fight_time', 0.0)
        total_fight_minutes = total_fight_time / 60 if total_fight_time else 0.0
        average_fight_time = (total_fight_time / state['total_fights']) / 60 if state['total_fights'] else 0.0

        def _pace(total_value, minutes):
            return total_value / minutes if minutes and minutes > 0 else 0.0

        striking_pace = _pace(state['stats_totals']['sig_str_landed'], total_fight_minutes)
        takedown_pace = _pace(state['stats_totals']['td_landed'], total_fight_minutes)
        submission_pace = _pace(state['stats_totals']['sub_att'], total_fight_minutes)
        control_share = (state['stats_totals']['ctrl_time'] / total_fight_time) if total_fight_time > 0 else 0.0

        recent_fight_times = list(state['recent_fight_times'])
        recent_time_total = sum(recent_fight_times)
        recent_minutes = recent_time_total / 60 if recent_time_total else 0.0
        recent_average_fight_time = (recent_time_total / len(recent_fight_times)) / 60 if recent_fight_times else 0.0

        recent_striking_pace = _pace(sum(state['recent_sig_str_landed']), recent_minutes)
        recent_takedown_pace = _pace(sum(state['recent_td_landed']), recent_minutes)
        recent_submission_pace = _pace(sum(state['recent_sub_att']), recent_minutes)
        recent_control_share = (sum(state['recent_ctrl_time']) / recent_time_total) if recent_time_total > 0 else 0.0
        recent_finish_rate = (sum(state['recent_finishes']) / len(state['recent_finishes'])) if state['recent_finishes'] else 0.0

        experience_score = math.log1p(state['total_fights']) if state['total_fights'] > 0 else 0.0

        return {
            'total_fights': state['total_fights'],
            'wins': state['wins'],
            'win_rate': win_rate,
            'recent_form': {
                'weighted_win_rate': weighted_win_rate,
                'performance_trend': performance_trend,
                'recent_win_rate': recent_win_rate,
            },
            'style': style,
            'avg_stats': averages,
            'activity_rate': activity_rate,
            'days_since_last': days_since_last,
            'finish_rate': finish_rate,
            'last_fight_date': last_fight_date,
            'recent_finish_rate': recent_finish_rate,
            'average_fight_time': average_fight_time,
            'recent_average_fight_time': recent_average_fight_time,
            'striking_pace': striking_pace,
            'recent_striking_pace': recent_striking_pace,
            'control_per_minute': control_share,
            'recent_control_per_minute': recent_control_share,
            'takedown_pace': takedown_pace,
            'recent_takedown_pace': recent_takedown_pace,
            'submission_pace': submission_pace,
            'recent_submission_pace': recent_submission_pace,
            'experience_score': experience_score,
            'last_result': state.get('last_result', 0),
        }

    def _get_fighter_snapshot(self, fighter_name, current_date):
        state = self.fighter_history.get(fighter_name)
        if state is None:
            return self._default_snapshot()
        return self._snapshot_from_state(state, current_date)

    def _build_profiles_from_history(self):
        profiles = {}
        for fighter, state in self.fighter_history.items():
            if state['total_fights'] == 0:
                continue
            reference_date = state['last_fight_date'] or pd.Timestamp(datetime.now())
            snapshot = self._snapshot_from_state(state, reference_date)
            profiles[fighter] = {
                'total_fights': snapshot['total_fights'],
                'win_rate': snapshot['win_rate'],
                'style': snapshot['style'],
                'recent_form': snapshot['recent_form'],
                'career_avg': snapshot['avg_stats'],
                'finish_rate': snapshot['finish_rate'],
                'activity_rate': snapshot['activity_rate'],
                'last_fight_date': snapshot['last_fight_date'],
                'recent_finish_rate': snapshot['recent_finish_rate'],
                'average_fight_time': snapshot['average_fight_time'],
                'recent_average_fight_time': snapshot['recent_average_fight_time'],
                'striking_pace': snapshot['striking_pace'],
                'recent_striking_pace': snapshot['recent_striking_pace'],
                'control_per_minute': snapshot['control_per_minute'],
                'recent_control_per_minute': snapshot['recent_control_per_minute'],
                'takedown_pace': snapshot['takedown_pace'],
                'recent_takedown_pace': snapshot['recent_takedown_pace'],
                'submission_pace': snapshot['submission_pace'],
                'recent_submission_pace': snapshot['recent_submission_pace'],
                'experience_score': snapshot['experience_score'],
                'last_result': snapshot['last_result'],
            }
        self.fighter_profiles = profiles

    def _build_feature_row(self, fighter_a, fighter_b, snapshot_a, snapshot_b, fight_date, title_fight, weight_class):
        style_multiplier = self.style_matchup_matrix.get((snapshot_a['style'], snapshot_b['style']), 1.0)
        experience_ratio_adv = ((snapshot_a['total_fights'] + 1) / (snapshot_b['total_fights'] + 1)) - 1
        recent_win_rate_adv = snapshot_a['recent_form']['recent_win_rate'] - snapshot_b['recent_form']['recent_win_rate']
        return {
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'title_fight': 1 if title_fight else 0,
            'weight_class': weight_class or 'Unknown',
            'experience_advantage': snapshot_a['total_fights'] - snapshot_b['total_fights'],
            'experience_gap': abs(snapshot_a['total_fights'] - snapshot_b['total_fights']),
            'experience_ratio_advantage': experience_ratio_adv,
            'experience_score_advantage': snapshot_a['experience_score'] - snapshot_b['experience_score'],
            'win_rate_advantage': snapshot_a['recent_form']['weighted_win_rate'] - snapshot_b['recent_form']['weighted_win_rate'],
            'recent_win_rate_advantage': recent_win_rate_adv,
            'performance_trend_diff': snapshot_a['recent_form']['performance_trend'] - snapshot_b['recent_form']['performance_trend'],
            'style_matchup_advantage': style_multiplier - 1.0,
            'style_a': snapshot_a['style'],
            'style_b': snapshot_b['style'],
            'activity_advantage': snapshot_a['activity_rate'] - snapshot_b['activity_rate'],
            'recency_advantage': snapshot_b['days_since_last'] - snapshot_a['days_since_last'],
            'finish_rate_advantage': snapshot_a['finish_rate'] - snapshot_b['finish_rate'],
            'recent_finish_rate_advantage': snapshot_a['recent_finish_rate'] - snapshot_b['recent_finish_rate'],
            'avg_fight_time_advantage': snapshot_a['average_fight_time'] - snapshot_b['average_fight_time'],
            'recent_avg_fight_time_advantage': snapshot_a['recent_average_fight_time'] - snapshot_b['recent_average_fight_time'],
            'striking_pace_advantage': snapshot_a['striking_pace'] - snapshot_b['striking_pace'],
            'recent_striking_pace_advantage': snapshot_a['recent_striking_pace'] - snapshot_b['recent_striking_pace'],
            'control_time_share_advantage': snapshot_a['control_per_minute'] - snapshot_b['control_per_minute'],
            'recent_control_time_share_advantage': snapshot_a['recent_control_per_minute'] - snapshot_b['recent_control_per_minute'],
            'takedown_pace_advantage': snapshot_a['takedown_pace'] - snapshot_b['takedown_pace'],
            'recent_takedown_pace_advantage': snapshot_a['recent_takedown_pace'] - snapshot_b['recent_takedown_pace'],
            'submission_pace_advantage': snapshot_a['submission_pace'] - snapshot_b['submission_pace'],
            'recent_submission_pace_advantage': snapshot_a['recent_submission_pace'] - snapshot_b['recent_submission_pace'],
            'striking_volume_advantage': snapshot_a['avg_stats']['sig_str_landed'] - snapshot_b['avg_stats']['sig_str_landed'],
            'striking_accuracy_advantage': snapshot_a['avg_stats']['sig_str_acc'] - snapshot_b['avg_stats']['sig_str_acc'],
            'grappling_advantage': snapshot_a['avg_stats']['td_landed'] - snapshot_b['avg_stats']['td_landed'],
            'submission_threat_advantage': snapshot_a['avg_stats']['sub_att'] - snapshot_b['avg_stats']['sub_att'],
            'control_advantage': snapshot_a['avg_stats']['ctrl_time'] - snapshot_b['avg_stats']['ctrl_time'],
            'last_fight_win_advantage': snapshot_a['last_result'] - snapshot_b['last_result'],
            'date': fight_date,
        }

    def _process_fights(self, capture_features=False):
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        self._reset_state()
        enhanced_rows = []

        for _, fight in self.df.iterrows():
            fighter_a = fight.get('r_name')
            fighter_b = fight.get('b_name')
            winner = fight.get('winner')

            if pd.isna(fighter_a) or pd.isna(fighter_b) or pd.isna(winner):
                continue

            fight_date = pd.to_datetime(fight.get('date'), errors='coerce')
            if pd.isna(fight_date):
                continue

            title_fight = bool(fight.get('title_fight', 0))
            weight_class = fight.get('division', 'Unknown')

            snapshot_a = self._get_fighter_snapshot(fighter_a, fight_date)
            snapshot_b = self._get_fighter_snapshot(fighter_b, fight_date)

            if capture_features:
                row = self._build_feature_row(
                    fighter_a,
                    fighter_b,
                    snapshot_a,
                    snapshot_b,
                    fight_date,
                    title_fight,
                    weight_class,
                )
                row['winner'] = winner
                row['target'] = 1 if winner == fighter_a else 0
                enhanced_rows.append(row)

            self._update_fighter_state(fighter_a, fight, 'red', fight_date, winner == fighter_a, winner)
            self._update_fighter_state(fighter_b, fight, 'blue', fight_date, winner == fighter_b, winner)

        self._build_profiles_from_history()
        return pd.DataFrame(enhanced_rows) if capture_features else pd.DataFrame()

    def create_enhanced_training_data(self):
        """Create training dataset with enhanced features."""
        print("🔄 Creating enhanced training dataset...")
        enhanced_df = self._process_fights(capture_features=True)
        print(f"✅ Created enhanced dataset with {len(enhanced_df)} fights")
        feature_columns = [col for col in enhanced_df.columns if col not in {'fighter_a', 'fighter_b', 'winner', 'target'}]
        print(f"📊 Features: {len(feature_columns)}")
        return enhanced_df

    def _ensure_history_initialized(self):
        if self.fighter_history or self.df is None:
            return
        self._process_fights(capture_features=False)

    def _create_fighter_profiles(self):
        """Populate fighter profiles without returning a dataset."""
        print("🔄 Building fighter profiles...")
        self._process_fights(capture_features=False)
        print(f"✅ Created profiles for {len(self.fighter_profiles)} fighters")
        return self.fighter_profiles

    def extract_enhanced_features(self, fighter_a, fighter_b, fight_date=None,
                                 title_fight=False, weight_class=None):
        """Extract enhanced features for a fighter matchup."""
        if fight_date is None:
            fight_date = datetime.now()

        fight_date = pd.to_datetime(fight_date, errors='coerce')
        if pd.isna(fight_date):
            fight_date = pd.Timestamp(datetime.now())

        self._ensure_history_initialized()

        snapshot_a = self._get_fighter_snapshot(fighter_a, fight_date)
        snapshot_b = self._get_fighter_snapshot(fighter_b, fight_date)

        return self._build_feature_row(
            fighter_a,
            fighter_b,
            snapshot_a,
            snapshot_b,
            fight_date,
            title_fight,
            weight_class or 'Unknown',
        )

    def create_enhanced_training_data_with_profiles(self):
        """Utility for tests needing both dataset and profiles."""
        dataset = self.create_enhanced_training_data()
        return dataset, self.fighter_profiles


def main():
    """Test the enhanced feature engineering."""
    print("🥊 ENHANCED FEATURE ENGINEERING TEST")
    print("=" * 50)

    engineer = EnhancedFeatureEngineer()

    if not engineer.load_and_prepare_data():
        return

    enhanced_df = engineer.create_enhanced_training_data()

    if enhanced_df.empty:
        print("❌ No enhanced features were generated")
        return

    print("\n📊 Sample Enhanced Features:")
    sample_fight = enhanced_df.iloc[0]
    print(f"Fight: {sample_fight['fighter_a']} vs {sample_fight['fighter_b']}")
    print(f"Winner: {sample_fight['winner']}")
    print("\nEnhanced Features:")
    for feature, value in sample_fight.items():
        if feature not in ['fighter_a', 'fighter_b', 'winner', 'target', 'weight_class']:
            if isinstance(value, (int, float)):
                print(f"  {feature}: {value:.3f}")
            else:
                print(f"  {feature}: {value}")

    output_file = 'enhanced_ufc_features.csv'
    enhanced_df.to_csv(output_file, index=False)
    print(f"\n✅ Enhanced features saved to {output_file}")
    print("🎯 Ready for enhanced random forest training!")


if __name__ == "__main__":
    main()
