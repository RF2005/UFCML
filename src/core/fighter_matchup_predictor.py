#!/usr/bin/env python3
"""
UFC Fighter Matchup Predictor
Uses fighter career averages and profiles for hypothetical matchup predictions
"""

import sys
import numpy as np
import pandas as pd
sys.path.append('/Users/ralphfrancolini/UFCML')
sys.path.append('/Users/ralphfrancolini/UFCML/src/core')

from enhanced_random_forest import EnhancedUFCRandomForest
from enhanced_feature_engineering import EnhancedFeatureEngineer
import joblib
import pickle

class FighterProfile:
    """Represents a UFC fighter's career statistics and fighting style."""

    def __init__(self, name, weight_class, record, **stats):
        self.name = name
        self.weight_class = weight_class
        self.record = record  # (wins, losses, draws)
        self.stats = stats

    def get_career_averages(self):
        """Get career average statistics."""
        return {
            'sig_str_landed_per_min': self.stats.get('sig_str_landed_per_min', 3.5),
            'sig_str_acc': self.stats.get('sig_str_acc', 45.0),
            'sig_str_absorbed_per_min': self.stats.get('sig_str_absorbed_per_min', 3.2),
            'sig_str_def': self.stats.get('sig_str_def', 55.0),
            'td_avg': self.stats.get('td_avg', 1.2),
            'td_acc': self.stats.get('td_acc', 40.0),
            'td_def': self.stats.get('td_def', 65.0),
            'sub_avg': self.stats.get('sub_avg', 0.8),
            'kd_avg': self.stats.get('kd_avg', 0.3),
            'ctrl_time_per_fight': self.stats.get('ctrl_time_per_fight', 90),  # seconds
            'finish_rate': self.stats.get('finish_rate', 0.6),
            'experience': len([w for w in self.record if w]) if isinstance(self.record, list) else sum(self.record[:2])
        }

class FighterMatchupPredictor:
    """Predicts UFC fight outcomes using enhanced random forest and advanced feature engineering."""

    def __init__(self):
        self.enhanced_model = None
        self.feature_engineer = None
        self.load_models()
        self.create_fighter_database()

    def load_models(self):
        """Load the enhanced UFC prediction model."""
        try:
            # Load the enhanced random forest model
            self.enhanced_model = EnhancedUFCRandomForest.load_model('models/enhanced_ufc_random_forest.pkl')
            if self.enhanced_model:
                print("✅ Enhanced UFC Random Forest loaded")
                self.feature_engineer = self.enhanced_model.feature_engineer
            else:
                print("⚠️  Enhanced model not found, creating new instance")
                self.enhanced_model = EnhancedUFCRandomForest()
                self.feature_engineer = EnhancedFeatureEngineer()

        except Exception as e:
            print(f"❌ Error loading enhanced model: {e}")
            print("🔄 Creating new enhanced model instance...")
            self.enhanced_model = EnhancedUFCRandomForest()
            self.feature_engineer = EnhancedFeatureEngineer()

            # Load and prepare data for the feature engineer
            if self.feature_engineer.load_and_prepare_data():
                print("✅ Feature engineer initialized with UFC data")

    def calculate_fighter_career_stats(self, df):
        """Calculate career statistics for all fighters from the UFC dataset."""
        from advanced_ml_models import load_enhanced_ufc_data

        print("📊 Calculating career statistics for all fighters...")
        fighter_stats = {}

        # Get unique fighters
        all_red_fighters = df[['r_name']].dropna()['r_name'].unique()
        all_blue_fighters = df[['b_name']].dropna()['b_name'].unique()
        all_fighters = set(list(all_red_fighters) + list(all_blue_fighters))

        print(f"🔢 Processing {len(all_fighters)} fighters...")

        for fighter_name in all_fighters:
            # Get all fights for this fighter (both red and blue corner)
            red_fights = df[df['r_name'] == fighter_name].copy()
            blue_fights = df[df['b_name'] == fighter_name].copy()

            # Calculate stats for red corner fights
            red_stats = {
                'sig_str_landed': red_fights['r_sig_str_landed'].fillna(0).mean(),
                'sig_str_attempted': red_fights['r_sig_str_atmpted'].fillna(0).mean(),
                'sig_str_acc': red_fights['r_sig_str_acc'].fillna(0).mean(),
                'td_landed': red_fights['r_td_landed'].fillna(0).mean(),
                'td_attempted': red_fights['r_td_atmpted'].fillna(0).mean(),
                'td_acc': red_fights['r_td_acc'].fillna(0).mean(),
                'ctrl_time': red_fights['r_ctrl'].fillna(0).mean(),
                'kd': red_fights['r_kd'].fillna(0).mean(),
                'sub_attempts': red_fights['r_sub_att'].fillna(0).mean(),
                'fights': len(red_fights)
            }

            # Calculate stats for blue corner fights
            blue_stats = {
                'sig_str_landed': blue_fights['b_sig_str_landed'].fillna(0).mean(),
                'sig_str_attempted': blue_fights['b_sig_str_atmpted'].fillna(0).mean(),
                'sig_str_acc': blue_fights['b_sig_str_acc'].fillna(0).mean(),
                'td_landed': blue_fights['b_td_landed'].fillna(0).mean(),
                'td_attempted': blue_fights['b_td_atmpted'].fillna(0).mean(),
                'td_acc': blue_fights['b_td_acc'].fillna(0).mean(),
                'ctrl_time': blue_fights['b_ctrl'].fillna(0).mean(),
                'kd': blue_fights['b_kd'].fillna(0).mean(),
                'sub_attempts': blue_fights['b_sub_att'].fillna(0).mean(),
                'fights': len(blue_fights)
            }

            # Combine red and blue stats (weighted average)
            total_fights = red_stats['fights'] + blue_stats['fights']
            if total_fights > 0:
                combined_stats = {}
                for stat in ['sig_str_landed', 'sig_str_attempted', 'sig_str_acc',
                           'td_landed', 'td_attempted', 'td_acc', 'ctrl_time', 'kd', 'sub_attempts']:
                    red_contribution = (red_stats[stat] * red_stats['fights']) if red_stats['fights'] > 0 else 0
                    blue_contribution = (blue_stats[stat] * blue_stats['fights']) if blue_stats['fights'] > 0 else 0
                    combined_stats[stat] = (red_contribution + blue_contribution) / total_fights

                # Calculate wins
                red_wins = len(red_fights[red_fights['winner'] == fighter_name])
                blue_wins = len(blue_fights[blue_fights['winner'] == fighter_name])
                total_wins = red_wins + blue_wins
                total_losses = total_fights - total_wins

                # Get most common division
                divisions = list(red_fights['division'].fillna('Unknown')) + list(blue_fights['division'].fillna('Unknown'))
                division_counts = {}
                for div in divisions:
                    if div != 'Unknown' and pd.notna(div):
                        division_counts[div] = division_counts.get(div, 0) + 1

                most_common_division = max(division_counts.keys(), key=division_counts.get) if division_counts else 'Unknown'

                # Calculate finish rate
                red_finishes = len(red_fights[(red_fights['winner'] == fighter_name) &
                                             (red_fights['method'].fillna('').str.contains('TKO|KO|Submission', case=False, na=False))])
                blue_finishes = len(blue_fights[(blue_fights['winner'] == fighter_name) &
                                               (blue_fights['method'].fillna('').str.contains('TKO|KO|Submission', case=False, na=False))])
                total_finishes = red_finishes + blue_finishes
                finish_rate = total_finishes / total_wins if total_wins > 0 else 0

                # Store fighter data
                fighter_stats[fighter_name] = {
                    'record': (total_wins, total_losses, 0),  # (wins, losses, draws)
                    'division': most_common_division,
                    'total_fights': total_fights,
                    'sig_str_landed_per_fight': combined_stats['sig_str_landed'],
                    'sig_str_acc': combined_stats['sig_str_acc'],
                    'td_avg': combined_stats['td_landed'],
                    'td_acc': combined_stats['td_acc'],
                    'ctrl_time_per_fight': combined_stats['ctrl_time'],
                    'kd_avg': combined_stats['kd'],
                    'sub_avg': combined_stats['sub_attempts'],
                    'finish_rate': finish_rate
                }

        return fighter_stats

    def create_fighter_database(self):
        """Create a comprehensive database of UFC fighters from the actual dataset."""
        from advanced_ml_models import load_enhanced_ufc_data

        # Load the UFC dataset
        df = load_enhanced_ufc_data()
        if df is None:
            print("❌ Could not load UFC dataset, using limited fighter database")
            self.create_limited_fighter_database()
            return

        # Calculate career stats for all fighters
        fighter_career_stats = self.calculate_fighter_career_stats(df)

        # Convert weight class names (lowercase from dataset to proper case)
        weight_class_mapping = {
            'heavyweight': 'Heavyweight',
            'light heavyweight': 'Light Heavyweight',
            'middleweight': 'Middleweight',
            'welterweight': 'Welterweight',
            'lightweight': 'Lightweight',
            'featherweight': 'Featherweight',
            'bantamweight': 'Bantamweight',
            'flyweight': 'Flyweight',
            'strawweight': 'Strawweight',
            "women's strawweight": 'Strawweight',
            "women's flyweight": 'Flyweight',
            "women's bantamweight": 'Bantamweight',
            "women's featherweight": 'Featherweight',
            # Handle interim titles
            'interim heavyweight': 'Heavyweight',
            'interim light heavyweight': 'Light Heavyweight',
            'interim middleweight': 'Middleweight',
            'interim welterweight': 'Welterweight',
            'interim lightweight': 'Lightweight',
            'interim featherweight': 'Featherweight',
            'interim bantamweight': 'Bantamweight',
            'interim flyweight': 'Flyweight'
        }

        # Create fighter profiles
        self.fighters = {}
        min_fights = 3  # Only include fighters with at least 3 fights

        for fighter_name, stats in fighter_career_stats.items():
            if stats['total_fights'] >= min_fights:
                # Map division to weight class
                weight_class = weight_class_mapping.get(stats['division'], 'Middleweight')

                # Convert to fight time (assume 15 minute fights on average)
                estimated_fight_time = 15  # minutes
                sig_str_per_min = stats['sig_str_landed_per_fight'] / estimated_fight_time

                # Estimate defensive stats (simplified)
                sig_str_def = max(30, min(80, 65 - (stats['sig_str_acc'] - 45)))  # Rough inverse correlation
                td_def = max(40, min(95, 70 + (stats['td_acc'] - 45)))  # Rough positive correlation

                self.fighters[fighter_name] = FighterProfile(
                    fighter_name,
                    weight_class,
                    stats['record'],
                    sig_str_landed_per_min=max(1.0, sig_str_per_min),
                    sig_str_acc=max(20, min(80, stats['sig_str_acc'])),
                    sig_str_absorbed_per_min=max(1.0, sig_str_per_min * 0.8),  # Estimate
                    sig_str_def=sig_str_def,
                    td_avg=max(0, stats['td_avg']),
                    td_acc=max(10, min(80, stats['td_acc'])),
                    td_def=td_def,
                    sub_avg=max(0, stats['sub_avg']),
                    kd_avg=max(0, stats['kd_avg']),
                    ctrl_time_per_fight=max(0, stats['ctrl_time_per_fight']),
                    finish_rate=max(0, min(1, stats['finish_rate']))
                )

        print(f"✅ Created fighter database with {len(self.fighters)} fighters (min {min_fights} fights)")

        # Show breakdown by weight class
        weight_class_counts = {}
        for fighter in self.fighters.values():
            wc = fighter.weight_class
            weight_class_counts[wc] = weight_class_counts.get(wc, 0) + 1

        print("📊 Fighters by weight class:")
        for wc, count in sorted(weight_class_counts.items()):
            print(f"   {wc}: {count} fighters")

    def create_limited_fighter_database(self):
        """Fallback method with limited fighter database."""
        self.fighters = {
            # HEAVYWEIGHT (265 lbs)
            "Jon Jones": FighterProfile(
                "Jon Jones", "Heavyweight", (27, 1, 1),
                sig_str_landed_per_min=4.3, sig_str_acc=58.0, sig_str_absorbed_per_min=2.4,
                sig_str_def=64.0, td_avg=2.4, td_acc=42.0, td_def=85.0,
                sub_avg=0.4, kd_avg=0.19, ctrl_time_per_fight=180, finish_rate=0.74
            ),
            "Stipe Miocic": FighterProfile(
                "Stipe Miocic", "Heavyweight", (20, 4, 0),
                sig_str_landed_per_min=4.1, sig_str_acc=52.0, sig_str_absorbed_per_min=3.1,
                sig_str_def=58.0, td_avg=1.8, td_acc=38.0, td_def=68.0,
                sub_avg=0.1, kd_avg=0.42, ctrl_time_per_fight=45, finish_rate=0.58
            ),
            "Francis Ngannou": FighterProfile(
                "Francis Ngannou", "Heavyweight", (17, 3, 0),
                sig_str_landed_per_min=4.8, sig_str_acc=51.0, sig_str_absorbed_per_min=2.7,
                sig_str_def=54.0, td_avg=0.8, td_acc=40.0, td_def=60.0,
                sub_avg=0.0, kd_avg=0.85, ctrl_time_per_fight=20, finish_rate=0.88
            ),
            "Tom Aspinall": FighterProfile(
                "Tom Aspinall", "Heavyweight", (14, 3, 0),
                sig_str_landed_per_min=6.2, sig_str_acc=58.0, sig_str_absorbed_per_min=2.1,
                sig_str_def=62.0, td_avg=1.2, td_acc=45.0, td_def=75.0,
                sub_avg=0.4, kd_avg=0.65, ctrl_time_per_fight=35, finish_rate=0.92
            ),

            # LIGHT HEAVYWEIGHT (205 lbs)
            "Alex Pereira": FighterProfile(
                "Alex Pereira", "Light Heavyweight", (11, 2, 0),
                sig_str_landed_per_min=5.2, sig_str_acc=61.0, sig_str_absorbed_per_min=3.8,
                sig_str_def=54.0, td_avg=0.2, td_acc=20.0, td_def=50.0,
                sub_avg=0.0, kd_avg=0.85, ctrl_time_per_fight=15, finish_rate=0.85
            ),
            "Jamahal Hill": FighterProfile(
                "Jamahal Hill", "Light Heavyweight", (12, 1, 1),
                sig_str_landed_per_min=5.8, sig_str_acc=48.0, sig_str_absorbed_per_min=4.2,
                sig_str_def=52.0, td_avg=0.3, td_acc=25.0, td_def=65.0,
                sub_avg=0.1, kd_avg=0.72, ctrl_time_per_fight=10, finish_rate=0.75
            ),
            "Jiri Prochazka": FighterProfile(
                "Jiri Prochazka", "Light Heavyweight", (30, 4, 1),
                sig_str_landed_per_min=5.1, sig_str_acc=49.0, sig_str_absorbed_per_min=4.8,
                sig_str_def=48.0, td_avg=1.1, td_acc=35.0, td_def=60.0,
                sub_avg=0.3, kd_avg=0.88, ctrl_time_per_fight=25, finish_rate=0.82
            ),

            # MIDDLEWEIGHT (185 lbs)
            "Israel Adesanya": FighterProfile(
                "Israel Adesanya", "Middleweight", (24, 3, 0),
                sig_str_landed_per_min=4.8, sig_str_acc=56.0, sig_str_absorbed_per_min=2.8,
                sig_str_def=63.0, td_avg=0.4, td_acc=25.0, td_def=82.0,
                sub_avg=0.1, kd_avg=0.35, ctrl_time_per_fight=30, finish_rate=0.59
            ),
            "Dricus du Plessis": FighterProfile(
                "Dricus du Plessis", "Middleweight", (21, 2, 0),
                sig_str_landed_per_min=4.9, sig_str_acc=52.0, sig_str_absorbed_per_min=3.6,
                sig_str_def=56.0, td_avg=2.1, td_acc=42.0, td_def=70.0,
                sub_avg=0.2, kd_avg=0.48, ctrl_time_per_fight=85, finish_rate=0.76
            ),
            "Sean Strickland": FighterProfile(
                "Sean Strickland", "Middleweight", (28, 6, 0),
                sig_str_landed_per_min=4.2, sig_str_acc=45.0, sig_str_absorbed_per_min=3.1,
                sig_str_def=58.0, td_avg=1.8, td_acc=38.0, td_def=75.0,
                sub_avg=0.1, kd_avg=0.21, ctrl_time_per_fight=65, finish_rate=0.44
            ),

            # WELTERWEIGHT (170 lbs)
            "Leon Edwards": FighterProfile(
                "Leon Edwards", "Welterweight", (22, 3, 1),
                sig_str_landed_per_min=3.8, sig_str_acc=47.0, sig_str_absorbed_per_min=2.9,
                sig_str_def=62.0, td_avg=1.2, td_acc=35.0, td_def=80.0,
                sub_avg=0.2, kd_avg=0.28, ctrl_time_per_fight=55, finish_rate=0.45
            ),
            "Belal Muhammad": FighterProfile(
                "Belal Muhammad", "Welterweight", (23, 3, 1),
                sig_str_landed_per_min=3.6, sig_str_acc=44.0, sig_str_absorbed_per_min=2.7,
                sig_str_def=64.0, td_avg=2.8, td_acc=45.0, td_def=78.0,
                sub_avg=0.1, kd_avg=0.12, ctrl_time_per_fight=125, finish_rate=0.35
            ),
            "Kamaru Usman": FighterProfile(
                "Kamaru Usman", "Welterweight", (20, 4, 0),
                sig_str_landed_per_min=3.9, sig_str_acc=48.0, sig_str_absorbed_per_min=2.1,
                sig_str_def=68.0, td_avg=3.2, td_acc=52.0, td_def=85.0,
                sub_avg=0.1, kd_avg=0.25, ctrl_time_per_fight=195, finish_rate=0.58
            ),
            "Colby Covington": FighterProfile(
                "Colby Covington", "Welterweight", (17, 4, 0),
                sig_str_landed_per_min=4.1, sig_str_acc=43.0, sig_str_absorbed_per_min=2.8,
                sig_str_def=62.0, td_avg=4.1, td_acc=48.0, td_def=72.0,
                sub_avg=0.0, kd_avg=0.18, ctrl_time_per_fight=210, finish_rate=0.24
            ),

            # LIGHTWEIGHT (155 lbs)
            "Islam Makhachev": FighterProfile(
                "Islam Makhachev", "Lightweight", (26, 1, 0),
                sig_str_landed_per_min=3.2, sig_str_acc=52.0, sig_str_absorbed_per_min=1.9,
                sig_str_def=65.0, td_avg=4.1, td_acc=51.0, td_def=82.0,
                sub_avg=1.1, kd_avg=0.08, ctrl_time_per_fight=285, finish_rate=0.81
            ),
            "Khabib Nurmagomedov": FighterProfile(
                "Khabib Nurmagomedov", "Lightweight", (29, 0, 0),
                sig_str_landed_per_min=3.4, sig_str_acc=50.0, sig_str_absorbed_per_min=1.8,
                sig_str_def=63.0, td_avg=4.7, td_acc=47.0, td_def=75.0,
                sub_avg=1.2, kd_avg=0.1, ctrl_time_per_fight=420, finish_rate=0.86
            ),
            "Conor McGregor": FighterProfile(
                "Conor McGregor", "Lightweight", (22, 6, 0),
                sig_str_landed_per_min=5.1, sig_str_acc=46.0, sig_str_absorbed_per_min=4.2,
                sig_str_def=54.0, td_avg=0.4, td_acc=20.0, td_def=64.0,
                sub_avg=0.1, kd_avg=0.61, ctrl_time_per_fight=25, finish_rate=0.86
            ),
            "Charles Oliveira": FighterProfile(
                "Charles Oliveira", "Lightweight", (34, 10, 1),
                sig_str_landed_per_min=4.2, sig_str_acc=48.0, sig_str_absorbed_per_min=3.8,
                sig_str_def=56.0, td_avg=1.8, td_acc=42.0, td_def=68.0,
                sub_avg=1.8, kd_avg=0.45, ctrl_time_per_fight=95, finish_rate=0.78
            ),
            "Justin Gaethje": FighterProfile(
                "Justin Gaethje", "Lightweight", (25, 5, 0),
                sig_str_landed_per_min=6.1, sig_str_acc=51.0, sig_str_absorbed_per_min=5.8,
                sig_str_def=48.0, td_avg=1.2, td_acc=35.0, td_def=65.0,
                sub_avg=0.0, kd_avg=0.68, ctrl_time_per_fight=15, finish_rate=0.77
            ),

            # FEATHERWEIGHT (145 lbs)
            "Ilia Topuria": FighterProfile(
                "Ilia Topuria", "Featherweight", (15, 0, 0),
                sig_str_landed_per_min=4.8, sig_str_acc=53.0, sig_str_absorbed_per_min=2.4,
                sig_str_def=61.0, td_avg=2.1, td_acc=48.0, td_def=78.0,
                sub_avg=0.4, kd_avg=0.52, ctrl_time_per_fight=75, finish_rate=0.87
            ),
            "Alexander Volkanovski": FighterProfile(
                "Alexander Volkanovski", "Featherweight", (26, 3, 0),
                sig_str_landed_per_min=4.6, sig_str_acc=49.0, sig_str_absorbed_per_min=2.8,
                sig_str_def=59.0, td_avg=1.9, td_acc=44.0, td_def=82.0,
                sub_avg=0.1, kd_avg=0.31, ctrl_time_per_fight=85, finish_rate=0.54
            ),
            "Max Holloway": FighterProfile(
                "Max Holloway", "Featherweight", (26, 7, 0),
                sig_str_landed_per_min=6.8, sig_str_acc=49.0, sig_str_absorbed_per_min=4.1,
                sig_str_def=55.0, td_avg=0.8, td_acc=32.0, td_def=75.0,
                sub_avg=0.0, kd_avg=0.35, ctrl_time_per_fight=25, finish_rate=0.36
            ),

            # BANTAMWEIGHT (135 lbs)
            "Sean O'Malley": FighterProfile(
                "Sean O'Malley", "Bantamweight", (18, 1, 1),
                sig_str_landed_per_min=5.2, sig_str_acc=51.0, sig_str_absorbed_per_min=3.1,
                sig_str_def=58.0, td_avg=0.6, td_acc=28.0, td_def=72.0,
                sub_avg=0.1, kd_avg=0.68, ctrl_time_per_fight=20, finish_rate=0.72
            ),
            "Merab Dvalishvili": FighterProfile(
                "Merab Dvalishvili", "Bantamweight", (17, 4, 0),
                sig_str_landed_per_min=3.8, sig_str_acc=42.0, sig_str_absorbed_per_min=2.9,
                sig_str_def=64.0, td_avg=5.2, td_acc=48.0, td_def=78.0,
                sub_avg=0.1, kd_avg=0.08, ctrl_time_per_fight=285, finish_rate=0.29
            ),
            "Amanda Nunes": FighterProfile(
                "Amanda Nunes", "Bantamweight", (22, 5, 0),
                sig_str_landed_per_min=4.6, sig_str_acc=47.0, sig_str_absorbed_per_min=2.9,
                sig_str_def=58.0, td_avg=1.1, td_acc=35.0, td_def=70.0,
                sub_avg=0.5, kd_avg=0.44, ctrl_time_per_fight=80, finish_rate=0.74
            ),
            "Valentina Shevchenko": FighterProfile(
                "Valentina Shevchenko", "Bantamweight", (23, 4, 0),
                sig_str_landed_per_min=4.2, sig_str_acc=54.0, sig_str_absorbed_per_min=2.1,
                sig_str_def=66.0, td_avg=1.8, td_acc=41.0, td_def=88.0,
                sub_avg=0.7, kd_avg=0.15, ctrl_time_per_fight=150, finish_rate=0.59
            ),

            # FLYWEIGHT (125 lbs)
            "Alexandre Pantoja": FighterProfile(
                "Alexandre Pantoja", "Flyweight", (27, 5, 0),
                sig_str_landed_per_min=4.1, sig_str_acc=45.0, sig_str_absorbed_per_min=3.2,
                sig_str_def=57.0, td_avg=2.8, td_acc=44.0, td_def=72.0,
                sub_avg=0.8, kd_avg=0.28, ctrl_time_per_fight=145, finish_rate=0.63
            ),
            "Brandon Moreno": FighterProfile(
                "Brandon Moreno", "Flyweight", (21, 7, 2),
                sig_str_landed_per_min=4.3, sig_str_acc=46.0, sig_str_absorbed_per_min=3.6,
                sig_str_def=54.0, td_avg=1.9, td_acc=38.0, td_def=68.0,
                sub_avg=1.1, kd_avg=0.32, ctrl_time_per_fight=125, finish_rate=0.57
            ),

            # STRAWWEIGHT (115 lbs)
            "Zhang Weili": FighterProfile(
                "Zhang Weili", "Strawweight", (24, 3, 0),
                sig_str_landed_per_min=4.8, sig_str_acc=48.0, sig_str_absorbed_per_min=3.4,
                sig_str_def=56.0, td_avg=1.6, td_acc=42.0, td_def=75.0,
                sub_avg=0.2, kd_avg=0.41, ctrl_time_per_fight=65, finish_rate=0.59
            ),
            "Rose Namajunas": FighterProfile(
                "Rose Namajunas", "Strawweight", (12, 6, 0),
                sig_str_landed_per_min=3.9, sig_str_acc=49.0, sig_str_absorbed_per_min=3.1,
                sig_str_def=59.0, td_avg=1.2, td_acc=35.0, td_def=78.0,
                sub_avg=0.6, kd_avg=0.28, ctrl_time_per_fight=85, finish_rate=0.56
            )
        }

    def get_weight_class_multiplier(self, weight_class_a, weight_class_b):
        """Calculate multiplier based on weight class differences."""
        weight_classes = {
            'Strawweight': 115,
            'Flyweight': 125,
            'Bantamweight': 135,
            'Featherweight': 145,
            'Lightweight': 155,
            'Welterweight': 170,
            'Middleweight': 185,
            'Light Heavyweight': 205,
            'Heavyweight': 265
        }

        weight_a = weight_classes.get(weight_class_a, 185)  # Default to middleweight
        weight_b = weight_classes.get(weight_class_b, 185)

        weight_diff = abs(weight_a - weight_b)

        # If same weight class, no adjustment
        if weight_diff <= 10:
            return 1.0, 1.0, "Same weight class"

        # Calculate advantage based on weight difference
        heavier_fighter = weight_class_a if weight_a > weight_b else weight_class_b
        lighter_fighter = weight_class_b if weight_a > weight_b else weight_class_a

        # Weight advantage factors
        if weight_diff <= 20:  # Adjacent weight classes
            power_multiplier = 1.15  # 15% power advantage
            cardio_penalty = 0.95    # 5% cardio penalty
            warning = f"Cross-division fight: {heavier_fighter} vs {lighter_fighter} (+{weight_diff} lbs)"
        elif weight_diff <= 40:  # Two weight classes apart
            power_multiplier = 1.3   # 30% power advantage
            cardio_penalty = 0.9     # 10% cardio penalty
            warning = f"Major weight disadvantage: {heavier_fighter} vs {lighter_fighter} (+{weight_diff} lbs)"
        else:  # Three+ weight classes apart
            power_multiplier = 1.5   # 50% power advantage
            cardio_penalty = 0.85    # 15% cardio penalty
            warning = f"EXTREME weight mismatch: {heavier_fighter} vs {lighter_fighter} (+{weight_diff} lbs)"

        # Return multipliers for heavier and lighter fighter
        if weight_a > weight_b:
            return power_multiplier, 1/power_multiplier, warning
        else:
            return 1/power_multiplier, power_multiplier, warning

    def get_fighting_style_multiplier(self, fighter_profile):
        """Calculate fighting style multiplier for control time based on career ratios."""
        stats = fighter_profile.get_career_averages()

        # Calculate striking vs grappling tendencies
        striking_tendency = stats.get('sig_str_landed_per_min', 3.5) / 6.0  # Normalize to ~0.6 for average
        grappling_tendency = stats.get('td_avg', 1.2) / 3.0  # Normalize to ~0.4 for average
        control_tendency = stats.get('ctrl_time_per_fight', 90) / 180.0  # Normalize to ~0.5 for average

        # Calculate overall grappling orientation (0.0 = pure striker, 1.0 = pure grappler)
        grappling_score = (grappling_tendency * 0.4 + control_tendency * 0.6)
        grappling_score = max(0.1, min(0.9, grappling_score))  # Keep in reasonable range

        return grappling_score

    def simulate_fight_stats(self, fighter_a_profile, fighter_b_profile, fight_duration=15):
        """Simulate fight statistics based on fighter profiles with weight class considerations."""
        a_stats = fighter_a_profile.get_career_averages()
        b_stats = fighter_b_profile.get_career_averages()

        # Get weight class adjustments
        a_multiplier, b_multiplier, weight_warning = self.get_weight_class_multiplier(
            fighter_a_profile.weight_class, fighter_b_profile.weight_class
        )

        # Print weight class warning if significant difference
        if "Cross-division" in weight_warning or "EXTREME" in weight_warning or "Major" in weight_warning:
            print(f"⚠️  {weight_warning}")

        # Simulate strikes per minute over fight duration
        duration_min = fight_duration

        # Add some randomness to make fights more realistic
        import random

        # Apply weight class adjustments to base stats
        adjusted_a_stats = {
            'sig_str_landed_per_min': a_stats['sig_str_landed_per_min'] * a_multiplier,
            'sig_str_acc': a_stats['sig_str_acc'],
            'sig_str_def': a_stats['sig_str_def'],
            'td_avg': a_stats['td_avg'] * a_multiplier,
            'td_def': a_stats['td_def'],
            'kd_avg': a_stats['kd_avg'] * a_multiplier,
        }

        adjusted_b_stats = {
            'sig_str_landed_per_min': b_stats['sig_str_landed_per_min'] * b_multiplier,
            'sig_str_acc': b_stats['sig_str_acc'],
            'sig_str_def': b_stats['sig_str_def'],
            'td_avg': b_stats['td_avg'] * b_multiplier,
            'td_def': b_stats['td_def'],
            'kd_avg': b_stats['kd_avg'] * b_multiplier,
        }

        # Fighter A stats (accounting for opponent's defense + variation + weight)
        a_str_base = adjusted_a_stats['sig_str_landed_per_min'] * duration_min * (1 - adjusted_b_stats['sig_str_def']/100)
        a_sig_str_landed = max(0, round(a_str_base * random.uniform(0.8, 1.4)))
        a_sig_str_attempted = round(a_sig_str_landed / (adjusted_a_stats['sig_str_acc']/100 * random.uniform(0.9, 1.1))) if adjusted_a_stats['sig_str_acc'] > 0 else 0

        # Fighter B stats (accounting for opponent's defense + variation + weight)
        b_str_base = adjusted_b_stats['sig_str_landed_per_min'] * duration_min * (1 - adjusted_a_stats['sig_str_def']/100)
        b_sig_str_landed = max(0, round(b_str_base * random.uniform(0.8, 1.4)))
        b_sig_str_attempted = round(b_sig_str_landed / (adjusted_b_stats['sig_str_acc']/100 * random.uniform(0.9, 1.1))) if adjusted_b_stats['sig_str_acc'] > 0 else 0

        # Takedowns: use proper rounding and add variation + weight adjustments
        a_td_base = adjusted_a_stats['td_avg'] * (1 - adjusted_b_stats['td_def']/100)
        b_td_base = adjusted_b_stats['td_avg'] * (1 - adjusted_a_stats['td_def']/100)

        a_takedowns = max(0, round(a_td_base * random.uniform(0.7, 1.3)))
        b_takedowns = max(0, round(b_td_base * random.uniform(0.7, 1.3)))

        # Knockdowns: add variation + weight adjustments
        a_kd_base = adjusted_a_stats['kd_avg'] * random.uniform(0.5, 2.0)
        b_kd_base = adjusted_b_stats['kd_avg'] * random.uniform(0.5, 2.0)

        a_knockdowns = np.random.poisson(a_kd_base) if a_kd_base > 0 else 0
        b_knockdowns = np.random.poisson(b_kd_base) if b_kd_base > 0 else 0

        # Control time based on grappling ability with style considerations
        # Get fighting style scores (0.1 = striker, 0.9 = grappler)
        a_grappling_score = self.get_fighting_style_multiplier(fighter_a_profile)
        b_grappling_score = self.get_fighting_style_multiplier(fighter_b_profile)

        # Calculate base control time with reduced variation
        base_total_ctrl = min(duration_min * 60, a_stats['ctrl_time_per_fight'] + b_stats['ctrl_time_per_fight'])
        total_ctrl = base_total_ctrl * random.uniform(0.85, 1.15)  # Reduced variation

        # Apply style-based adjustments
        if a_stats['ctrl_time_per_fight'] + b_stats['ctrl_time_per_fight'] > 0:
            base_a_ratio = a_stats['ctrl_time_per_fight'] / (a_stats['ctrl_time_per_fight'] + b_stats['ctrl_time_per_fight'])
        else:
            base_a_ratio = 0.5

        # Adjust ratio based on fighting styles (grappler vs striker gets advantage)
        style_diff = a_grappling_score - b_grappling_score
        style_adjustment = style_diff * 0.2  # Max 20% adjustment
        adjusted_a_ratio = max(0.1, min(0.9, base_a_ratio + style_adjustment))

        # Calculate control times with reduced random variation
        a_ctrl_time = int(total_ctrl * adjusted_a_ratio * random.uniform(0.9, 1.1))
        b_ctrl_time = int(total_ctrl * (1 - adjusted_a_ratio) * random.uniform(0.9, 1.1))

        # Historical maximum checks - prevent extreme outliers
        a_career_max = a_stats['ctrl_time_per_fight'] * 1.5  # Allow 50% above career average
        b_career_max = b_stats['ctrl_time_per_fight'] * 1.5

        a_ctrl_time = min(a_ctrl_time, int(a_career_max))
        b_ctrl_time = min(b_ctrl_time, int(b_career_max))

        return {
            'r_name': fighter_a_profile.name,
            'b_name': fighter_b_profile.name,
            'winner': fighter_a_profile.name,  # Placeholder
            'r_sig_str_landed': int(a_sig_str_landed),
            'b_sig_str_landed': int(b_sig_str_landed),
            'r_sig_str_atmpted': int(a_sig_str_attempted),
            'b_sig_str_atmpted': int(b_sig_str_attempted),
            'r_sig_str_acc': a_stats['sig_str_acc'],
            'b_sig_str_acc': b_stats['sig_str_acc'],
            'r_td': int(a_takedowns),
            'b_td': int(b_takedowns),
            'r_kd': int(a_knockdowns),
            'b_kd': int(b_knockdowns),
            'r_ctrl_time_sec': a_ctrl_time,
            'b_ctrl_time_sec': b_ctrl_time,
            'total_rounds': 3,
            'method': 'Decision',
            'finish_round': 3,
            'match_time_sec': duration_min * 60,
            'referee': 'Herb Dean'
        }

    def predict_matchup(self, fighter_a_name, fighter_b_name, title_fight=False, weight_class=None):
        """Predict the outcome of a hypothetical matchup using the enhanced model."""
        if fighter_a_name not in self.fighters:
            print(f"❌ Fighter '{fighter_a_name}' not found in database")
            return None

        if fighter_b_name not in self.fighters:
            print(f"❌ Fighter '{fighter_b_name}' not found in database")
            return None

        if not self.enhanced_model or not self.enhanced_model.is_trained:
            print(f"❌ Enhanced model not loaded or trained")
            return None

        fighter_a = self.fighters[fighter_a_name]
        fighter_b = self.fighters[fighter_b_name]

        # Use enhanced feature engineering for prediction
        try:
            result = self.enhanced_model.predict_fight(
                fighter_a_name,
                fighter_b_name,
                title_fight=title_fight,
                weight_class=weight_class or fighter_a.weight_class
            )

            if result:
                predicted_winner = result['predicted_winner']
                confidence = result['confidence']
                fighter_a_prob = result['fighter_a_prob']
                fighter_b_prob = result['fighter_b_prob']

                # Add realism adjustment to confidence
                # Cap extremely high confidences for sports predictions
                if confidence > 0.85:
                    adjusted_confidence = 0.75 + (confidence - 0.85) * 0.33  # Max ~80%
                elif confidence < 0.55:
                    adjusted_confidence = max(0.52, confidence)  # Min ~52%
                else:
                    adjusted_confidence = confidence

                final_prediction = f"{predicted_winner} ({'Red Corner' if predicted_winner == fighter_a_name else 'Blue Corner'})"
                final_confidence = adjusted_confidence

                return {
                    'final_prediction': final_prediction,
                    'final_confidence': final_confidence,
                    'predicted_winner': predicted_winner,
                    'fighter_a_probability': fighter_a_prob,
                    'fighter_b_probability': fighter_b_prob,
                    'model_confidence': confidence,
                    'adjusted_confidence': adjusted_confidence,
                    'fighter_a_profile': fighter_a,
                    'fighter_b_profile': fighter_b,
                    'features_used': result.get('features_used', {})
                }
            else:
                print(f"❌ Enhanced model returned no prediction")
                return None

        except Exception as e:
            print(f"❌ Enhanced model prediction failed: {e}")
            return None

    def display_matchup_prediction(self, result):
        """Display matchup prediction results."""
        if not result:
            print("❌ Unable to generate prediction")
            return

        fighter_a = result['fighter_a_profile']
        fighter_b = result['fighter_b_profile']

        print("\n" + "🥊" * 50)
        print(f"FIGHTER MATCHUP PREDICTION")
        print("🥊" * 50)

        print(f"\n🔴 RED CORNER: {fighter_a.name}")
        print(f"   Record: {fighter_a.record[0]}-{fighter_a.record[1]}-{fighter_a.record[2] if len(fighter_a.record) > 2 else 0}")
        print(f"   Weight Class: {fighter_a.weight_class}")

        print(f"\n🔵 BLUE CORNER: {fighter_b.name}")
        print(f"   Record: {fighter_b.record[0]}-{fighter_b.record[1]}-{fighter_b.record[2] if len(fighter_b.record) > 2 else 0}")
        print(f"   Weight Class: {fighter_b.weight_class}")

        print(f"\n🏆 PREDICTION: {result['final_prediction']}")
        print(f"🎯 CONFIDENCE: {result['final_confidence']:.1%}")

        print(f"\n📊 ENHANCED MODEL BREAKDOWN:")
        print(f"   {fighter_a.name} probability: {result['fighter_a_probability']:.1%}")
        print(f"   {fighter_b.name} probability: {result['fighter_b_probability']:.1%}")
        print(f"   Raw model confidence: {result['model_confidence']:.1%}")
        print(f"   Adjusted confidence: {result['adjusted_confidence']:.1%}")

        # Show key features if available
        if result.get('features_used'):
            print(f"\n🔍 KEY FEATURES ANALYZED:")
            features = result['features_used']
            key_features = ['win_rate_advantage', 'recency_advantage', 'experience_advantage',
                          'style_matchup_advantage', 'striking_volume_advantage']
            for feature in key_features:
                if feature in features and isinstance(features[feature], (int, float)):
                    print(f"   {feature.replace('_', ' ').title()}: {features[feature]:.3f}")

        print("🥊" * 50)

def main():
    """Interactive fighter matchup predictor."""
    print("🥊 UFC FIGHTER MATCHUP PREDICTOR")
    print("=" * 50)
    print("Predict hypothetical fights using fighter profiles!")
    print("=" * 50)

    predictor = FighterMatchupPredictor()

    print(f"\n📋 Available Fighters:")
    fighters = list(predictor.fighters.keys())
    for i, fighter in enumerate(fighters, 1):
        record = predictor.fighters[fighter].record
        print(f"{i:2d}. {fighter} ({record[0]}-{record[1]}-{record[2] if len(record) > 2 else 0})")

    while True:
        print(f"\n🎯 Choose fighters for matchup prediction:")
        print("Enter fighter names or numbers from the list above")
        print("Type 'quit' to exit")

        try:
            fighter_a_input = input("\n🔴 Red Corner Fighter: ").strip()
            if fighter_a_input.lower() == 'quit':
                break

            fighter_b_input = input("🔵 Blue Corner Fighter: ").strip()
            if fighter_b_input.lower() == 'quit':
                break

            # Handle numeric input
            if fighter_a_input.isdigit():
                idx = int(fighter_a_input) - 1
                if 0 <= idx < len(fighters):
                    fighter_a_name = fighters[idx]
                else:
                    print("❌ Invalid fighter number")
                    continue
            else:
                fighter_a_name = fighter_a_input

            if fighter_b_input.isdigit():
                idx = int(fighter_b_input) - 1
                if 0 <= idx < len(fighters):
                    fighter_b_name = fighters[idx]
                else:
                    print("❌ Invalid fighter number")
                    continue
            else:
                fighter_b_name = fighter_b_input

            # Make prediction
            print(f"\n🔄 Analyzing matchup: {fighter_a_name} vs {fighter_b_name}...")
            result = predictor.predict_matchup(fighter_a_name, fighter_b_name)

            if result:
                predictor.display_matchup_prediction(result)

        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the Fighter Matchup Predictor!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()