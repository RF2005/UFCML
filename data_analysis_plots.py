#!/usr/bin/env python3
"""
UFC Data Analysis and Visualization
===================================

Analyzes UFC fight data to determine noise levels and relationships.
Helps determine if linear regression would be more appropriate than
random forest by visualizing data patterns.

Key Analysis:
- Feature vs outcome scatter plots
- Noise level assessment
- Linear vs non-linear relationship detection
- Data variance and distribution analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import sys
sys.path.append('/Users/ralphfrancolini/UFCML')

from src.core.advanced_ml_models import load_enhanced_ufc_data
from enhanced_feature_engineering import EnhancedFeatureEngineer

class UFCDataAnalyzer:
    """Comprehensive analysis of UFC data patterns and noise levels."""

    def __init__(self):
        self.df = None
        self.enhanced_df = None
        self.engineer = None

    def load_and_prepare_data(self):
        """Load both raw and enhanced UFC data."""
        print("📊 Loading UFC data for analysis...")

        # Load raw data
        self.df = load_enhanced_ufc_data()
        if self.df is None:
            print("❌ Failed to load UFC data")
            return False

        # Create enhanced features
        self.engineer = EnhancedFeatureEngineer()
        if not self.engineer.load_and_prepare_data():
            return False

        self.enhanced_df = self.engineer.create_enhanced_training_data()

        print(f"✅ Loaded {len(self.df)} raw fights")
        print(f"✅ Created {len(self.enhanced_df)} enhanced feature fights")

        return True

    def analyze_data_noise(self):
        """Analyze noise levels in the UFC data."""
        print("\n🔍 ANALYZING DATA NOISE LEVELS")
        print("=" * 50)

        # Basic statistics
        print(f"Raw Data Shape: {self.df.shape}")
        print(f"Enhanced Data Shape: {self.enhanced_df.shape}")

        # Missing data analysis
        print(f"\n📊 Missing Data Analysis:")
        missing_raw = self.df.isnull().sum().sum()
        missing_enhanced = self.enhanced_df.isnull().sum().sum()
        print(f"Raw data missing values: {missing_raw}")
        print(f"Enhanced data missing values: {missing_enhanced}")

        # Outcome distribution
        if 'winner' in self.df.columns:
            red_wins = (self.df['winner'] == self.df['r_name']).sum()
            total_fights = len(self.df)
            red_win_rate = red_wins / total_fights
            print(f"\n🎯 Fight Outcome Distribution:")
            print(f"Red corner wins: {red_win_rate:.1%}")
            print(f"Blue corner wins: {1-red_win_rate:.1%}")

        # Feature variance analysis
        numeric_cols = self.enhanced_df.select_dtypes(include=[np.number]).columns
        numeric_data = self.enhanced_df[numeric_cols].drop(['target'], errors='ignore')

        print(f"\n📈 Feature Variance Analysis:")
        print(f"Number of numeric features: {len(numeric_data.columns)}")

        # Calculate coefficient of variation (CV) for each feature
        feature_cv = {}
        for col in numeric_data.columns:
            if numeric_data[col].std() > 0:
                cv = numeric_data[col].std() / abs(numeric_data[col].mean()) if numeric_data[col].mean() != 0 else np.inf
                feature_cv[col] = cv

        # Sort by noise level (high CV = high noise)
        sorted_cv = sorted(feature_cv.items(), key=lambda x: x[1], reverse=True)

        print(f"🔥 Highest noise features (top 5):")
        for feature, cv in sorted_cv[:5]:
            print(f"  {feature}: CV = {cv:.2f}")

        print(f"✅ Lowest noise features (top 5):")
        for feature, cv in sorted_cv[-5:]:
            print(f"  {feature}: CV = {cv:.2f}")

        return feature_cv

    def plot_feature_relationships(self):
        """Plot key feature vs outcome relationships."""
        print("\n📈 PLOTTING FEATURE RELATIONSHIPS")
        print("=" * 50)

        # Set up plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('UFC Fight Data: Feature vs Outcome Analysis', fontsize=16, fontweight='bold')

        # Key features to analyze
        key_features = [
            'win_rate_advantage',
            'experience_advantage',
            'style_matchup_advantage',
            'striking_volume_advantage',
            'grappling_advantage',
            'finish_rate_advantage'
        ]

        for idx, feature in enumerate(key_features):
            if feature not in self.enhanced_df.columns:
                continue

            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            # Create scatter plot
            x = self.enhanced_df[feature]
            y = self.enhanced_df['target']

            # Add jitter to binary target for better visualization
            y_jittered = y + np.random.normal(0, 0.02, len(y))

            ax.scatter(x, y_jittered, alpha=0.5, s=20)

            # Add trend line
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)

                # Calculate R-squared
                correlation = np.corrcoef(x, y)[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

            except:
                pass

            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Win Probability')
            ax.set_title(f'{feature.replace("_", " ").title()} vs Fight Outcome')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ufc_feature_relationships.png', dpi=300, bbox_inches='tight')
        print("✅ Feature relationship plots saved as 'ufc_feature_relationships.png'")

    def plot_noise_analysis(self):
        """Create noise analysis visualizations."""
        print("\n📊 CREATING NOISE ANALYSIS PLOTS")
        print("=" * 50)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('UFC Data Noise Analysis', fontsize=16, fontweight='bold')

        # 1. Feature variance distribution
        numeric_cols = self.enhanced_df.select_dtypes(include=[np.number]).columns
        numeric_data = self.enhanced_df[numeric_cols].drop(['target'], errors='ignore')

        variances = [numeric_data[col].var() for col in numeric_data.columns]
        axes[0, 0].hist(variances, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Distribution of Feature Variances')
        axes[0, 0].set_xlabel('Variance')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Correlation heatmap (subset of features)
        key_features = [col for col in ['win_rate_advantage', 'experience_advantage',
                                       'style_matchup_advantage', 'striking_volume_advantage',
                                       'grappling_advantage', 'target'] if col in self.enhanced_df.columns]

        if len(key_features) > 1:
            correlation_matrix = self.enhanced_df[key_features].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes[0, 1], square=True)
            axes[0, 1].set_title('Feature Correlation Matrix')

        # 3. Win rate vs features scatter
        if 'win_rate_advantage' in self.enhanced_df.columns:
            x = self.enhanced_df['win_rate_advantage']
            y = self.enhanced_df['target']
            axes[1, 0].scatter(x, y, alpha=0.5)
            axes[1, 0].set_xlabel('Win Rate Advantage')
            axes[1, 0].set_ylabel('Actual Outcome')
            axes[1, 0].set_title('Win Rate Advantage vs Outcome')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Feature importance comparison
        if len(numeric_data.columns) > 0:
            # Calculate simple correlation with target
            correlations = []
            feature_names = []
            for col in numeric_data.columns:
                if col != 'target':
                    corr = abs(np.corrcoef(numeric_data[col], self.enhanced_df['target'])[0, 1])
                    if not np.isnan(corr):
                        correlations.append(corr)
                        feature_names.append(col)

            # Plot top correlations
            if correlations:
                sorted_features = sorted(zip(feature_names, correlations),
                                       key=lambda x: x[1], reverse=True)[:10]
                names, corrs = zip(*sorted_features)

                axes[1, 1].barh(range(len(names)), corrs)
                axes[1, 1].set_yticks(range(len(names)))
                axes[1, 1].set_yticklabels([name.replace('_', ' ') for name in names])
                axes[1, 1].set_xlabel('Absolute Correlation with Outcome')
                axes[1, 1].set_title('Top 10 Most Predictive Features')

        plt.tight_layout()
        plt.savefig('ufc_noise_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Noise analysis plots saved as 'ufc_noise_analysis.png'")

    def compare_linear_vs_nonlinear(self):
        """Compare linear regression vs random forest performance."""
        print("\n⚖️  COMPARING LINEAR VS NON-LINEAR MODELS")
        print("=" * 50)

        # Prepare data
        feature_cols = [col for col in self.enhanced_df.columns
                       if col not in ['fighter_a', 'fighter_b', 'winner', 'target', 'weight_class', 'style_a', 'style_b']]

        X = self.enhanced_df[feature_cols].fillna(0)
        y = self.enhanced_df['target']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features for linear regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"📊 Training on {len(X_train)} fights, testing on {len(X_test)} fights")
        print(f"🔢 Using {len(feature_cols)} features")

        # Linear Regression
        print("\n🔧 Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_pred_binary = (lr_pred > 0.5).astype(int)
        lr_accuracy = accuracy_score(y_test, lr_pred_binary)
        lr_r2 = r2_score(y_test, lr_pred)

        # Random Forest
        print("🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]

        # Results
        print(f"\n📊 MODEL COMPARISON RESULTS:")
        print(f"Linear Regression:")
        print(f"  • Accuracy: {lr_accuracy:.1%}")
        print(f"  • R² Score: {lr_r2:.3f}")
        print(f"  • Mean prediction: {lr_pred.mean():.3f}")

        print(f"\nRandom Forest:")
        print(f"  • Accuracy: {rf_accuracy:.1%}")
        print(f"  • Mean probability: {rf_prob.mean():.3f}")

        # Feature importance (Random Forest)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🏆 Top 5 Most Important Features (Random Forest):")
        for i, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")

        # Linearity assessment
        print(f"\n🔍 LINEARITY ASSESSMENT:")
        accuracy_diff = rf_accuracy - lr_accuracy
        print(f"Random Forest advantage: +{accuracy_diff:.1%}")

        if accuracy_diff > 0.05:
            print("✅ Random Forest significantly outperforms linear regression")
            print("   → Data has strong non-linear patterns")
            print("   → Tree-based models are more appropriate")
        elif accuracy_diff > 0.02:
            print("⚡ Random Forest moderately outperforms linear regression")
            print("   → Some non-linear patterns present")
            print("   → Tree-based models recommended")
        else:
            print("⚠️  Similar performance between models")
            print("   → Data may be mostly linear")
            print("   → Either approach could work")

        # Noise assessment
        if lr_r2 < 0.3:
            print(f"\n📊 NOISE ASSESSMENT:")
            print(f"R² Score: {lr_r2:.3f} - High noise detected")
            print("   → Data is quite noisy/unpredictable")
            print("   → Random Forest better at handling noise")
        elif lr_r2 < 0.5:
            print(f"\n📊 NOISE ASSESSMENT:")
            print(f"R² Score: {lr_r2:.3f} - Moderate noise")
            print("   → Some predictable patterns exist")
        else:
            print(f"\n📊 NOISE ASSESSMENT:")
            print(f"R² Score: {lr_r2:.3f} - Low noise")
            print("   → Data has clear predictable patterns")

        return {
            'linear_accuracy': lr_accuracy,
            'rf_accuracy': rf_accuracy,
            'r2_score': lr_r2,
            'feature_importance': feature_importance
        }

    def generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        print("\n📋 GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 70)

        # Run all analyses
        noise_analysis = self.analyze_data_noise()
        model_comparison = self.compare_linear_vs_nonlinear()

        # Create plots
        self.plot_feature_relationships()
        self.plot_noise_analysis()

        # Summary report
        print(f"\n" + "="*70)
        print("🎯 UFC DATA ANALYSIS SUMMARY REPORT")
        print("="*70)

        print(f"📊 Dataset Overview:")
        print(f"  • Total fights analyzed: {len(self.enhanced_df)}")
        print(f"  • Enhanced features created: {len([col for col in self.enhanced_df.columns if col not in ['fighter_a', 'fighter_b', 'winner', 'target']])}")

        print(f"\n🔍 Model Performance:")
        print(f"  • Linear Regression: {model_comparison['linear_accuracy']:.1%}")
        print(f"  • Random Forest: {model_comparison['rf_accuracy']:.1%}")
        print(f"  • RF Advantage: +{model_comparison['rf_accuracy'] - model_comparison['linear_accuracy']:.1%}")

        print(f"\n📈 Data Characteristics:")
        print(f"  • R² Score (linearity): {model_comparison['r2_score']:.3f}")

        if model_comparison['r2_score'] < 0.3:
            print("  • Assessment: High noise, complex non-linear relationships")
            print("  • Recommendation: ✅ Random Forest is optimal")
        elif model_comparison['rf_accuracy'] - model_comparison['linear_accuracy'] > 0.03:
            print("  • Assessment: Significant non-linear patterns")
            print("  • Recommendation: ✅ Random Forest is better")
        else:
            print("  • Assessment: Moderate linearity")
            print("  • Recommendation: ⚡ Either approach viable")

        print(f"\n🏆 Top Predictive Features:")
        for i, row in model_comparison['feature_importance'].head(3).iterrows():
            print(f"  • {row['feature'].replace('_', ' ').title()}: {row['importance']:.3f}")

        print(f"\n💡 Recommendations for Improvement:")
        print("  • Enhanced feature engineering ✅ (implemented)")
        print("  • Weighted recent performance ✅ (implemented)")
        print("  • Style matchup dynamics ✅ (implemented)")
        if model_comparison['rf_accuracy'] < 0.65:
            print("  • Consider XGBoost for further improvement")
            print("  • Add external data (betting odds, physical stats)")
            print("  • Implement neural network embeddings")

        print("="*70)

def main():
    """Run comprehensive UFC data analysis."""
    analyzer = UFCDataAnalyzer()

    if not analyzer.load_and_prepare_data():
        return

    # Run full analysis
    analyzer.generate_analysis_report()

    print(f"\n✅ Analysis complete!")
    print(f"📈 Visualizations saved:")
    print(f"  • ufc_feature_relationships.png")
    print(f"  • ufc_noise_analysis.png")

if __name__ == "__main__":
    main()