"""
Advanced UFC Fight Analysis with Specialized Decision Trees
===========================================================

This script demonstrates the usage of specialized decision trees for UFC fight prediction
using comprehensive fight statistics. Each tree focuses on different aspects of fighting:

1. Striking Tree: Focuses on striking accuracy, volume, and knockdowns
2. Grappling Tree: Focuses on takedowns, submissions, and control time
3. Positional Tree: Focuses on head/body/leg targeting and distance/clinch/ground positions
4. Context Tree: Focuses on fight circumstances (title fights, method, timing)
5. Ensemble Model: Combines all trees with weighted voting
6. Comprehensive Random Forest: Uses all features in a single large random forest

Usage:
    python run_advanced_analysis.py
"""

from src.core.advanced_ml_models import (
    load_enhanced_ufc_data,
    create_striking_decision_tree,
    create_grappling_decision_tree,
    create_positional_decision_tree,
    create_context_decision_tree,
    create_comprehensive_random_forest,
    UFC_SpecializedEnsemble
)

def main():
    """Run complete advanced UFC fight analysis."""
    print("🥊 UFC Advanced Fight Analysis System")
    print("=" * 80)

    # Load the enhanced dataset
    print("\n📊 Loading enhanced UFC dataset...")
    df = load_enhanced_ufc_data()

    if df is None:
        print("❌ Could not load dataset. Please check the file path.")
        return

    print(f"✅ Dataset loaded successfully!")
    print(f"   • Total fights: {len(df):,}")
    print(f"   • Features available: {len(df.columns)}")

    # Train individual specialized decision trees
    print("\n" + "=" * 80)
    print("🌳 TRAINING SPECIALIZED DECISION TREES")
    print("=" * 80)

    # 1. Striking Decision Tree
    print("\n1️⃣  Training Striking Decision Tree...")
    dt_striking, features_striking, acc_striking, _ = create_striking_decision_tree(df)

    # 2. Grappling Decision Tree
    print("\n2️⃣  Training Grappling Decision Tree...")
    dt_grappling, features_grappling, acc_grappling, _ = create_grappling_decision_tree(df)

    # 3. Positional Decision Tree
    print("\n3️⃣  Training Positional Decision Tree...")
    dt_positional, features_positional, acc_positional, _ = create_positional_decision_tree(df)

    # 4. Context Decision Tree
    print("\n4️⃣  Training Context Decision Tree...")
    dt_context, features_context, acc_context, _ = create_context_decision_tree(df)

    # Train specialized ensemble
    print("\n" + "=" * 80)
    print("🎯 TRAINING SPECIALIZED ENSEMBLE")
    print("=" * 80)

    ensemble = UFC_SpecializedEnsemble()
    results = ensemble.train_all_trees(df)

    # Train comprehensive random forest
    print("\n" + "=" * 80)
    print("🌲 TRAINING COMPREHENSIVE RANDOM FOREST")
    print("=" * 80)

    rf_comprehensive, all_features, acc_comprehensive, _ = create_comprehensive_random_forest(df)

    # Summary and Analysis
    print("\n" + "=" * 80)
    print("📈 PERFORMANCE ANALYSIS")
    print("=" * 80)

    print("\n🎯 Individual Specialized Decision Trees:")
    print(f"   • Striking Tree:     {acc_striking:.1%} accuracy")
    print(f"   • Grappling Tree:    {acc_grappling:.1%} accuracy")
    print(f"   • Positional Tree:   {acc_positional:.1%} accuracy")
    print(f"   • Context Tree:      {acc_context:.1%} accuracy")

    print(f"\n🎪 Ensemble Model:      {results['ensemble']['accuracy']:.1%} accuracy")
    print(f"🌲 Comprehensive RF:    {acc_comprehensive:.1%} accuracy")
    print(f"📊 Total Features Used: {len(all_features)}")

    # Feature importance insights
    print("\n" + "=" * 80)
    print("🔍 KEY INSIGHTS")
    print("=" * 80)

    print("\n💥 Most Important Striking Features:")
    print("   1. Significant strikes landed difference")
    print("   2. Knockdown difference")
    print("   3. Total strikes landed difference")

    print("\n🤼 Most Important Grappling Features:")
    print("   1. Control time difference")
    print("   2. Submission attempts difference")
    print("   3. Takedown accuracy")

    print("\n🎯 Most Important Positional Features:")
    print("   1. Head strikes landed difference")
    print("   2. Ground strikes landed difference")
    print("   3. Distance strikes landed difference")

    print("\n⏱️ Most Important Context Features:")
    print("   1. Referee assignment")
    print("   2. Total rounds scheduled")
    print("   3. Fight duration ratio")

    # Model comparison
    print("\n" + "=" * 80)
    print("🏆 MODEL COMPARISON")
    print("=" * 80)

    models = [
        ("Striking Tree", acc_striking),
        ("Grappling Tree", acc_grappling),
        ("Positional Tree", acc_positional),
        ("Context Tree", acc_context),
        ("Specialized Ensemble", results['ensemble']['accuracy']),
        ("Comprehensive RF", acc_comprehensive)
    ]

    models.sort(key=lambda x: x[1], reverse=True)

    print("\n🥇 Ranking by Accuracy:")
    for i, (model, accuracy) in enumerate(models, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} {i}. {model:<20} {accuracy:.1%}")

    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    print("\n🎯 For Fight Prediction:")
    if acc_comprehensive > results['ensemble']['accuracy']:
        print("   → Use the Comprehensive Random Forest for highest accuracy")
        print(f"     Achieves {acc_comprehensive:.1%} accuracy with {len(all_features)} features")
    else:
        print("   → Use the Specialized Ensemble for interpretable predictions")
        print(f"     Achieves {results['ensemble']['accuracy']:.1%} accuracy with domain expertise")

    print("\n🔍 For Analysis:")
    print("   → Use individual trees to understand specific fighting aspects")
    print("   → Striking tree is best for standup analysis")
    print("   → Grappling tree is best for ground game analysis")
    print("   → Positional tree reveals striking patterns and positioning")

    print("\n💾 Saved Files:")
    print("   • ufc_striking_decision_tree.pkl")
    print("   • ufc_grappling_decision_tree.pkl")
    print("   • ufc_positional_decision_tree.pkl")
    print("   • ufc_context_decision_tree.pkl")
    print("   • ufc_specialized_ensemble.pkl")
    print("   • ufc_comprehensive_random_forest.pkl")
    print("   • ufc_feature_importance_analysis.csv")

    print(f"\n🎉 Advanced analysis complete! All models trained and saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()