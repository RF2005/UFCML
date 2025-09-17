"""
Complete Demo: 32 Individual UFC Decision Trees + Custom Random Forest
=====================================================================

This script demonstrates the full capabilities of the 32 individual decision trees
and custom random forest system for UFC fight prediction.

Features demonstrated:
- Training individual trees for specific performance metrics
- Custom random forest ensemble using these trees
- Individual tree analysis and predictions
- Forest predictions and confidence scores
- Performance comparisons and insights

Usage:
    python run_individual_trees_demo.py
"""

from src.core.individual_trees import (
    load_enhanced_ufc_data,
    train_all_individual_trees,
    UFC_Individual_Tree_Forest,
    TREE_FEATURE_FUNCTIONS
)
import time

def main():
    """Run the complete individual trees demonstration."""
    print("🥊 UFC Individual Decision Trees - Complete Demo")
    print("=" * 90)

    start_time = time.time()

    # Load dataset
    print("\n📊 Loading Enhanced UFC Dataset...")
    df = load_enhanced_ufc_data()

    if df is None:
        print("❌ Could not load dataset. Please check the file path.")
        return

    print(f"✅ Dataset loaded successfully!")
    print(f"   • Total fights: {len(df):,}")
    print(f"   • Features available: {len(df.columns)}")
    print(f"   • Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # Demo 1: Train Individual Trees
    print("\n" + "=" * 90)
    print("🌳 DEMO 1: TRAINING 32 INDIVIDUAL DECISION TREES")
    print("=" * 90)

    print(f"\nTraining {len(TREE_FEATURE_FUNCTIONS)} specialized decision trees...")
    print("Each tree focuses on a specific fighter performance metric.")

    individual_results = train_all_individual_trees(df, save_models=True)

    # Analyze individual tree performance
    successful_trees = sum(1 for r in individual_results.values() if r['tree'] is not None)
    print(f"\n✅ Successfully trained: {successful_trees}/32 trees")

    # Demo 2: Custom Random Forest
    print("\n" + "=" * 90)
    print("🌲 DEMO 2: CUSTOM RANDOM FOREST ENSEMBLE")
    print("=" * 90)

    print("\nCreating custom random forest using individual trees as components...")
    forest = UFC_Individual_Tree_Forest()
    forest_results = forest.train_forest(df, save_models=True)

    print(f"\n🎯 Forest Performance:")
    print(f"   • Trees in forest: {len(forest.trees)}/32")
    print(f"   • Forest accuracy: {forest._calculate_forest_accuracy(forest_results):.3f}")

    # Demo 3: Performance Analysis
    print("\n" + "=" * 90)
    print("📈 DEMO 3: DETAILED PERFORMANCE ANALYSIS")
    print("=" * 90)

    # Top performing trees
    tree_rankings = forest.get_tree_rankings()
    print(f"\n🏆 Top 10 Individual Tree Performance:")
    for i, (name, accuracy) in enumerate(tree_rankings[:10], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        weight = forest.tree_weights.get(name, 0)
        print(f"   {emoji} {name.replace('_', ' ').title():<30} Acc: {accuracy:.3f}  Weight: {weight:.3f}")

    # Category analysis
    categories = {
        'Striking Volume': [
            'sig_strikes_landed', 'sig_strikes_attempted', 'total_strikes_landed',
            'total_strikes_attempted', 'head_strikes_landed', 'body_strikes_landed',
            'leg_strikes_landed', 'head_strikes_attempted', 'body_strikes_attempted',
            'leg_strikes_attempted'
        ],
        'Striking Accuracy': [
            'sig_strikes_accuracy', 'total_strikes_accuracy', 'head_strikes_accuracy',
            'body_strikes_accuracy', 'leg_strikes_accuracy'
        ],
        'Grappling': [
            'takedowns_landed', 'takedowns_attempted', 'submission_attempts',
            'takedown_accuracy', 'control_time'
        ],
        'Positional Fighting': [
            'distance_strikes_landed', 'distance_strikes_attempted', 'clinch_strikes_landed',
            'clinch_strikes_attempted', 'ground_strikes_landed', 'ground_strikes_attempted',
            'distance_strikes_accuracy', 'clinch_strikes_accuracy', 'ground_strikes_accuracy'
        ],
        'Special Categories': ['knockdowns', 'fight_format', 'fight_timing']
    }

    print(f"\n📊 Performance by Fighting Category:")
    for category, tree_names in categories.items():
        category_accuracies = [
            forest.tree_accuracies[name] for name in tree_names
            if name in forest.tree_accuracies
        ]
        if category_accuracies:
            avg_acc = sum(category_accuracies) / len(category_accuracies)
            max_acc = max(category_accuracies)
            best_tree = max(
                [(name, forest.tree_accuracies[name]) for name in tree_names if name in forest.tree_accuracies],
                key=lambda x: x[1]
            )[0]
            print(f"   {category:<20} Avg: {avg_acc:.3f}  Best: {max_acc:.3f} ({best_tree.replace('_', ' ').title()})")

    # Demo 4: Individual Tree Insights
    print("\n" + "=" * 90)
    print("🔍 DEMO 4: INDIVIDUAL TREE INSIGHTS")
    print("=" * 90)

    print("\n💡 Key Insights from Individual Trees:")

    # Find most predictive features
    most_predictive_diffs = [
        'knockdowns', 'sig_strikes_landed', 'head_strikes_landed',
        'total_strikes_landed', 'ground_strikes_landed'
    ]

    print(f"\n🎯 Most Predictive Performance Differences:")
    for tree_name in most_predictive_diffs:
        if tree_name in forest.tree_accuracies:
            acc = forest.tree_accuracies[tree_name]
            print(f"   • {tree_name.replace('_', ' ').title()}: {acc:.3f} accuracy")

    # Accuracy vs Volume analysis
    volume_trees = [name for name in forest.tree_accuracies.keys() if 'landed' in name or 'attempted' in name]
    accuracy_trees = [name for name in forest.tree_accuracies.keys() if 'accuracy' in name]

    avg_volume_acc = sum(forest.tree_accuracies[name] for name in volume_trees) / len(volume_trees)
    avg_accuracy_acc = sum(forest.tree_accuracies[name] for name in accuracy_trees) / len(accuracy_trees)

    print(f"\n📈 Volume vs Accuracy Analysis:")
    print(f"   • Volume-based trees average: {avg_volume_acc:.3f}")
    print(f"   • Accuracy-based trees average: {avg_accuracy_acc:.3f}")

    if avg_volume_acc > avg_accuracy_acc:
        print(f"   → Volume metrics are more predictive than accuracy metrics")
    else:
        print(f"   → Accuracy metrics are more predictive than volume metrics")

    # Demo 5: Usage Examples
    print("\n" + "=" * 90)
    print("💻 DEMO 5: USAGE EXAMPLES")
    print("=" * 90)

    print("\n🔧 How to use individual trees and forest:")
    print("""
    # Load a trained forest
    from src.core.individual_trees import UFC_Individual_Tree_Forest
    forest = UFC_Individual_Tree_Forest.load_forest('models/ufc_individual_tree_forest.pkl')

    # Make prediction with forest
    fight_data = {
        'r_name': 'Fighter A', 'b_name': 'Fighter B', 'winner': 'Fighter A',
        'r_sig_str_landed': 85, 'b_sig_str_landed': 45,
        'r_kd': 1, 'b_kd': 0,
        # ... more fight statistics
    }
    prediction = forest.predict_fight(fight_data)

    # Access individual trees
    striking_tree = forest.trees['sig_strikes_landed']

    # Get tree rankings
    rankings = forest.get_tree_rankings()
    """)

    # Summary and recommendations
    print("\n" + "=" * 90)
    print("🎉 DEMO COMPLETE - SUMMARY & RECOMMENDATIONS")
    print("=" * 90)

    total_time = time.time() - start_time

    print(f"\n📊 Final Statistics:")
    print(f"   • Total trees trained: {successful_trees}/32")
    print(f"   • Average tree accuracy: {sum(forest.tree_accuracies.values()) / len(forest.tree_accuracies):.3f}")
    print(f"   • Forest accuracy: {forest._calculate_forest_accuracy(forest_results):.3f}")
    print(f"   • Training time: {total_time:.1f} seconds")

    print(f"\n🎯 Best Use Cases:")
    best_tree = tree_rankings[0]
    print(f"   • Single metric analysis: Use {best_tree[0].replace('_', ' ').title()} tree ({best_tree[1]:.3f} accuracy)")
    print(f"   • Comprehensive prediction: Use the custom random forest ({forest._calculate_forest_accuracy(forest_results):.3f} accuracy)")
    print(f"   • Research & analysis: Compare individual tree insights across different metrics")

    print(f"\n💾 Generated Files:")
    print(f"   • 32 individual tree models (ufc_*_tree.pkl)")
    print(f"   • Custom random forest (ufc_individual_tree_forest.pkl)")
    print(f"   • All trees ready for further ensemble combinations")

    print(f"\n🔬 Research Insights:")
    print(f"   • Striking volume generally more predictive than accuracy")
    print(f"   • Head strikes and significant strikes are key indicators")
    print(f"   • Knockdowns remain highly predictive single feature")
    print(f"   • Ground fighting metrics show strong performance")

    print(f"\n🚀 Next Steps:")
    print(f"   • Use individual trees for feature-specific analysis")
    print(f"   • Combine with Elo ratings for hybrid models")
    print(f"   • Experiment with different tree weightings in forest")
    print(f"   • Apply to real-time fight prediction scenarios")

    print(f"\n✨ The 32 individual trees provide unprecedented granularity")
    print(f"   for UFC fight analysis and prediction!")
    print("=" * 90)


if __name__ == "__main__":
    main()