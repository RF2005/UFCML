"""
Test Script for 32 Individual UFC Decision Trees
===============================================

This script tests all 32 individual decision trees and the custom random forest
ensemble built from them.
"""

from individual_trees import (
    load_enhanced_ufc_data,
    train_all_individual_trees,
    UFC_Individual_Tree_Forest,
    TREE_FEATURE_FUNCTIONS
)
import time

def main():
    """Test all individual trees and the custom forest."""
    print("🥊 Testing 32 Individual UFC Decision Trees")
    print("=" * 80)

    start_time = time.time()

    # Load dataset
    print("\n📊 Loading UFC dataset...")
    df = load_enhanced_ufc_data()

    if df is None:
        print("❌ Could not load dataset. Please check the file path.")
        return

    print(f"✅ Dataset loaded successfully!")
    print(f"   • Total fights: {len(df):,}")
    print(f"   • Features available: {len(df.columns)}")

    # Test individual trees
    print(f"\n🌳 Training and Testing {len(TREE_FEATURE_FUNCTIONS)} Individual Trees...")
    individual_results = train_all_individual_trees(df, save_models=True)

    # Test custom forest
    print(f"\n🌲 Testing Custom Random Forest...")
    forest = UFC_Individual_Tree_Forest()
    forest_results = forest.train_forest(df, save_models=True)

    # Analysis and Summary
    print(f"\n📈 COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # Individual tree performance analysis
    successful_trees = [name for name, result in individual_results.items()
                       if result['tree'] is not None]
    failed_trees = [name for name, result in individual_results.items()
                   if result['tree'] is None]

    print(f"\n✅ Successful Trees: {len(successful_trees)}/32")
    print(f"❌ Failed Trees: {len(failed_trees)}/32")

    if failed_trees:
        print(f"\nFailed trees:")
        for tree in failed_trees:
            print(f"   • {tree.replace('_', ' ').title()}")

    # Performance rankings
    accuracies = [(name, result['accuracy']) for name, result in individual_results.items()
                 if result['accuracy'] > 0]
    accuracies.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top 10 Individual Tree Performance:")
    for i, (name, acc) in enumerate(accuracies[:10], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        print(f"   {emoji} {name.replace('_', ' ').title():<30} {acc:.3f}")

    print(f"\n📊 Performance by Category:")

    # Categorize trees
    categories = {
        'Striking - Landed': ['knockdowns', 'sig_strikes_landed', 'sig_strikes_attempted',
                             'total_strikes_landed', 'total_strikes_attempted',
                             'head_strikes_landed', 'body_strikes_landed', 'leg_strikes_landed'],
        'Striking - Accuracy': ['sig_strikes_accuracy', 'total_strikes_accuracy',
                               'head_strikes_accuracy', 'body_strikes_accuracy', 'leg_strikes_accuracy',
                               'head_strikes_attempted', 'body_strikes_attempted', 'leg_strikes_attempted'],
        'Grappling': ['takedowns_landed', 'takedowns_attempted', 'submission_attempts',
                     'takedown_accuracy', 'control_time'],
        'Positional - Landed': ['distance_strikes_landed', 'distance_strikes_attempted',
                               'clinch_strikes_landed', 'clinch_strikes_attempted',
                               'ground_strikes_landed', 'ground_strikes_attempted'],
        'Positional - Accuracy': ['distance_strikes_accuracy', 'clinch_strikes_accuracy',
                                 'ground_strikes_accuracy'],
        'Fight Context': ['fight_format', 'fight_timing']
    }

    for category, tree_names in categories.items():
        category_accuracies = [acc for name, acc in accuracies if name in tree_names]
        if category_accuracies:
            avg_acc = sum(category_accuracies) / len(category_accuracies)
            max_acc = max(category_accuracies)
            min_acc = min(category_accuracies)
            print(f"   {category:<25} Avg: {avg_acc:.3f}  Max: {max_acc:.3f}  Min: {min_acc:.3f}")

    # Overall statistics
    all_accuracies = [acc for _, acc in accuracies]
    if all_accuracies:
        print(f"\n📈 Overall Statistics:")
        print(f"   Average Accuracy: {sum(all_accuracies) / len(all_accuracies):.3f}")
        print(f"   Highest Accuracy: {max(all_accuracies):.3f}")
        print(f"   Lowest Accuracy:  {min(all_accuracies):.3f}")
        print(f"   Standard Deviation: {(sum((x - sum(all_accuracies)/len(all_accuracies))**2 for x in all_accuracies) / len(all_accuracies))**0.5:.3f}")

    # Forest performance
    print(f"\n🌲 Custom Random Forest Performance:")
    forest_rankings = forest.get_tree_rankings()
    print(f"   Trees in Forest: {len(forest.trees)}/32")

    if hasattr(forest, '_calculate_forest_accuracy'):
        try:
            forest_acc = forest._calculate_forest_accuracy(forest_results)
            print(f"   Forest Accuracy: {forest_acc:.3f}")
        except:
            print(f"   Forest Accuracy: Could not calculate")

    print(f"\n🎯 Top 5 Trees by Weight in Forest:")
    for i, (name, accuracy) in enumerate(forest_rankings[:5], 1):
        weight = forest.tree_weights.get(name, 0)
        print(f"   {i}. {name.replace('_', ' ').title():<30} Accuracy: {accuracy:.3f}  Weight: {weight:.3f}")

    # Execution time
    total_time = time.time() - start_time
    print(f"\n⏱️  Total Execution Time: {total_time:.1f} seconds")

    # File outputs
    print(f"\n💾 Generated Files:")
    successful_count = len(successful_trees)
    print(f"   • {successful_count} individual tree files (ufc_*_tree.pkl)")
    print(f"   • ufc_individual_tree_forest.pkl")
    print(f"   • Test results and analysis")

    # Recommendations
    print(f"\n💡 Recommendations:")
    if all_accuracies:
        best_tree = accuracies[0]
        print(f"   • Best individual tree: {best_tree[0].replace('_', ' ').title()} ({best_tree[1]:.3f})")

        above_60 = len([acc for acc in all_accuracies if acc > 0.6])
        print(f"   • Trees above 60% accuracy: {above_60}/{len(all_accuracies)}")

        if above_60 >= len(all_accuracies) * 0.7:
            print(f"   • ✅ Most trees performing well - forest should be effective")
        elif above_60 >= len(all_accuracies) * 0.5:
            print(f"   • ⚠️  Mixed performance - consider weighting stronger trees more")
        else:
            print(f"   • ❌ Many trees underperforming - may need feature engineering review")

    print(f"\n🎉 Individual tree testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()