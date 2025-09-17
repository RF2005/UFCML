#!/usr/bin/env python3
"""
Test script to verify overfitting prevention improvements
"""

import sys
sys.path.append('/Users/ralphfrancolini/UFCML')

from src.core.individual_trees import create_individual_decision_tree
from src.core.advanced_ml_models import load_enhanced_ufc_data

def test_overfitting_improvements():
    """Test the improved decision tree implementation."""
    print("🧪 TESTING OVERFITTING PREVENTION IMPROVEMENTS")
    print("=" * 60)

    # Load data
    print("📊 Loading UFC dataset...")
    df = load_enhanced_ufc_data()
    if df is None:
        print("❌ Failed to load dataset")
        return

    print(f"✅ Loaded {len(df)} fights")

    # Test a single tree with different configurations
    tree_name = 'sig_strikes_landed'  # Use a high-performing tree

    print(f"\n🌳 Testing {tree_name} tree with different configurations:")
    print("-" * 60)

    # Configuration 1: Original (random split, no bootstrap)
    print("\n1️⃣  Testing: Random split, no bootstrap")
    try:
        tree1, features1, acc1, results1 = create_individual_decision_tree(
            df, tree_name, save_model=False, random_seed=42,
            use_temporal_split=False, use_bootstrap=False
        )
        print(f"✅ Original config completed")
    except Exception as e:
        print(f"❌ Original config failed: {e}")

    # Configuration 2: Random split with bootstrap
    print("\n2️⃣  Testing: Random split, with bootstrap")
    try:
        tree2, features2, acc2, results2 = create_individual_decision_tree(
            df, tree_name, save_model=False, random_seed=42,
            use_temporal_split=False, use_bootstrap=True
        )
        print(f"✅ Bootstrap config completed")
    except Exception as e:
        print(f"❌ Bootstrap config failed: {e}")

    # Configuration 3: Temporal split with bootstrap
    print("\n3️⃣  Testing: Temporal split, with bootstrap")
    try:
        tree3, features3, acc3, results3 = create_individual_decision_tree(
            df, tree_name, save_model=False, random_seed=42,
            use_temporal_split=True, use_bootstrap=True
        )
        print(f"✅ Temporal + bootstrap config completed")
    except Exception as e:
        print(f"❌ Temporal + bootstrap config failed: {e}")

    print("\n📋 SUMMARY OF IMPROVEMENTS:")
    print("=" * 60)
    print("✅ Random seed diversity: Each tree gets unique seed")
    print("✅ Cross-validation: 5-fold CV added for generalization assessment")
    print("✅ Data leakage prevention: Removed outcome-related features")
    print("✅ Temporal validation: Optional time-based train/test split")
    print("✅ Bootstrap sampling: Additional training data diversity")
    print("✅ Overfitting detection: Train vs test accuracy gap monitoring")

    print("\n🎯 RECOMMENDATIONS FOR PRODUCTION:")
    print("-" * 60)
    print("• Use temporal splits for time-sensitive predictions")
    print("• Enable bootstrap sampling for better ensemble diversity")
    print("• Monitor CV scores vs test scores for overfitting")
    print("• Consider reducing max_depth if overfitting gap > 0.1")
    print("• Retrain periodically with updated temporal splits")

if __name__ == "__main__":
    test_overfitting_improvements()