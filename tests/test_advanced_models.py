"""
Test Script for Advanced UFC ML Models
======================================

This script tests all the specialized decision trees and ensemble models
created for UFC fight prediction using comprehensive fight statistics.
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
    """Test all advanced ML models."""
    print("UFC Advanced Machine Learning Models Test")
    print("=" * 70)

    # Load enhanced dataset
    print("\nLoading enhanced UFC dataset...")
    df = load_enhanced_ufc_data()

    if df is None:
        print("Could not load dataset. Please check the file path.")
        return

    print(f"Dataset loaded successfully!")
    print(f"Total fights: {len(df)}")
    print(f"Columns available: {len(df.columns)}")

    # Test individual specialized decision trees
    print("\n" + "=" * 70)
    print("TESTING INDIVIDUAL SPECIALIZED DECISION TREES")
    print("=" * 70)

    # Test striking decision tree
    print("\n1. Testing Striking Decision Tree...")
    try:
        dt_striking, features_striking, acc_striking, _ = create_striking_decision_tree(df, save_model=False)
        print(f"✓ Striking tree trained successfully")
    except Exception as e:
        print(f"✗ Striking tree failed: {e}")

    # Test grappling decision tree
    print("\n2. Testing Grappling Decision Tree...")
    try:
        dt_grappling, features_grappling, acc_grappling, _ = create_grappling_decision_tree(df, save_model=False)
        print(f"✓ Grappling tree trained successfully")
    except Exception as e:
        print(f"✗ Grappling tree failed: {e}")

    # Test positional decision tree
    print("\n3. Testing Positional Decision Tree...")
    try:
        dt_positional, features_positional, acc_positional, _ = create_positional_decision_tree(df, save_model=False)
        print(f"✓ Positional tree trained successfully")
    except Exception as e:
        print(f"✗ Positional tree failed: {e}")

    # Test context decision tree
    print("\n4. Testing Context Decision Tree...")
    try:
        dt_context, features_context, acc_context, _ = create_context_decision_tree(df, save_model=False)
        print(f"✓ Context tree trained successfully")
    except Exception as e:
        print(f"✗ Context tree failed: {e}")

    # Test specialized ensemble
    print("\n" + "=" * 70)
    print("TESTING SPECIALIZED ENSEMBLE")
    print("=" * 70)

    try:
        ensemble = UFC_SpecializedEnsemble()
        results = ensemble.train_all_trees(df, save_models=False)
        print(f"✓ Specialized ensemble trained successfully")
    except Exception as e:
        print(f"✗ Specialized ensemble failed: {e}")

    # Test comprehensive random forest
    print("\n" + "=" * 70)
    print("TESTING COMPREHENSIVE RANDOM FOREST")
    print("=" * 70)

    try:
        rf_comprehensive, all_features, acc_comprehensive, _ = create_comprehensive_random_forest(
            df, n_estimators=50, save_model=False  # Reduced trees for faster testing
        )
        print(f"✓ Comprehensive random forest trained successfully")
    except Exception as e:
        print(f"✗ Comprehensive random forest failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("Individual Decision Trees:")
    try:
        print(f"  Striking Tree:      {acc_striking:.3f} accuracy")
        print(f"  Grappling Tree:     {acc_grappling:.3f} accuracy")
        print(f"  Positional Tree:    {acc_positional:.3f} accuracy")
        print(f"  Context Tree:       {acc_context:.3f} accuracy")
    except:
        print("  Some individual trees failed to train")

    try:
        print(f"\nEnsemble Model:       {results['ensemble']['accuracy']:.3f} accuracy")
    except:
        print("\nEnsemble Model:       Failed to train")

    try:
        print(f"Comprehensive RF:     {acc_comprehensive:.3f} accuracy")
        print(f"Total Features Used:  {len(all_features)}")
    except:
        print("Comprehensive RF:     Failed to train")

    print("\n✓ All tests completed!")


if __name__ == "__main__":
    main()