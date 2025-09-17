"""
Main Runner Script for UFC Machine Learning Project
===================================================

This script runs the complete UFC analysis pipeline:
1. Load UFC fight data
2. Calculate Elo ratings for all fighters
3. Train machine learning models
4. Export results to Excel

Usage:
    python main.py
"""

from src.data.elo_system import EloRatingSystem
from src.data.data_processor import load_ufc_data, process_fights_and_calculate_elo, get_dataset_info
from src.core.ml_models import create_elo_decision_tree, create_elo_random_forest
from src.utils.utils import export_to_excel, print_top_fighters, print_rating_stats


def main():
    """Main function to run the UFC Elo rating system and ML analysis."""
    print("UFC Fighter Elo Rating System & Machine Learning")
    print("=" * 60)

    # Initialize Elo system
    print("\nInitializing Elo rating system...")
    elo_system = EloRatingSystem(initial_rating=1500, k_factor=32)

    # Load data
    print("Loading UFC dataset...")
    df = load_ufc_data()

    if df is not None:
        # Show dataset info
        dataset_info = get_dataset_info(df)
        print(f"Dataset loaded successfully!")
        print(f"Total fights: {len(df)}")
        print(f"Unique fighters: {len(dataset_info['unique_fighters'])}")
        if dataset_info['date_range']:
            print(f"Date range: {dataset_info['date_range'][0].strftime('%Y-%m-%d')} to {dataset_info['date_range'][1].strftime('%Y-%m-%d')}")

        # Process fights and calculate Elo ratings
        print("\nProcessing fights and calculating Elo ratings...")
        fight_results = process_fights_and_calculate_elo(df, elo_system)

        if fight_results:
            print(f"Successfully processed {len(fight_results)} fights")

            # Get final ratings
            final_ratings = elo_system.get_all_ratings()
            print(f"Calculated ratings for {len(final_ratings)} fighters")

            # Display top fighters
            print_top_fighters(final_ratings, n=20)

            # Display rating statistics
            print_rating_stats(final_ratings)

            # Export results to Excel
            print("\n" + "="*60)
            print("EXPORTING RESULTS")
            print("="*60)
            export_to_excel(final_ratings)

            # Create and train machine learning models
            print("\n" + "="*60)
            print("MACHINE LEARNING ANALYSIS")
            print("="*60)

            print("\nTraining Decision Tree...")
            dt, dt_features, dt_accuracy, dt_results = create_elo_decision_tree(fight_results)

            print("\nTraining Random Forest...")
            rf, rf_features, rf_accuracy, rf_results = create_elo_random_forest(fight_results)

            # Summary
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(f"✓ Processed {len(fight_results)} fights")
            print(f"✓ Calculated Elo ratings for {len(final_ratings)} fighters")
            print(f"✓ Decision Tree accuracy: {dt_accuracy:.1%}")
            print(f"✓ Random Forest accuracy: {rf_accuracy:.1%}")
            print(f"✓ Results exported to Excel")
            print(f"✓ Models saved for future use")

            print("\nFiles created:")
            print("- UFC_Fighter_Elo_Ratings.xlsx")
            print("- ufc_elo_decision_tree.pkl")
            print("- ufc_elo_random_forest.pkl")
            print("- ufc_model_data.pkl")

        else:
            print("No fight results to process")
    else:
        print("Could not load dataset")
        print("Please ensure 'ufc_data.csv' is available at the specified path")


if __name__ == "__main__":
    main()