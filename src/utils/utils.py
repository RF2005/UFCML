"""
Utility Functions for UFC ML Project
====================================

Contains export functions and other helper utilities.
"""

import pandas as pd


def export_to_excel(fighter_ratings, filename="UFC_Fighter_Elo_Ratings.xlsx"):
    """
    Export fighter ratings to Excel format.

    Args:
        fighter_ratings (dict): Dictionary of fighter names and ratings
        filename (str): Output filename for the Excel file

    Returns:
        pandas.DataFrame: DataFrame containing the exported data
    """
    try:
        # Convert ratings to DataFrame
        ratings_df = pd.DataFrame(list(fighter_ratings.items()),
                                columns=['Fighter', 'Elo Rating'])
        ratings_df = ratings_df.sort_values('Elo Rating', ascending=False).reset_index(drop=True)
        ratings_df['Rank'] = range(1, len(ratings_df) + 1)

        # Reorder columns
        ratings_df = ratings_df[['Rank', 'Fighter', 'Elo Rating']]

        print("Fighter Elo Ratings (Top 20):")
        print(ratings_df.head(20).to_string(index=False))

        # Save to Excel
        ratings_df.to_excel(filename, index=False, sheet_name='UFC Fighter Elo Ratings')
        print(f"\nRatings saved to '{filename}'")

        return ratings_df

    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return None


def export_to_csv(fighter_ratings, filename="UFC_Fighter_Elo_Ratings.csv"):
    """
    Export fighter ratings to CSV format.

    Args:
        fighter_ratings (dict): Dictionary of fighter names and ratings
        filename (str): Output filename for the CSV file

    Returns:
        pandas.DataFrame: DataFrame containing the exported data
    """
    try:
        # Convert ratings to DataFrame
        ratings_df = pd.DataFrame(list(fighter_ratings.items()),
                                columns=['Fighter', 'Elo Rating'])
        ratings_df = ratings_df.sort_values('Elo Rating', ascending=False).reset_index(drop=True)
        ratings_df['Rank'] = range(1, len(ratings_df) + 1)

        # Reorder columns
        ratings_df = ratings_df[['Rank', 'Fighter', 'Elo Rating']]

        # Save to CSV
        ratings_df.to_csv(filename, index=False)
        print(f"Ratings saved to '{filename}'")

        return ratings_df

    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return None


def print_top_fighters(fighter_ratings, n=20):
    """
    Print the top N fighters by Elo rating.

    Args:
        fighter_ratings (dict): Dictionary of fighter names and ratings
        n (int): Number of top fighters to display
    """
    ratings_df = pd.DataFrame(list(fighter_ratings.items()),
                            columns=['Fighter', 'Elo Rating'])
    ratings_df = ratings_df.sort_values('Elo Rating', ascending=False).reset_index(drop=True)
    ratings_df['Rank'] = range(1, len(ratings_df) + 1)

    print(f"\nTop {n} UFC Fighters by Elo Rating:")
    print("=" * 50)
    print(ratings_df.head(n).to_string(index=False))


def get_fighter_stats(fighter_ratings):
    """
    Get basic statistics about the fighter ratings.

    Args:
        fighter_ratings (dict): Dictionary of fighter names and ratings

    Returns:
        dict: Dictionary containing rating statistics
    """
    ratings = list(fighter_ratings.values())

    stats = {
        'total_fighters': len(ratings),
        'highest_rating': max(ratings) if ratings else 0,
        'lowest_rating': min(ratings) if ratings else 0,
        'average_rating': sum(ratings) / len(ratings) if ratings else 0,
        'median_rating': sorted(ratings)[len(ratings) // 2] if ratings else 0
    }

    return stats


def print_rating_stats(fighter_ratings):
    """
    Print statistics about the fighter ratings.

    Args:
        fighter_ratings (dict): Dictionary of fighter names and ratings
    """
    stats = get_fighter_stats(fighter_ratings)

    print(f"\nFighter Rating Statistics:")
    print("=" * 30)
    print(f"Total Fighters: {stats['total_fighters']}")
    print(f"Highest Rating: {stats['highest_rating']:.2f}")
    print(f"Lowest Rating: {stats['lowest_rating']:.2f}")
    print(f"Average Rating: {stats['average_rating']:.2f}")
    print(f"Median Rating: {stats['median_rating']:.2f}")


def create_rating_categories(fighter_ratings):
    """
    Categorize fighters by their Elo ratings.

    Args:
        fighter_ratings (dict): Dictionary of fighter names and ratings

    Returns:
        dict: Dictionary of rating categories with fighter lists
    """
    categories = {
        'Elite (1650+)': [],
        'Excellent (1550-1649)': [],
        'Good (1450-1549)': [],
        'Average (1350-1449)': [],
        'Below Average (<1350)': []
    }

    for fighter, rating in fighter_ratings.items():
        if rating >= 1650:
            categories['Elite (1650+)'].append((fighter, rating))
        elif rating >= 1550:
            categories['Excellent (1550-1649)'].append((fighter, rating))
        elif rating >= 1450:
            categories['Good (1450-1549)'].append((fighter, rating))
        elif rating >= 1350:
            categories['Average (1350-1449)'].append((fighter, rating))
        else:
            categories['Below Average (<1350)'].append((fighter, rating))

    # Sort each category by rating
    for category in categories:
        categories[category].sort(key=lambda x: x[1], reverse=True)

    return categories


def print_rating_categories(fighter_ratings):
    """
    Print fighters grouped by rating categories.

    Args:
        fighter_ratings (dict): Dictionary of fighter names and ratings
    """
    categories = create_rating_categories(fighter_ratings)

    print(f"\nFighters by Rating Category:")
    print("=" * 40)

    for category, fighters in categories.items():
        print(f"\n{category}: {len(fighters)} fighters")
        if fighters:
            for i, (fighter, rating) in enumerate(fighters[:5]):  # Show top 5 in each category
                print(f"  {i+1}. {fighter}: {rating:.1f}")
            if len(fighters) > 5:
                print(f"  ... and {len(fighters) - 5} more")