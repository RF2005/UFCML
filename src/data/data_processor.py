"""
Data Processing Module for UFC Fight Data
=========================================

Handles loading, cleaning, and processing of UFC fight data.
Processes fights chronologically to calculate Elo ratings.
"""

import pandas as pd


def load_ufc_data(file_path='/Users/ralphfrancolini/Desktop/ufc_data.csv'):
    """
    Load UFC dataset from specified path.

    Args:
        file_path (str): Path to the UFC dataset CSV file

    Returns:
        pandas.DataFrame or None: Loaded dataset or None if file not found
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except FileNotFoundError:
        print(f"Could not find UFC dataset at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading UFC dataset: {e}")
        return None


def process_fights_and_calculate_elo(df, elo_system):
    """
    Process fight data and update Elo ratings chronologically.

    Args:
        df (pandas.DataFrame): UFC fight dataset
        elo_system (EloRatingSystem): Elo rating system instance

    Returns:
        list: List of dictionaries containing fight results and rating updates
    """
    if df is None:
        return None

    # Sort by date to process fights chronologically
    if 'date' in df.columns:
        df = df.sort_values('date')

    fight_results = []

    for index, row in df.iterrows():
        # Using actual dataset column names
        fighter_a = row['r_name']  # Red corner fighter
        fighter_b = row['b_name']  # Blue corner fighter
        winner = row['winner']

        # Skip fights with missing data
        if pd.isna(fighter_a) or pd.isna(fighter_b) or pd.isna(winner):
            continue

        # Determine result (1 if fighter_a won, 0 if fighter_b won, 0.5 for draw)
        if winner == fighter_a:
            result_a = 1
        elif winner == fighter_b:
            result_a = 0
        else:  # Draw or no contest
            result_a = 0.5

        # Get ratings before fight
        rating_a_before = elo_system.get_fighter_rating(fighter_a)
        rating_b_before = elo_system.get_fighter_rating(fighter_b)

        # Calculate expected probabilities
        expected_a, expected_b = elo_system.calculate_expected_probability(rating_a_before, rating_b_before)

        # Update ratings
        rating_a_after, rating_b_after = elo_system.update_ratings(fighter_a, fighter_b, result_a)

        # Store fight result information
        fight_results.append({
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'winner': winner,
            'rating_a_before': rating_a_before,
            'rating_b_before': rating_b_before,
            'expected_prob_a': expected_a,
            'expected_prob_b': expected_b,
            'rating_a_after': rating_a_after,
            'rating_b_after': rating_b_after,
            'date': row.get('date', ''),
            'event': row.get('event_name', ''),
            'method': row.get('method', ''),
            'division': row.get('division', '')
        })

    return fight_results


def get_dataset_info(df):
    """
    Get basic information about the UFC dataset.

    Args:
        df (pandas.DataFrame): UFC fight dataset

    Returns:
        dict: Dictionary containing dataset statistics
    """
    if df is None:
        return None

    info = {
        'total_fights': len(df),
        'columns': list(df.columns),
        'date_range': None,
        'unique_fighters': set(),
        'divisions': set()
    }

    # Get date range if date column exists
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            info['date_range'] = (df['date'].min(), df['date'].max())
        except:
            pass

    # Count unique fighters
    if 'r_name' in df.columns and 'b_name' in df.columns:
        red_fighters = set(df['r_name'].dropna().unique())
        blue_fighters = set(df['b_name'].dropna().unique())
        info['unique_fighters'] = red_fighters.union(blue_fighters)

    # Get divisions
    if 'division' in df.columns:
        info['divisions'] = set(df['division'].dropna().unique())

    return info


def filter_fights_by_date(df, start_date=None, end_date=None):
    """
    Filter fights by date range.

    Args:
        df (pandas.DataFrame): UFC fight dataset
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        pandas.DataFrame: Filtered dataset
    """
    if df is None or 'date' not in df.columns:
        return df

    try:
        df['date'] = pd.to_datetime(df['date'])

        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['date'] >= start_date]

        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['date'] <= end_date]

        return df
    except Exception as e:
        print(f"Error filtering by date: {e}")
        return df


def filter_fights_by_division(df, divisions):
    """
    Filter fights by weight division(s).

    Args:
        df (pandas.DataFrame): UFC fight dataset
        divisions (str or list): Division name(s) to filter by

    Returns:
        pandas.DataFrame: Filtered dataset
    """
    if df is None or 'division' not in df.columns:
        return df

    if isinstance(divisions, str):
        divisions = [divisions]

    return df[df['division'].isin(divisions)]