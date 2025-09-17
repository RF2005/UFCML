"""
Elo Rating System for UFC Fighters
==================================

Implements the chess Elo rating system adapted for UFC fights.
Each fighter starts with an initial rating of 1500 and ratings are updated
after each fight based on the outcome and expected probabilities.
"""

class EloRatingSystem:
    def __init__(self, initial_rating=1500, k_factor=32):
        """
        Initialize the Elo rating system.

        Args:
            initial_rating (int): Starting rating for new fighters (default: 1500)
            k_factor (int): K-factor for rating updates (default: 32)
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.fighter_ratings = {}

    def get_fighter_rating(self, fighter_name):
        """
        Get current rating for a fighter, initialize if new.

        Args:
            fighter_name (str): Name of the fighter

        Returns:
            float: Current Elo rating of the fighter
        """
        if fighter_name not in self.fighter_ratings:
            self.fighter_ratings[fighter_name] = self.initial_rating
        return self.fighter_ratings[fighter_name]

    def calculate_expected_probability(self, rating_a, rating_b):
        """
        Calculate expected probability of each player winning using Elo formula.

        Uses the formula: Ea = 1/(1 + 10^((Rb - Ra)/400))

        Args:
            rating_a (float): Rating of fighter A
            rating_b (float): Rating of fighter B

        Returns:
            tuple: (expected_probability_a, expected_probability_b)
        """
        # Ea = 1/(1 + 10^((Rb - Ra)/400))
        expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
        expected_b = 1 / (1 + 10**((rating_a - rating_b) / 400))
        return expected_a, expected_b

    def update_ratings(self, fighter_a, fighter_b, result_a):
        """
        Update ratings after a fight using the formula: R' = R + K(S - E)

        Args:
            fighter_a (str): Name of fighter A
            fighter_b (str): Name of fighter B
            result_a (float): Result for fighter A (1 if won, 0 if lost, 0.5 for draw)

        Returns:
            tuple: (new_rating_a, new_rating_b)
        """
        rating_a = self.get_fighter_rating(fighter_a)
        rating_b = self.get_fighter_rating(fighter_b)

        expected_a, expected_b = self.calculate_expected_probability(rating_a, rating_b)

        # R' = R + K(S - E)
        new_rating_a = rating_a + self.k_factor * (result_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - result_a) - expected_b)

        self.fighter_ratings[fighter_a] = new_rating_a
        self.fighter_ratings[fighter_b] = new_rating_b

        return new_rating_a, new_rating_b

    def get_all_ratings(self):
        """
        Return all fighter ratings sorted by rating (highest first).

        Returns:
            dict: Dictionary of fighter names and ratings, sorted by rating
        """
        return dict(sorted(self.fighter_ratings.items(), key=lambda x: x[1], reverse=True))

    def get_fighter_count(self):
        """
        Get the total number of fighters in the system.

        Returns:
            int: Number of fighters with ratings
        """
        return len(self.fighter_ratings)

    def get_top_fighters(self, n=10):
        """
        Get the top N fighters by Elo rating.

        Args:
            n (int): Number of top fighters to return

        Returns:
            dict: Dictionary of top N fighters and their ratings
        """
        all_ratings = self.get_all_ratings()
        return dict(list(all_ratings.items())[:n])