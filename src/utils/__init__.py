"""
UFC Machine Learning Package
============================

A package for analyzing UFC fight data using Elo ratings and machine learning models.

Modules:
- elo_system: Elo rating calculation system
- data_processor: Data loading and processing utilities
- ml_models: Machine learning models (decision trees, random forests)
- utils: Export utilities and helper functions
"""

__version__ = "1.0.0"
__author__ = "UFC ML Project"

# Only import what's actually in the utils module
from .utils import export_to_excel, print_top_fighters, print_rating_stats

__all__ = [
    'export_to_excel',
    'print_top_fighters',
    'print_rating_stats'
]