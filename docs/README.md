# UFC Machine Learning Project

A comprehensive system for analyzing UFC fight data using Elo ratings and machine learning models.

## Project Structure

```
UFCML/
├── __init__.py          # Package initialization
├── main.py              # Main runner script
├── elo_system.py        # Elo rating system implementation
├── data_processor.py    # Data loading and processing utilities
├── ml_models.py         # Machine learning models (decision trees, random forests)
├── utils.py             # Export utilities and helper functions
└── README.md            # This file
```

## Features

### 1. Elo Rating System
- Implements chess Elo rating system for UFC fighters
- Each fighter starts with 1500 rating
- Updates ratings based on fight outcomes
- Uses formulas:
  - Expected probability: `Ea = 1/(1 + 10^((Rb - Ra)/400))`
  - Rating update: `R' = R + K(S - E)`

### 2. Data Processing
- Loads UFC fight data from CSV
- Processes fights chronologically
- Handles missing data and edge cases
- Supports filtering by date range and division

### 3. Machine Learning Models
- **Decision Trees**: Single tree for interpretable predictions
- **Random Forest**: Ensemble of 100 trees for improved accuracy
- Features based on Elo ratings:
  - Elo difference between fighters
  - Individual Elo ratings
  - Expected probabilities
  - Statistical aggregations (min, max, average)

### 4. Export & Analysis
- Export rankings to Excel format
- Display top fighters and statistics
- Save trained models for future use
- Comprehensive analysis reports

## Usage

### Quick Start
```bash
cd UFCML
python main.py
```

### Using Individual Modules
```python
from UFCML import EloRatingSystem, load_ufc_data, create_elo_decision_tree

# Initialize Elo system
elo = EloRatingSystem()

# Load and process data
df = load_ufc_data()
fight_results = process_fights_and_calculate_elo(df, elo)

# Train models
dt, features, accuracy = create_elo_decision_tree(fight_results)
```

## Requirements

```bash
pip install pandas numpy scikit-learn openpyxl joblib matplotlib
```

## Input Data

The system expects a CSV file with columns:
- `r_name`: Red corner fighter name
- `b_name`: Blue corner fighter name
- `winner`: Name of the winning fighter
- `date`: Fight date (for chronological processing)
- `event_name`: Event name (optional)
- `division`: Weight division (optional)

## Output Files

- `UFC_Fighter_Elo_Ratings.xlsx`: Complete fighter rankings
- `ufc_elo_decision_tree.pkl`: Trained decision tree model
- `ufc_elo_random_forest.pkl`: Trained random forest model
- `ufc_model_data.pkl`: Complete model data for analysis

## Model Performance

Typical results with ~8,000 UFC fights:
- Decision Tree accuracy: ~66%
- Random Forest accuracy: ~65%
- Most important features: Fighter Elo ratings and probability differences

## Key Features by Module

### `elo_system.py`
- `EloRatingSystem`: Core rating calculation engine
- Expected probability calculations
- Rating updates after fights

### `data_processor.py`
- `load_ufc_data()`: Load CSV data
- `process_fights_and_calculate_elo()`: Process fights chronologically
- Dataset filtering and statistics

### `ml_models.py`
- `create_elo_decision_tree()`: Train decision tree
- `create_elo_random_forest()`: Train random forest
- Model saving and prediction utilities

### `utils.py`
- `export_to_excel()`: Export rankings to Excel
- Rating statistics and categorization
- Display utilities

## Future Enhancements

- Additional fighter statistics (reach, age, etc.)
- Style-based matchup analysis
- Real-time prediction interface
- Historical performance tracking
- Advanced ensemble methods