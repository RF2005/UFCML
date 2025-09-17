# UFC Machine Learning Project

A comprehensive UFC fight prediction system using machine learning models and Elo ratings.

## 📁 Project Structure

```
UFCML/
├── src/                     # Core source code
│   ├── core/               # Main ML modules
│   │   ├── individual_trees.py      # 67 individual decision trees
│   │   ├── advanced_ml_models.py    # Specialized ensemble models
│   │   ├── ml_models.py             # Basic ML models
│   │   └── fighter_matchup_predictor.py  # Fighter prediction system
│   ├── data/               # Data processing
│   │   ├── data_processor.py        # Data loading and processing
│   │   └── elo_system.py           # Elo rating calculations
│   └── utils/              # Utility functions
│       └── utils.py                # Export and helper functions
├── models/                 # Trained model files (42 .pkl files)
├── data/                   # Raw data files (.csv, .xlsx)
├── tests/                  # Test scripts
├── demos/                  # Demo and example scripts
├── web/                    # Web application
│   ├── app.py             # Flask web app
│   └── templates/         # HTML templates
├── docs/                   # Documentation
└── scripts/                # Standalone utility scripts
```

## 🚀 Quick Start

### Basic Usage
```bash
# Run main analysis pipeline
python main.py

# Run ultimate fight predictor (most comprehensive)
python ultimate_ufc_predictor.py

# Run web application
python web/app.py
```

### Demo Scripts
```bash
# Individual trees demo
python demos/run_individual_trees_demo.py

# Advanced models demo
python demos/run_advanced_analysis.py

# Fighter matchup demo
python demos/demo_fighter_matchups.py
```

## 🧪 Testing

```bash
# Test individual trees
python tests/test_individual_trees.py

# Test advanced models
python tests/test_advanced_models.py

# Test overfitting fixes
python tests/test_overfitting_fixes.py
```

## 📊 Models Available

- **67 Individual Decision Trees** - Each focused on specific fighter metrics
- **Specialized Ensemble Models** - Combined striking, grappling, positional trees
- **Fighter Matchup Predictor** - Hypothetical fight predictions
- **Elo Rating System** - Dynamic fighter rankings

## 🔧 Key Features

- **Overfitting Prevention**: Temporal validation, bootstrap sampling, cross-validation
- **Data Leakage Prevention**: Cleaned features, proper train/test splits
- **Comprehensive Analysis**: 67 trees covering all fight aspects
- **Web Interface**: Interactive fighter matchup predictions
- **Professional Structure**: Clean, organized codebase

## 📈 Performance

- **Accuracy Range**: 63.4% - 84.5% across different trees
- **Overfitting Risk**: Low (3-13% train/test gaps)
- **Cross-Validation**: 5-fold stratified with temporal splits
- **Model Count**: 67 trees within statistical safety bounds

## 🛠️ Development

The project follows Python best practices with:
- Modular architecture
- Clean import structure
- Comprehensive testing
- Professional organization
- Scalable design for future additions