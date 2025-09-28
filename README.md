# UFC Machine Learning Project

Predicting UFC fight outcomes is noisy, data-scarce, and extremely sensitive to leakage. This repository implements a realistic end-to-end pipeline built around a regularized random forest plus aggressive feature engineering and temporal validation. The goal is not headline accuracy, but a transparent benchmark you can iterate on without fooling yourself.

## Highlights
- **Regularized random forest** with engineered matchup features, calibrated probabilities, and overfitting diagnostics.
- **Dual temporal validation system**:
  - **Single-split temporal holdout** (`tests/validation/proper_temporal_holdout_test.py`) that freezes fighter profiles before a holdout period to avoid leakage
  - **Rolling temporal backtesting** (`tests/validation/enhanced_temporal_backtesting.py`) that validates model stability across multiple historical periods
- **Walk-forward backtesting** (`tests/comprehensive_backtest.py`) for multi-era evaluation.
- **Interactive predictor** (`src/core/fighter_matchup_predictor.py`) for quick what-if matchups using the trained model.

## Current Performance (March 2025 data freeze)
All results come from the temporal split inside `tests/validation/proper_temporal_holdout_test.py`, using data up to 10 March 2025 for feature construction and fights after that date for holdout evaluation.

| Metric                                   | Value  | Notes |
|------------------------------------------|--------|-------|
| Training accuracy (temporal split)       | 70.1%  | Chronological train segment |
| Validation accuracy                      | 61.0%  | Held-out chronological slice |
| Test accuracy                            | 59.6%  | Final temporal test slice |
| Proper temporal holdout accuracy         | 60.3%  | Fights after the freeze date |
| 5-fold CV (stratified, shuffled)         | 59.3% ± 1.9% | For sanity only; temporal metrics take priority |

Anything well above ~60% should be viewed with suspicion unless you can rule out leakage or bad labeling. These numbers are realistic for matchup-only models with historical stats.

## Repository Layout
```
UFCML/
├── main.py                            # Main entry point
├── requirements.txt                   # Python dependencies
├── src/
│   ├── core/                          # Core ML components
│   │   ├── enhanced_random_forest.py         # Main enhanced random forest model
│   │   ├── enhanced_feature_engineering.py   # Advanced feature engineering
│   │   ├── regularized_random_forest.py      # Regularized random forest
│   │   ├── fighter_matchup_predictor.py      # Interactive prediction interface
│   │   ├── advanced_ml_models.py             # Specialized ML models
│   │   ├── individual_trees.py               # Individual decision trees
│   │   ├── ultimate_ufc_predictor.py         # Ultimate predictor interface
│   │   └── ml_models.py                      # Baseline models and helpers
│   ├── data/                          # Data ingestion utilities
│   └── utils/                         # Shared helper functions
├── tests/                             # Validation and testing
│   ├── validation/
│   │   ├── proper_temporal_holdout_test.py   # Single-split temporal holdout (gold standard)
│   │   ├── enhanced_temporal_backtesting.py  # Rolling temporal backtesting
│   │   └── true_fighter_holdout_test.py      # Complete fighter separation test
│   ├── comprehensive_backtest.py     # Walk-forward testing across eras
│   ├── test_advanced_models.py       # Unit tests for advanced models
│   ├── test_overfitting_fixes.py     # Overfitting prevention tests
│   └── test_individual_trees.py      # Individual tree testing
├── analysis/                          # Data analysis and visualizations
│   ├── data_analysis_plots.py         # Analysis plotting utilities
│   └── ufc_features_focused.png       # Feature correlation visualization
├── backtest/                          # Backtesting utilities
│   ├── realistic_backtest.py          # Betting scenario simulations
│   └── simple_backtest.py             # Basic temporal backtesting
├── models/                            # Trained model artifacts
│   └── enhanced_ufc_random_forest.pkl # Main enhanced model (Git LFS)
├── web/                               # Web interface components
├── demos/                             # Demo scripts and examples
├── deployment/                        # Deployment configurations
├── scripts/                           # Utility scripts
└── archive/                           # Archived/legacy components
```

## Setup
1. **Clone & virtual environment**
   ```bash
   git clone https://github.com/<you>/UFCML.git
   cd UFCML
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Provide the dataset**
   `EnhancedFeatureEngineer` expects a CSV with detailed fight stats. By default `src/core/advanced_ml_models.load_enhanced_ufc_data` looks at `~/Desktop/ufc_data.csv`. Update that path or set up a symlink to point to your copy.

## Training & Evaluation

### Core Validation Methods
The repository implements multiple validation approaches to ensure robust, realistic performance estimates:

#### 1. Single-Split Temporal Holdout (Gold Standard)
```bash
python tests/validation/proper_temporal_holdout_test.py
```
- **Purpose**: Test real-world prediction capability with frozen fighter knowledge
- **Method**: Single temporal split at March 10, 2025
- **Key Feature**: Fighter profiles frozen at cutoff (no future data leakage)
- **Result**: Definitive performance numbers referenced in metrics table above

#### 2. Rolling Temporal Backtesting (Stability Analysis)
```bash
python tests/validation/enhanced_temporal_backtesting.py
```
- **Purpose**: Validate model stability across different historical periods
- **Method**: Multiple rolling windows with strict temporal gaps
- **Key Feature**: Detects model degradation over time and era-specific performance
- **Result**: Comprehensive stability analysis across historical periods

#### 3. True Fighter Holdout (Generalization Test)
```bash
python tests/validation/true_fighter_holdout_test.py
```
- **Purpose**: Test generalization to completely unknown fighters
- **Method**: 20% of fighters held out entirely from training
- **Key Feature**: Complete fighter separation between train/test sets
- **Result**: Tests if model learns transferable patterns vs memorizes specific fighters

### Training & Quick Testing
- **Baseline training run**
  ```bash
  python src/core/enhanced_random_forest.py
  ```
  Outputs temporal train/validation/test metrics, calibrates probabilities, and saves the calibrated model to `models/enhanced_ufc_random_forest.pkl`.

- **Walk-forward backtest**
  ```bash
  python tests/comprehensive_backtest.py
  ```
  Multi-era walk-forward validation across historical periods.

- **Interactive predictions**
  ```bash
  python -c "from src.core.fighter_matchup_predictor import FighterMatchupPredictor
pred = FighterMatchupPredictor()
res = pred.predict_matchup('Jon Jones', 'Stipe Miocic')
pred.display_matchup_prediction(res)"
  ```
  (Requires the trained model artefact.)

## Key Implementation Details
- **Feature engineering** blends matchup-level differentials (win rate, striking pace, control time, etc.) with profile momentum. Profiles are frozen at the temporal cutoff when evaluating future fights.
- **Regularization**: 200 estimators, `max_depth=4`, `min_samples_split=60`, `min_samples_leaf=25`, and per-tree feature subsampling. Additional dynamic guards inflate these thresholds when data is scarce.
- **Calibration & diagnostics**: Isotonic vs. sigmoid calibration is chosen by validation Brier score; the script reports Brier, log-loss, and train/validation/test gaps to make overfitting obvious.

## Data Ethics & Caution
Sports models degrade quickly as fighters age, camps change, or odds move. The holdout set above covers only 267 fights after March 2025; expect wide confidence intervals. Treat this repository as a research sandbox, not a production betting system.

## Roadmap
- Trim or reweight low-signal features flagged by permutation importance.
- Automate temporal cross-validation so each board decision is backed by out-of-sample evidence.
- Integrate betting market data for calibration and ROI analysis.
- Explore lightweight ensembling (e.g., averaging the RF with a calibrated logistic) once stability is proven.

Questions or contributions welcome—open an issue or PR with reproducible metrics from the provided scripts.
