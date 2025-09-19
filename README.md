# UFC Machine Learning Project

Predicting UFC fight outcomes is noisy, data-scarce, and extremely sensitive to leakage. This repository implements a realistic end-to-end pipeline built around a regularized random forest plus aggressive feature engineering and temporal validation. The goal is not headline accuracy, but a transparent benchmark you can iterate on without fooling yourself.

## Highlights
- **Regularized random forest** with engineered matchup features, calibrated probabilities, and overfitting diagnostics.
- **Strict temporal pipeline** (`tests/validation/proper_temporal_holdout_test.py`) that freezes fighter profiles before a holdout period to avoid leakage.
- **Backtesting utilities** for multi-era walk-forward evaluation (`tests/comprehensive_backtest.py`, `backtest/comprehensive_backtest.py`).
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

## Repository Layout (selected folders)
```
UFCML/
├── enhanced_random_forest.py          # Main training/CLI script (regularized RF)
├── enhanced_feature_engineering.py    # Feature generation and fighter profiles
├── tests/
│   ├── validation/
│   │   └── proper_temporal_holdout_test.py  # Leak-proof temporal evaluation
│   ├── comprehensive_backtest.py      # Walk-forward testing entry point
│   └── ...                            # Additional diagnostics
├── src/
│   ├── core/
│   │   ├── fighter_matchup_predictor.py  # Interactive prediction interface
│   │   └── ml_models.py                  # Baseline models and helpers
│   ├── data/                            # Data ingestion utilities
│   └── utils/                           # Shared helpers
└── models/
    └── enhanced_ufc_random_forest.pkl  # Saved model (Git LFS)
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
- **Baseline training run**
  ```bash
  python enhanced_random_forest.py
  ```
  Outputs temporal train/validation/test metrics, calibrates probabilities, and saves the calibrated model to `models/enhanced_ufc_random_forest.pkl`.

- **Leak-proof temporal holdout**
  ```bash
  python tests/validation/proper_temporal_holdout_test.py
  ```
  Rebuilds fighter profiles using only pre-cutoff data, retrains the model, and scores the post-cutoff fights. These outputs are the definitive performance numbers referenced above.

- **Comprehensive walk-forward backtest**
  ```bash
  python tests/comprehensive_backtest.py
  ```
  Iterates through historical time windows to show how performance drifts across eras.

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
