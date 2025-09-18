# UFC Machine Learning Project

A **unified** UFC fight prediction system using enhanced random forest with advanced feature engineering, achieving **75.7% validated accuracy** across temporal splits.

## 🏗️ **Architecture (2024)**

**UNIFIED SYSTEM**: Single enhanced random forest with advanced feature engineering
**PERFORMANCE**: 75.7% average accuracy with proper regularization
**VALIDATION**: Comprehensive backtesting across 73 temporal periods (2010-2025)

## 📁 Project Structure

```
UFCML/
├── 🎯 CORE UNIFIED SYSTEM
│   ├── enhanced_random_forest.py       # Main prediction model (77% accuracy)
│   ├── enhanced_feature_engineering.py # Advanced feature extraction
│   ├── comprehensive_backtest.py       # Temporal validation framework
│   └── data_analysis_plots.py          # Data noise analysis
├── 🔧 SUPPORTING INFRASTRUCTURE
│   ├── src/core/
│   │   ├── fighter_matchup_predictor.py # Fighter prediction interface
│   │   └── ml_models.py                 # Basic ML models
│   ├── src/data/                        # Data processing
│   │   ├── data_processor.py
│   │   └── elo_system.py
│   └── src/utils/                       # Utility functions
├── 📊 TRAINED MODEL
│   └── enhanced_ufc_random_forest.pkl  # Main prediction model (20MB)
├── 📈 ANALYSIS & VALIDATION
│   ├── ufc_backtest_results.png        # Temporal validation results
│   ├── ufc_feature_relationships.png   # Feature analysis
│   └── ufc_noise_analysis.png          # Data noise assessment
├── 🧪 TESTING & DEMOS
│   ├── tests/                           # Test scripts
│   ├── demos/                           # Demo scripts
│   └── web/                             # Web application
└── 📚 DOCUMENTATION
    └── docs/                            # Documentation
```

## 🚀 Quick Start

### **Unified System (Recommended)**
```bash
# Train and validate the enhanced model
python enhanced_random_forest.py

# Run comprehensive backtesting
python comprehensive_backtest.py

# Analyze data characteristics
python data_analysis_plots.py

# Interactive fighter predictions
python -c "
from src.core.fighter_matchup_predictor import FighterMatchupPredictor
predictor = FighterMatchupPredictor()
result = predictor.predict_matchup('Jon Jones', 'Stipe Miocic')
predictor.display_matchup_prediction(result)
"
```

### **Additional Tools**
```bash
# Web application
python web/app.py
```

## 🌐 Deploying the Web App

The Flask interface runs anywhere you can install Python. This repo now ships with:

- `requirements.txt` – runtime dependencies (Flask, scikit-learn, etc.)
- `Procfile` – starts `gunicorn web.app:app`
- `runtime.txt` – pins Python 3.10 for Heroku/Render style hosts
- `models/enhanced_ufc_random_forest.pkl` – calibrated model (tracked via Git LFS)

### Render / Railway / Fly.io
1. Connect the repo to your platform of choice.
2. Build command: `pip install -r requirements.txt`.
3. Start command: `gunicorn web.app:app`.
4. (Optional) Set `MODEL_PATH` if you relocate the model file.

### Manual deployment (any VPS/container)
```bash
git clone https://github.com/<your-org>/UFCML.git
cd UFCML
pip install -r requirements.txt
gunicorn web.app:app --bind 0.0.0.0:8000
```
Expose the chosen port through your reverse proxy and the app is available publicly.

## 📊 **Performance Metrics**

### **Enhanced Random Forest (Main System)**
- **Test Accuracy**: 77.0% (regularized, realistic)
- **Cross-Validation**: 78.5% ± 0.5%
- **Overfitting Gap**: 1.4% (excellent generalization)
- **Features**: 17 advanced engineered features

### **Temporal Backtesting Results**
- **Average Accuracy**: 75.7% ± 2.9% across 73 periods
- **Time Range**: 2010-2025 UFC eras
- **Consistency**: Very stable (CV = 0.038)
- **Performance Trend**: Consistent across different UFC eras

### **Top Predictive Features**
1. **Win Rate Advantage** (35.7% importance)
2. **Recency Advantage** (26.5% importance)
3. **Striking Volume Advantage** (9.3% importance)
4. **Performance Trend Diff** (6.0% importance)
5. **Experience Advantage** (5.7% importance)

## 🔧 **Key Technical Features**

### **Enhanced Feature Engineering**
- **Weighted Recent Performance**: Last 3 fights get 60% weight
- **Style Matchup Matrix**: Striker vs Grappler dynamics
- **Temporal Features**: Days since last fight, activity level
- **Experience Differentials**: Fight count and career trajectory
- **Physical Advantages**: Height, reach, age differentials

### **Proper Regularization**
- **Controlled Trees**: 100 estimators (down from 200)
- **Shallow Depth**: 6 levels (down from 12)
- **Sample Requirements**: 20 min split, 10 min leaf
- **Feature Subsampling**: sqrt(n_features) per tree
- **Bootstrap Sampling**: 80% data per tree

### **Validation Methodology**
- **3-way Split**: Train (70%) / Validation (10%) / Test (20%)
- **Temporal Ordering**: Chronological splits prevent leakage
- **Cross-Validation**: 5-fold stratified validation
- **Backtesting**: Walk-forward temporal validation

## 🎯 **System Performance**

| Metric | Value | Status |
|--------|-------|---------|
| **Test Accuracy** | **77.0%** | ✅ Excellent |
| **Overfitting Gap** | **1.4%** | ✅ Very Low |
| **Cross-Validation** | **78.5% ± 0.5%** | ✅ Consistent |
| **Backtest Average** | **75.7% ± 2.9%** | ✅ Stable |

## 🧪 **Testing & Validation**

```bash
# Test the unified system
python -c "
from enhanced_random_forest import EnhancedUFCRandomForest
from enhanced_feature_engineering import EnhancedFeatureEngineer

engineer = EnhancedFeatureEngineer()
engineer.load_and_prepare_data()
enhanced_df = engineer.create_enhanced_training_data()

model = EnhancedUFCRandomForest()
model.feature_engineer = engineer
results = model.train(enhanced_df)
print(f'Test Accuracy: {results[\"test_accuracy\"]:.1%}')
"

# Run comprehensive backtesting
python comprehensive_backtest.py
```

## 📈 **Data Analysis**

The system includes comprehensive data analysis showing:
- **Moderate noise levels** (R² = 0.306)
- **Non-linear relationships** favoring tree-based models
- **Feature correlations** and importance rankings
- **Temporal consistency** across UFC eras

## 🛠️ **Development Notes**

### **Architecture**
- **Unified System**: Single enhanced random forest with advanced feature engineering
- **Performance**: Validated 75.7% accuracy across temporal periods
- **Design**: Clean, modular, production-ready implementation

### **Code Quality**
- **Modular Design**: Clean separation of concerns
- **Proper Validation**: No data leakage, temporal splits
- **Comprehensive Testing**: 73 temporal validation periods
- **Professional Structure**: Production-ready codebase

## 🚀 **Future Improvements**

- **Hyperparameter Optimization**: Grid search on validation sets
- **Neural Networks**: Explore deep learning approaches
- **External Data**: Incorporate betting odds, physical measurements
- **Real-time Updates**: Live fighter performance tracking
- **Web Interface**: Update to use unified system

---

## 📋 **Usage Guide**

**Main System**:
```python
from enhanced_random_forest import EnhancedUFCRandomForest
from enhanced_feature_engineering import EnhancedFeatureEngineer

# Initialize and train
engineer = EnhancedFeatureEngineer()
model = EnhancedUFCRandomForest()
model.feature_engineer = engineer

# Train and validate
enhanced_df = engineer.create_enhanced_training_data()
results = model.train(enhanced_df)
```

**Performance Expectations**:
- **Realistic Sports Prediction**: 75-77% accuracy
- **Proper Regularization**: <5% overfitting gap
- **Consistent Results**: Validated across time periods

---

*Last Updated: September 2024 - Unified enhanced random forest system*
