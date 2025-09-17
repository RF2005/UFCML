#!/usr/bin/env python3
"""
UFC Fight Prediction Demo
Demonstrates the combined random forest prediction system
"""

import sys
sys.path.append('/Users/ralphfrancolini/UFCML')

from src.core.individual_trees import UFC_Individual_Tree_Forest
from src.core.advanced_ml_models import UFC_SpecializedEnsemble

def run_demo():
    print("🥊 UFC FIGHT PREDICTION DEMO")
    print("=" * 50)
    print("Powered by 32+ specialized AI models")
    print("=" * 50)

    print("\n✅ This demo has been simplified!")
    print("\n📝 Usage Instructions:")
    print("1. Run 'python3 ultimate_ufc_predictor.py' for comprehensive fight predictions")
    print("2. Run 'python3 demos/run_individual_trees_demo.py' for individual tree analysis")
    print("3. Run 'python3 web/app.py' for interactive web interface")
    print("4. Use the prediction results to inform your UFC betting/analysis")

if __name__ == "__main__":
    run_demo()