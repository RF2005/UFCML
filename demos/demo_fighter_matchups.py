#!/usr/bin/env python3
"""
Demo of the Fighter Matchup Predictor
Shows how to predict hypothetical fights using fighter profiles
"""

import sys
sys.path.append('/Users/ralphfrancolini/UFCML')

from src.core.fighter_matchup_predictor import FighterMatchupPredictor

def demo_matchups():
    print("🥊 UFC FIGHTER MATCHUP PREDICTOR DEMO")
    print("=" * 50)
    print("Predict hypothetical fights using fighter profiles!")
    print("=" * 50)

    predictor = FighterMatchupPredictor()

    # Demo matchups to predict - realistic same-weight-class fights
    matchups = [
        ("Jon Jones", "Alex Pereira"),  # Both Light Heavyweight
        ("Conor McGregor", "Khabib Nurmagomedov"),  # Both Lightweight
        ("Amanda Nunes", "Valentina Shevchenko"),  # Different but closer weight classes
        ("Custom Fighter A", "Custom Fighter B")  # Both Middleweight
    ]

    print(f"\n🚀 Running {len(matchups)} hypothetical matchup predictions...")

    for i, (fighter_a, fighter_b) in enumerate(matchups, 1):
        print(f"\n{'='*60}")
        print(f"HYPOTHETICAL MATCHUP {i}: {fighter_a} vs {fighter_b}")
        print('='*60)

        result = predictor.predict_matchup(fighter_a, fighter_b)
        if result:
            predictor.display_matchup_prediction(result)
        else:
            print("❌ Unable to generate prediction for this matchup")

    print(f"\n✅ Demo complete!")
    print(f"\n📝 Key Features:")
    print("• Uses real fighter career statistics and profiles")
    print("• Simulates fight stats based on fighter styles")
    print("• No need to guess hypothetical fight numbers")
    print("• Accounts for fighting styles and matchup dynamics")
    print("• Provides confidence levels and model breakdowns")

    print(f"\n🎯 To use interactively:")
    print("Run: python3 fighter_matchup_predictor.py")
    print("Then just enter fighter names or numbers!")

if __name__ == "__main__":
    demo_matchups()