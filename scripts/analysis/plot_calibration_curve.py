#!/usr/bin/env python3
"""Plot calibration curve and confidence histogram for the enhanced UFC model."""
import argparse
import os
import pickle
import sys
from pathlib import Path

# Determine project root (two levels up from this script)
project_root = Path(__file__).resolve().parents[2]

# Ensure matplotlib has a writable config directory before importing it
default_mpl_dir = Path(os.environ.get('MPLCONFIGDIR', project_root / '.matplotlib_cache'))
default_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(default_mpl_dir)

# Make sure project root (containing core package) is on sys.path for unpickling
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_calibration_info(model_path: Path):
    with model_path.open('rb') as f:
        data = pickle.load(f)
    info = data.get('calibration_info')
    if not info:
        raise ValueError("No calibration information found in model artifact")
    return info


def plot_calibration(info, output: Path | None):
    frac_pos = np.array(info['calibration_curve_frac_pos'])
    mean_pred = np.array(info['calibration_curve_mean_pred'])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, marker='o', label=f"Model ({info.get('method', 'unknown')})")

    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed win rate')
    ax.set_title('Calibration Curve')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    if output:
        fig.savefig(output, dpi=120, bbox_inches='tight')
        print(f"✅ Saved calibration plot to {output}")
    else:
        plt.show()


def plot_confidence_hist(info, output: Path | None):
    bins = np.array(info.get('calibration_bins'))
    counts = np.array(info.get('bin_counts')) if info.get('bin_counts') is not None else None

    if bins is None or counts is None:
        raise ValueError('Calibration bins/counts not available in model artifact')

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bin_centers, counts, width=bins[1] - bins[0], alpha=0.7, edgecolor='k')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Validation fights in bin')
    ax.set_title('Confidence Distribution (validation split)')

    if output:
        fig.savefig(output, dpi=120, bbox_inches='tight')
        print(f"✅ Saved confidence distribution to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/enhanced_ufc_random_forest.pkl',
                        help='Path to calibrated model artifact')
    parser.add_argument('--calibration-plot', default=None,
                        help='Optional output path for calibration curve PNG')
    parser.add_argument('--confidence-plot', default=None,
                        help='Optional output path for confidence histogram PNG')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    info = load_calibration_info(model_path)

    if args.calibration_plot or args.confidence_plot:
        if args.calibration_plot:
            plot_calibration(info, Path(args.calibration_plot))
        if args.confidence_plot:
            plot_confidence_hist(info, Path(args.confidence_plot))
    else:
        plot_calibration(info, None)
        plot_confidence_hist(info, None)


if __name__ == '__main__':
    main()
