"""Shared constants, paths, and small helpers used by every pipeline stage."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
FIG_DIR = os.path.join(REPO_ROOT, 'figures')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELED_PATH = os.path.join(DATA_DIR, 'velo_and_mechanics.csv')
FCL_PATH = os.path.join(DATA_DIR, 'fcl_mechanics_2025.csv')
METRIC_DESC_PATH = os.path.join(DATA_DIR, 'metric_descriptions.csv')

TARGET = 'velocity'
PITCHER_COL = 'pitcher_id'
DATE_COL = 'date'

sns.set_theme(style='whitegrid')
PHI_RED = 'crimson'
PHI_BLUE = 'navy'
PHI_PURPLE = 'rebeccapurple'
PALETTE = sns.color_palette([PHI_RED, PHI_BLUE, PHI_PURPLE, 'darkseagreen', 'goldenrod', 'dimgray', 'teal'])


def safe_to_csv(dataframe, path, **kwargs):
    """Write a CSV, falling back to a different filename if the target is locked (e.g. open in Excel)."""
    try:
        dataframe.to_csv(path, **kwargs)
        print('saved', path, flush=True)
        return path
    except PermissionError:
        fallback = path.replace('.csv', '_OUTPUT.csv')
        dataframe.to_csv(fallback, **kwargs)
        print(f'WARNING: {path} is locked by another process (likely open in an editor/preview tab) -- '
              f'wrote to {fallback} instead. Close the locked file and rename.', flush=True)
        return fallback


def savefig(name):
    """Save the current matplotlib figure into FIG_DIR and close it."""
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('saved', path, flush=True)
