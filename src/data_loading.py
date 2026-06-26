"""Stage 1: load the labeled (velocity-known) and FCL-2025 (target) datasets.

Mirrors notebook Section 1, "The Data".
"""
import numpy as np
import pandas as pd

from . import config
from .config import DATE_COL, FCL_PATH, LABELED_PATH, METRIC_DESC_PATH, PITCHER_COL


def run(ctx):
    metric_desc = pd.read_csv(METRIC_DESC_PATH)
    ctx['metric_desc_map'] = dict(zip(metric_desc['Column'], metric_desc['Description']))
    ctx['metric_units_map'] = dict(zip(metric_desc['Column'], metric_desc['Units'].fillna('')))

    df = pd.read_csv(LABELED_PATH)
    fcl = pd.read_csv(FCL_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    fcl[DATE_COL] = pd.to_datetime(fcl[DATE_COL])
    df['orig_order'] = np.arange(len(df))
    fcl['orig_order'] = np.arange(len(fcl))
    df['season'] = df[DATE_COL].dt.year
    df['level'] = np.where(df['season'] == 2024, 'FCL_2024', 'FSL_2025')

    print('Labeled (velocity known):', df.shape, ' FCL-2025 target (velocity unknown):', fcl.shape)
    print('Pitchers labeled:', df[PITCHER_COL].nunique(), ' Pitchers in FCL-2025 target:', fcl[PITCHER_COL].nunique())
    print('Overlap pitchers:', len(set(df[PITCHER_COL]) & set(fcl[PITCHER_COL])))

    raw_mechanics_cols = [c for c in df.columns if c in fcl.columns
                          and c not in [DATE_COL, PITCHER_COL, 'pitcher_handedness', 'orig_order']
                          and df[c].dtype != object]
    print('n raw mechanics columns shared by both files:', len(raw_mechanics_cols))

    ctx['df'] = df
    ctx['fcl'] = fcl
    ctx['raw_mechanics_cols'] = raw_mechanics_cols
    return ctx
