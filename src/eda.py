"""Stage 2: exploratory analysis -- what do the mechanics say about velocity?

Mirrors notebook Section 2. Produces figures 01-05 and 08-09, plus the pitch-count
overrepresentation numbers used later for inverse-pitch-count sample weighting.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import DATE_COL, PHI_BLUE, PHI_RED, PITCHER_COL, TARGET, savefig


def run(ctx):
    df = ctx['df']
    raw_mechanics_cols = ctx['raw_mechanics_cols']

    # --- Level / promotion confound (Figure 1) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df, x='level', y=TARGET, ax=axes[0], palette=[PHI_BLUE, PHI_RED])
    axes[0].set_title('Velocity by Level (2024 Rookie-Ball vs 2025 Single-A)')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Velocity (mph)')
    monthly = df.groupby(df[DATE_COL].dt.to_period('M'))[TARGET].mean()
    for lvl, g in df.groupby('level'):
        m = g.groupby(g[DATE_COL].dt.to_period('M'))[TARGET].mean()
        axes[1].plot(m.index.astype(str), m.values, marker='o', label=lvl)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_title('Mean Velocity by Month, Within Level')
    axes[1].set_ylabel('Velocity (mph)')
    savefig('01_level_confound.png')
    print(df.groupby('level')[TARGET].describe())

    season_counts = df.groupby(PITCHER_COL)['season'].nunique()
    dev_pitchers_idx = season_counts[season_counts > 1].index
    dev_compare = (
        df[df[PITCHER_COL].isin(dev_pitchers_idx)]
          .groupby([PITCHER_COL, 'season'])[TARGET].mean().unstack()
    )
    dev_compare['delta'] = dev_compare[2025] - dev_compare[2024]
    n_dev_pitchers = len(dev_compare)
    global_dev_delta = dev_compare['delta'].mean()
    print(f'{n_dev_pitchers} pitchers appear in both 2024 (rookie ball) and 2025 (Single-A); '
          f'mean within-pitcher gain = {global_dev_delta:.2f} mph')

    # --- Velocity distributions + correlation structure (Figures 2-4) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df[TARGET], kde=True, ax=axes[0], color=PHI_RED)
    axes[0].set_title('Pitch-Level Velocity Distribution')
    axes[0].set_xlabel('Velocity (mph)')
    pitcher_velo = df.groupby(PITCHER_COL)[TARGET].mean()
    sns.histplot(pitcher_velo, kde=True, ax=axes[1], color=PHI_BLUE)
    axes[1].set_title('Per-Pitcher Mean Velocity Distribution')
    axes[1].set_xlabel('Velocity (mph)')
    savefig('02_velocity_distribution.png')

    numeric_cols = [c for c in raw_mechanics_cols if c not in ['season']]
    pooled_corr = df[numeric_cols + [TARGET]].corr()[TARGET].drop(TARGET)
    pooled_corr_sorted = pooled_corr.reindex(pooled_corr.abs().sort_values(ascending=False).index)

    plt.figure(figsize=(10, 8))
    top20 = pooled_corr_sorted.head(20).index.tolist()
    sns.heatmap(df[top20 + [TARGET]].corr(), cmap='coolwarm', center=0, annot=False)
    plt.title('Velocity and Its Top 20 Most-Correlated Biomechanics')
    savefig('03_correlation_heatmap.png')

    df_centered = df.copy()
    df_centered[numeric_cols] = df_centered.groupby(PITCHER_COL)[numeric_cols].transform(lambda x: x - x.mean())
    within_corr = df_centered[numeric_cols + [TARGET]].corr()[TARGET].drop(TARGET)

    comp = pd.DataFrame({'across pitchers': pooled_corr_sorted, 'within a pitcher': within_corr}).reindex(
        pooled_corr_sorted.index).head(15)
    comp.plot(kind='barh', figsize=(9, 7), color=[PHI_RED, PHI_BLUE])
    plt.title('What Predicts Velocity Across Pitchers vs. Within One Pitcher\'s Own Outings')
    plt.xlabel('Correlation with velocity')
    plt.gca().invert_yaxis()
    savefig('04_pooled_vs_within_pitcher.png')

    # --- Pitch-count overrepresentation (Figure 5) ---
    pitch_counts = df.groupby(PITCHER_COL).size().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(pitch_counts, bins=30, ax=axes[0], color=PHI_BLUE)
    axes[0].set_title('Pitches Recorded per Pitcher')
    axes[0].set_xlabel('pitches')
    cum_share = pitch_counts.sort_values(ascending=False).cumsum() / pitch_counts.sum()
    axes[1].plot(range(1, len(cum_share) + 1), cum_share.values, color=PHI_RED)
    axes[1].set_xlabel('Pitcher rank, by volume')
    axes[1].set_ylabel('Cumulative share of all pitches')
    axes[1].set_title('How Concentrated Is the Sample?')
    savefig('05_pitcher_overrepresentation.png')
    top10_share = cum_share.iloc[9]
    print(f'Top 10 of {len(pitch_counts)} pitchers account for {top10_share:.1%} of all recorded pitches')

    ctx['pitch_counts'] = pitch_counts
    ctx['dev_compare'] = dev_compare
    ctx['n_dev_pitchers'] = n_dev_pitchers
    ctx['global_dev_delta'] = global_dev_delta
    return ctx


def run_fatigue_and_development(ctx):
    """Within-outing fatigue drift and year-over-year development (Figures 8-9).

    Split out from `run()` because it depends on `velo_drift_df`, which is computed in
    feature_engineering.py (it reuses the same `compute_outing_drift` helper as the arm-slot
    and stride-length drift features). Call this after feature_engineering.run().
    """
    df = ctx['df']
    pitch_counts = ctx['pitch_counts']
    dev_compare = ctx['dev_compare']
    global_dev_delta = ctx['global_dev_delta']
    velo_drift_df = ctx['velo_drift_df']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(velo_drift_df, bins=25, ax=axes[0], color=PHI_RED)
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set_title('Within-Outing Velocity Drift (mph per pitch)')
    axes[1].scatter(pitch_counts.reindex(velo_drift_df.index), velo_drift_df, alpha=0.5, color=PHI_BLUE)
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set_xlabel('Total pitches thrown (season volume)')
    axes[1].set_ylabel('Velocity drift (mph per pitch, within an outing)')
    axes[1].set_title('Workload vs In-Outing Fatigue')
    savefig('08_fatigue_drift.png')
    print('Mean within-outing velocity drift:', round(velo_drift_df.mean(), 4), 'mph/pitch (essentially flat)')

    plt.figure(figsize=(9, 6))
    dev_plot = dev_compare.sort_values('delta', ascending=False)
    colors = [PHI_RED if v < 0 else PHI_BLUE for v in dev_plot['delta']]
    plt.bar(range(len(dev_plot)), dev_plot['delta'], color=colors)
    plt.axhline(global_dev_delta, color='black', linestyle='--', label=f'mean = {global_dev_delta:.2f} mph')
    plt.xlabel('Pitcher (sorted by year-over-year change)')
    plt.ylabel('2025 minus 2024 mean velocity (mph)')
    plt.title('Year-over-Year Development: the Few Pitchers Seen at Both Levels')
    plt.legend()
    savefig('09_dual_year_development.png')
    print('done with exploratory section')
    return ctx
