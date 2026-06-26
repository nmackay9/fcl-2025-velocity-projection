"""Stage 12: case studies -- what makes the top consensus pitchers stand out, mechanically.

Mirrors notebook Section 13. Z-scores each top pitcher's raw mechanics against the full
FCL-2025 peer population to surface, in plain biomechanical terms, what actually distinguishes
each of the top 5 consensus-ranked pitchers from everyone else in the cohort.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import PHI_BLUE, PHI_PURPLE, PHI_RED, PITCHER_COL, savefig


def run(ctx):
    fcl = ctx['fcl']
    convergence_df = ctx['convergence_df']
    raw_mechanics_cols = ctx['raw_mechanics_cols']

    profile_pitcher = convergence_df.sort_values('consensus_rank').iloc[0][PITCHER_COL]
    fcl_profile = fcl.groupby(PITCHER_COL)[raw_mechanics_cols].mean()
    fcl_pop_mean = fcl_profile.mean()
    fcl_pop_std = fcl_profile.std()
    p_vals = fcl_profile.loc[profile_pitcher]
    p_z = ((p_vals - fcl_pop_mean) / fcl_pop_std).sort_values(key=lambda s: -s.abs())
    top_p_feats = p_z.head(12)

    plt.figure(figsize=(9, 7))
    top_p_feats.sort_values().plot(kind='barh', color=[PHI_RED if v > 0 else PHI_BLUE
                                                        for v in top_p_feats.sort_values().values])
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel(f'Pitcher {profile_pitcher}, in FCL-2025 population SD units (z-score)')
    plt.title(f'Pitcher {profile_pitcher} Mechanical Profile vs FCL-2025 Peers\n'
              f'red = above peer average, blue = below')
    savefig('21_top_pitcher_profile.png')

    profile_table = top_p_feats.rename('z_score').to_frame()
    profile_table['direction'] = np.where(profile_table['z_score'] > 0, 'above peer average', 'below peer average')
    print(f'Pitcher {profile_pitcher} top deviations from {len(fcl_profile)} FCL-2025 peers:')
    print(profile_table.round(2), flush=True)

    # --- Top 5 consensus pitchers' two largest standout traits, scatter ---
    top5_consensus = convergence_df.sort_values('consensus_rank').head(5)[PITCHER_COL].tolist()
    compare_pitchers = [profile_pitcher] + [p for p in top5_consensus if p != profile_pitcher]
    star_colors = [PHI_RED, PHI_BLUE, PHI_PURPLE, 'goldenrod', 'darkseagreen']
    print('Top 5 by consensus rank (case-study comparison set):', compare_pitchers, flush=True)

    # Pick the #1 consensus pitcher's own two largest-magnitude deviations as the scatter axes,
    # rather than a hardcoded pair, so this plot always highlights whatever actually defines the
    # current top-ranked pitcher's profile, even if that profile changes across reruns.
    scatter_x, scatter_y = top_p_feats.index[0], top_p_feats.index[1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(fcl_profile[scatter_x], fcl_profile[scatter_y], alpha=0.4, color='lightgray', s=50,
               label='other FCL-2025 pitchers', zorder=1)
    for pid, c in zip(compare_pitchers, star_colors):
        ax.scatter(fcl_profile.loc[pid, scatter_x], fcl_profile.loc[pid, scatter_y],
                   color=c, s=220, marker='*', edgecolor='black', linewidth=1, zorder=5,
                   label=f'pitcher {pid}' + (' (consensus rank 1)' if pid == profile_pitcher else ''))
    ax.set_xlabel(f'{scatter_x.replace("_", " ").capitalize()} (deg)')
    ax.set_ylabel(f'{scatter_y.replace("_", " ").capitalize()} (deg)')
    ax.set_title(f'Top 5 Consensus-Ranked Pitchers vs FCL-2025 Peers\n'
                f'on Pitcher {profile_pitcher}\'s Two Largest Standout Traits')
    ax.legend()
    savefig('22_top_pitcher_standout_scatter.png')
    print(f'Scatter axes chosen from pitcher {profile_pitcher}\'s own top deviations: {scatter_x}, {scatter_y}',
          flush=True)

    # --- Side-by-side heatmap of all 5 pitchers' top standout traits ---
    z_all = (fcl_profile.loc[compare_pitchers] - fcl_pop_mean) / fcl_pop_std
    union_feats = set()
    for p in compare_pitchers:
        union_feats.update(z_all.loc[p].sort_values(key=lambda s: -s.abs()).head(6).index)
    union_feats = sorted(union_feats, key=lambda f: -z_all[f].abs().mean())
    heat_data = z_all[union_feats].T
    heat_data.columns = [f'Pitcher {p}' for p in heat_data.columns]

    plt.figure(figsize=(10, 0.45 * len(union_feats) + 2))
    sns.heatmap(heat_data, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
                cbar_kws={'label': 'z-score vs FCL-2025 peers'})
    plt.xticks(rotation=0)
    plt.xlabel('Top 5 consensus-ranked FCL-2025 pitchers')
    plt.ylabel('Mechanical feature')
    plt.title('What Stands Out for the Top 5 Consensus-Ranked Pitchers\n'
              '(each cell: standard deviations from FCL-2025 peer average)')
    plt.tight_layout()
    savefig('22b_top_consensus_pitchers_heatmap.png')
    print('Top consensus pitchers compared:', compare_pitchers, flush=True)
    print(heat_data.round(2), flush=True)

    ctx['profile_pitcher'] = profile_pitcher
    ctx['fcl_profile'] = fcl_profile
    ctx['top_p_feats'] = top_p_feats
    return ctx
