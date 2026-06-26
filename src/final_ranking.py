"""Stage 14: the final ranking.

Mirrors notebook Section 15. Sorts every FCL-2025 pitcher by consensus rank, writes the full
ranking to CSV, and plots the top 15 (posterior mean mechanical efficiency vs. expected/p90
maximum velocity).
"""
import os

import matplotlib.pyplot as plt

from .config import OUTPUT_DIR, PHI_RED, PHI_BLUE, PITCHER_COL, safe_to_csv, savefig


def run(ctx):
    convergence_df = ctx['convergence_df']
    PRIMARY_N = ctx['PRIMARY_N']

    final_ranking = convergence_df.sort_values('consensus_rank').reset_index(drop=True)
    final_ranking['rank'] = final_ranking.index + 1
    out_csv = os.path.join(OUTPUT_DIR, 'fcl_2025_pitcher_rankings.csv')
    out_csv = safe_to_csv(final_ranking, out_csv, index=False)

    top15 = final_ranking.head(15)[['rank', PITCHER_COL, f'expected_max_N{PRIMARY_N}', f'p90_max_N{PRIMARY_N}',
                                     'methodA_rank', 'methodB_rank', 'methodC_rank', 'consensus_rank',
                                     'mech_archetype', 'handedness']]
    print(top15, flush=True)

    plot_top15 = final_ranking.head(15).sort_values('rank', ascending=False)
    plot_min = final_ranking['methodA_score'].min() - 1
    plt.figure(figsize=(9, 8))
    plt.barh(plot_top15[PITCHER_COL].astype(str), plot_top15[f'expected_max_N{PRIMARY_N}'] - plot_min,
             left=plot_min, color=PHI_RED,
             xerr=[plot_top15[f'expected_max_N{PRIMARY_N}'] - plot_top15['methodA_score'],
                   plot_top15[f'p90_max_N{PRIMARY_N}'] - plot_top15[f'expected_max_N{PRIMARY_N}']])
    plt.scatter(plot_top15['methodA_score'], plot_top15[PITCHER_COL].astype(str), color=PHI_BLUE, zorder=5,
                label='posterior mean (current mechanical efficiency)', s=40)
    plt.xlim(plot_min, final_ranking[f'p90_max_N{PRIMARY_N}'].max() + 0.5)
    plt.xlabel(f'Velocity (mph) -- blue dot = posterior mean, bar end = expected max over {PRIMARY_N} pitches')
    plt.ylabel('pitcher_id')
    plt.title('Top 15 by Consensus Rank (#1 at top)')
    plt.legend(loc='lower right')
    savefig('24_final_ranking.png')

    ctx['final_ranking'] = final_ranking
    return ctx
