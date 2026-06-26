"""Stage 13: put a number on the story -- explicit predictions for the top 3 by consensus rank.

Mirrors notebook Section 14. Ranks by the MEAN of each pitcher's simulated-maximum distribution
("expected max"), not by the single hardest pitch any one simulation happened to produce (the
mode) -- the whole reason simulation.py runs a full Monte Carlo instead of just reporting the
highest number observed in a handful of draws.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gumbel_r

from .config import PHI_BLUE, PHI_PURPLE, PHI_RED, PITCHER_COL, savefig


def run(ctx):
    top3_ids = ctx['top3_ids']
    posterior_store = ctx['posterior_store']
    convergence_df = ctx['convergence_df']
    PRIMARY_N = ctx['PRIMARY_N']

    gumbel_predictions = {}
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    colors3 = [PHI_RED, PHI_BLUE, PHI_PURPLE]
    for i, pid in enumerate(top3_ids):
        mu_draws = posterior_store[pid]['mu_draws']
        maxes = posterior_store[pid]['maxes']
        rank_i = convergence_df.loc[convergence_df[PITCHER_COL] == pid, 'consensus_rank'].iloc[0]

        sns.histplot(mu_draws, kde=True, ax=axes[0, i], color=colors3[i], stat='density')
        axes[0, i].axvline(mu_draws.mean(), color='black', linestyle='--')
        axes[0, i].set_title(f'Pitcher {pid} (consensus rank {rank_i:.1f})\nCurrent mechanical efficiency')
        axes[0, i].set_xlabel('mph')

        loc, scale = gumbel_r.fit(maxes)
        pred_mode = loc
        pred_p5, pred_p95 = gumbel_r.ppf([0.05, 0.95], loc, scale)
        pred_mean = maxes.mean()
        gumbel_predictions[pid] = {'mode': pred_mode, 'mean': pred_mean, 'p5': pred_p5, 'p95': pred_p95}

        ax2 = axes[1, i]
        sns.histplot(maxes, stat='density', ax=ax2, color=colors3[i], alpha=0.5, bins=25)
        xs = np.linspace(min(maxes.min(), pred_p5) - 0.2, max(maxes.max(), pred_p95) + 0.2, 200)
        ax2.plot(xs, gumbel_r.pdf(xs, loc, scale), color='black', linewidth=2, label='fitted Gumbel curve')
        ax2.axvspan(pred_p5, pred_p95, color='gray', alpha=0.2, label='90% interval')
        ax2.axvline(pred_mode, color='gray', linestyle='-', linewidth=1.5)
        ax2.axvline(pred_mean, color='black', linestyle=':', linewidth=2.5)
        y_top = ax2.get_ylim()[1]
        ax2.annotate(f'PREDICTED MAX: {pred_mean:.1f} mph',
                     xy=(pred_mean, y_top * 0.9), xytext=(pred_mean, y_top * 1.15),
                     ha='center', fontsize=9, fontweight='bold', color='black',
                     arrowprops=dict(arrowstyle='-', color='black', linewidth=1.2),
                     annotation_clip=False)
        ax2.annotate(f'mode: {pred_mode:.1f}', xy=(pred_mode, y_top * 0.55), xytext=(pred_mode, y_top * 0.55),
                     ha='center', fontsize=8, color='dimgray')
        ax2.annotate(f'90% interval:\n[{pred_p5:.1f}, {pred_p95:.1f}] mph', xy=(0.02, 0.97),
                     xycoords='axes fraction', ha='left', va='top', fontsize=7.5, color='dimgray')
        ax2.set_ylim(top=y_top * 1.3)
        ax2.set_title(f'Pitcher {pid}: simulated hardest pitch over {PRIMARY_N} throws')
        ax2.set_xlabel('mph')
        ax2.legend(fontsize=7.5, loc='center right', framealpha=0.9)
    fig.suptitle('Top 3 by Consensus Rank: Projected Ceiling Distributions', fontsize=13, y=1.02)
    savefig('23_top3_explicit_predictions.png')

    for pid, g in gumbel_predictions.items():
        print(f"Pitcher {pid}: most likely single hardest pitch = {g['mode']:.2f} mph "
              f"(90% interval {g['p5']:.2f}-{g['p95']:.2f}), expected max = {g['mean']:.2f} mph", flush=True)

    ctx['gumbel_predictions'] = gumbel_predictions
    return ctx
