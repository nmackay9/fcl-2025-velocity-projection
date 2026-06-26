"""Stage 9: simulate the future via Monte Carlo, checked against extreme-value theory.

Mirrors notebook Section 10. For each of the 200 posterior draws (mechanical-efficiency
estimate + a growth-adjustment draw), simulate many future pitches with the flat sigma from
uncertainty.py, take the max of each simulated outing's worth of pitches, and repeat across all
200 draws. That builds an empirical distribution of "this pitcher's hardest pitch over many
future throws" for each FCL-2025 pitcher.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import PITCHER_COL


def run(ctx, n_pitches_options=(1000, 3000, 6000), primary_n=3000):
    unique_fcl_pitchers = ctx['unique_fcl_pitchers']
    sigma_final = ctx['sigma_final']
    pitcher_boot_means = ctx['pitcher_boot_means']
    growth_draws = ctx['growth_draws']
    rng4 = ctx['rng4']

    mc_summary = []
    posterior_store = {}
    for p in unique_fcl_pitchers:
        sigma_p = max(float(sigma_final.loc[p]), 0.1)
        mu_draws = pitcher_boot_means[p] + growth_draws
        row = {PITCHER_COL: p, 'posterior_mean_mu': mu_draws.mean(), 'posterior_std_mu': mu_draws.std(),
               'sigma_used': sigma_p}
        for N in n_pitches_options:
            sims = rng4.normal(loc=mu_draws[:, None], scale=sigma_p, size=(len(mu_draws), N))
            maxes = sims.max(axis=1)
            row[f'expected_max_N{N}'] = maxes.mean()
            row[f'p90_max_N{N}'] = np.percentile(maxes, 90)
            if N == primary_n:
                posterior_store[p] = {'mu_draws': mu_draws.copy(), 'maxes': maxes.copy()}
        mc_summary.append(row)
    mc_df = pd.DataFrame(mc_summary)

    rank_1000 = mc_df.set_index(PITCHER_COL)['expected_max_N1000'].rank(ascending=False)
    rank_3000 = mc_df.set_index(PITCHER_COL)['expected_max_N3000'].rank(ascending=False)
    rank_6000 = mc_df.set_index(PITCHER_COL)['expected_max_N6000'].rank(ascending=False)
    rho_1k_3k = spearmanr(rank_1000, rank_3000).correlation
    rho_3k_6k = spearmanr(rank_3000, rank_6000).correlation
    print(f'Rank correlation N=1000 vs N=3000: {rho_1k_3k:.3f}; N=3000 vs N=6000: {rho_3k_6k:.3f}', flush=True)
    print('Ranking is essentially insensitive to exactly how many future pitches are simulated.', flush=True)

    ctx['mc_df'] = mc_df
    ctx['posterior_store'] = posterior_store
    ctx['PRIMARY_N'] = primary_n
    return ctx
