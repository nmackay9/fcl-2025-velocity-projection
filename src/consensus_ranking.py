"""Stage 11: combine Methods A, B, and C into one consensus rank for every FCL-2025 pitcher.

Mirrors notebook Section 12. None of the three methods is treated as the answer on its own --
each has a different blind spot. Consensus rank is the simple average of each method's
independently-computed rank. Also includes the Gumbel goodness-of-fit check (run on the actual
#1 consensus pitcher, not whoever has the single highest one-method score) and a follow-up
check on why two mechanical archetypes recur near the top of the ranking.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gumbel_r, kstest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import PHI_BLUE, PHI_RED, PITCHER_COL, savefig


def run_consensus(ctx):
    df = ctx['df']
    fcl = ctx['fcl']
    stable_features = ctx['stable_features']
    mc_df = ctx['mc_df']
    posterior_store = ctx['posterior_store']
    PRIMARY_N = ctx['PRIMARY_N']

    methodA_fcl = mc_df.set_index(PITCHER_COL)['posterior_mean_mu']

    train_agg_full = df.groupby(PITCHER_COL)[stable_features].mean()
    train_agg_y_full = df.groupby(PITCHER_COL)['velocity'].mean().reindex(train_agg_full.index)
    fcl_agg_full = fcl.groupby(PITCHER_COL)[stable_features].mean()

    imp_f = SimpleImputer(strategy='median')
    scaler_f = StandardScaler()
    train_agg_full_scaled = scaler_f.fit_transform(imp_f.fit_transform(train_agg_full))
    fcl_agg_full_scaled = scaler_f.transform(imp_f.transform(fcl_agg_full))

    nn_full = NearestNeighbors(n_neighbors=10)
    nn_full.fit(train_agg_full_scaled)
    dist_fcl, idx_fcl = nn_full.kneighbors(fcl_agg_full_scaled)
    train_y_full_arr = train_agg_y_full.values
    methodB_fcl = pd.Series(train_y_full_arr[idx_fcl].mean(axis=1), index=fcl_agg_full.index)
    methodB_fcl_confidence = pd.Series(dist_fcl.mean(axis=1), index=fcl_agg_full.index, name='avg_neighbor_distance')

    n_tr_full = len(train_agg_y_full)
    pi, pj = np.meshgrid(np.arange(n_tr_full), np.arange(n_tr_full), indexing='ij')
    pi, pj = pi.ravel(), pj.ravel()
    keep_full = pi != pj
    pi, pj = pi[keep_full], pj[keep_full]
    clf3_full = LogisticRegression(max_iter=2000)
    clf3_full.fit(train_agg_full_scaled[pi] - train_agg_full_scaled[pj],
                  (train_y_full_arr[pi] > train_y_full_arr[pj]).astype(int))
    win_rates_full = [clf3_full.predict_proba(v[None, :] - train_agg_full_scaled)[:, 1].mean()
                       for v in fcl_agg_full_scaled]
    methodC_fcl = pd.Series(win_rates_full, index=fcl_agg_full.index)

    convergence_df = pd.DataFrame({
        'methodA_score': methodA_fcl, 'methodB_score': methodB_fcl, 'methodC_score': methodC_fcl,
        'methodB_avg_neighbor_distance': methodB_fcl_confidence,
    }).dropna()
    for col in ['methodA_score', 'methodB_score', 'methodC_score']:
        convergence_df[col.replace('_score', '_rank')] = convergence_df[col].rank(ascending=False)
    convergence_df['consensus_rank'] = convergence_df[['methodA_rank', 'methodB_rank', 'methodC_rank']].mean(axis=1)
    convergence_df = convergence_df.merge(
        fcl.groupby(PITCHER_COL).agg(handedness=('pitcher_handedness', lambda x: x.mode().iloc[0]),
                                      mech_archetype=('mech_archetype', lambda x: x.value_counts().idxmax())),
        left_index=True, right_index=True
    )
    convergence_df = convergence_df.sort_values('consensus_rank').reset_index().rename(columns={'index': PITCHER_COL})
    convergence_df = convergence_df.merge(mc_df[[PITCHER_COL, f'expected_max_N{PRIMARY_N}', f'p90_max_N{PRIMARY_N}']],
                                           on=PITCHER_COL, how='left')

    rank_corr_matrix = convergence_df[['methodA_rank', 'methodB_rank', 'methodC_rank']].corr(method='spearman')
    print('FCL-2025 inter-method rank correlation:')
    print(rank_corr_matrix.round(3), flush=True)

    top10_sets = {m: set(convergence_df.nsmallest(10, f'{m}_rank')[PITCHER_COL])
                  for m in ['methodA', 'methodB', 'methodC']}
    overlap_AB = len(top10_sets['methodA'] & top10_sets['methodB'])
    overlap_AC = len(top10_sets['methodA'] & top10_sets['methodC'])
    overlap_BC = len(top10_sets['methodB'] & top10_sets['methodC'])
    overlap_all3 = len(top10_sets['methodA'] & top10_sets['methodB'] & top10_sets['methodC'])
    print(f'Top-10 overlap: A/B={overlap_AB}, A/C={overlap_AC}, B/C={overlap_BC}, all three={overlap_all3}',
          flush=True)

    top3_ids = convergence_df.sort_values('consensus_rank').head(3)[PITCHER_COL].tolist()
    print('Top 3 by consensus rank (the same top 3 as the final ranking):', top3_ids, flush=True)
    for pid in top3_ids:
        row = convergence_df[convergence_df[PITCHER_COL] == pid].iloc[0]
        print(f"Pitcher {pid}: A rank={row['methodA_rank']:.0f}, B rank={row['methodB_rank']:.0f}, "
              f"C rank={row['methodC_rank']:.0f}, consensus={row['consensus_rank']:.1f}, "
              f"comparable-pitcher distance={row['methodB_avg_neighbor_distance']:.2f}", flush=True)

    archetype_2_share_top10 = (convergence_df.head(10)['mech_archetype'] ==
                                convergence_df.head(10)['mech_archetype'].mode().iloc[0]).sum()
    print(f'Top 10 by consensus: {archetype_2_share_top10}/10 share the single most-common archetype', flush=True)

    # Gumbel goodness-of-fit check, run on the actual #1 consensus pitcher (not whoever happens
    # to have the highest single-method expected max).
    example_pid = top3_ids[0]
    ex_maxes = posterior_store[example_pid]['maxes']
    loc_ex, scale_ex = gumbel_r.fit(ex_maxes)
    ks_stat, ks_p = kstest(ex_maxes, 'gumbel_r', args=(loc_ex, scale_ex))
    plt.figure(figsize=(8, 5))
    sns.histplot(ex_maxes, stat='density', color=PHI_BLUE, alpha=0.5, bins=25)
    xs_ex = np.linspace(ex_maxes.min(), ex_maxes.max(), 200)
    plt.plot(xs_ex, gumbel_r.pdf(xs_ex, loc_ex, scale_ex), color=PHI_RED, linewidth=2, label='fitted Gumbel curve')
    ks_verdict = 'fails to reject' if ks_p >= 0.05 else 'rejects'
    plt.title(f'Gumbel Fit Check: Pitcher {example_pid} (#1 by consensus)\n'
              f'KS test {ks_verdict} fit at p={ks_p:.3f} (alpha=0.05)')
    plt.xlabel('Simulated hardest pitch over next 3,000 throws (mph)')
    plt.legend()
    savefig('18_gumbel_validation.png')
    print(f'Gumbel KS test for pitcher {example_pid}: stat={ks_stat:.4f}, p={ks_p:.4f}', flush=True)

    highlight_ids = top3_ids
    plt.figure(figsize=(10, 9))
    methods_x = ['Method A\n(model prediction)', 'Method B\n(comparable pitchers)', 'Method C\n(pairwise comparison)']
    palette = sns.color_palette('tab10', n_colors=len(highlight_ids))
    for _, row in convergence_df.iterrows():
        pid = row[PITCHER_COL]
        ranks = [row['methodA_rank'], row['methodB_rank'], row['methodC_rank']]
        if pid in highlight_ids:
            c = palette[highlight_ids.index(pid)]
            plt.plot(methods_x, ranks, marker='o', color=c, linewidth=2.5, label=f'Pitcher {pid}', zorder=3)
        else:
            plt.plot(methods_x, ranks, marker='o', color='lightgray', linewidth=1, alpha=0.5, zorder=1)
    plt.gca().invert_yaxis()
    plt.ylabel('Rank (1 = projected hardest thrower)')
    plt.title('Do Three Independently-Built Ranking Methods Agree on FCL-2025?')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    savefig('20_ranking_method_agreement.png')

    ctx['convergence_df'] = convergence_df
    ctx['top3_ids'] = top3_ids
    return ctx


def run_archetype_followup(ctx):
    """Why archetypes 0 and 2 recur at the top of the ranking -- a real mechanical signal
    (archetype 0) vs. simply being the largest group in the cohort (archetype 2)."""
    fcl = ctx['fcl']
    mc_df = ctx['mc_df']
    convergence_df = ctx['convergence_df']
    raw_mechanics_cols = ctx['raw_mechanics_cols']

    archetype_lookup_fcl = fcl.groupby(PITCHER_COL)['mech_archetype'].agg(lambda x: x.value_counts().idxmax())
    arche_mc = mc_df.set_index(PITCHER_COL).join(archetype_lookup_fcl)
    arche_profile = arche_mc.groupby('mech_archetype').agg(
        n_pitchers=('posterior_mean_mu', 'size'),
        mean_posterior_mu=('posterior_mean_mu', 'mean'),
        mean_posterior_uncertainty=('posterior_std_mu', 'mean'),
    ).sort_values('mean_posterior_mu', ascending=False)
    print('FCL-2025: posterior mean efficiency and model uncertainty, by mechanical archetype:')
    print(arche_profile.round(3), flush=True)

    top15_archetype_counts = convergence_df.head(15)['mech_archetype'].value_counts()
    print('\nArchetype counts among the top 15 by consensus rank (71 FCL-2025 pitchers total):')
    print(top15_archetype_counts, flush=True)

    archetype_pop_share = archetype_lookup_fcl.value_counts()
    print('\nArchetype counts across the full FCL-2025 cohort:')
    print(archetype_pop_share, flush=True)

    feat_compare = fcl.groupby('mech_archetype')[raw_mechanics_cols].mean()
    fcl_overall_mean = fcl[raw_mechanics_cols].mean()
    fcl_overall_std = fcl[raw_mechanics_cols].std()
    archetype_z = (feat_compare - fcl_overall_mean) / fcl_overall_std
    top_diff_0_2 = (archetype_z.loc['0'] - archetype_z.loc['2']).sort_values(key=lambda s: -s.abs()).head(8)
    print('\nLargest mechanical differences between archetype 0 and archetype 2 (FCL-2025 population z-units):')
    print(top_diff_0_2, flush=True)

    plt.figure(figsize=(9, 6))
    top_diff_0_2.sort_values().plot(kind='barh',
                                     color=[PHI_RED if v > 0 else PHI_BLUE for v in top_diff_0_2.sort_values()])
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel('Archetype 0 minus archetype 2 (FCL-2025 population z-score units)')
    plt.title('What Mechanically Separates Archetype 0 (Most Overrepresented at the Top)\n'
              'from Archetype 2 (Largest Archetype Overall)')
    savefig('20b_archetype_0_vs_2.png')
    return ctx


def run(ctx):
    ctx = run_consensus(ctx)
    ctx = run_archetype_followup(ctx)
    return ctx
