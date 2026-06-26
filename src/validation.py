"""Stage 10: validating a ranking is not the same as validating a regression.

Mirrors notebook Section 11. Three structurally different ranking methods are scored
out-of-fold via GroupKFold: Method A (the XGBoost model), Method B (10 nearest mechanically
comparable labeled pitchers), and Method C (a pairwise Bradley-Terry-style classifier). Spearman
rank correlation and pairwise concordance are computed for each, with a permutation test and
bootstrap confidence intervals, plus a "close pairs only" concordance check on the hardest 25%
of comparisons.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import OUTPUT_DIR, PHI_BLUE, PHI_PURPLE, PHI_RED, PITCHER_COL, RANDOM_STATE, TARGET, safe_to_csv, savefig
import os


def pairwise_concordance(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(actual)
    diff_actual = actual[:, None] - actual[None, :]
    diff_pred = pred[:, None] - pred[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    comparable = upper & (diff_actual != 0)
    concordant = (np.sign(diff_actual) == np.sign(diff_pred))
    return concordant[comparable].mean(), comparable.sum()


def close_pairs_concordance(actual, pred, gap_threshold):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(actual)
    diff_actual = actual[:, None] - actual[None, :]
    diff_pred = pred[:, None] - pred[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    abs_gap = np.abs(diff_actual)
    comparable = upper & (abs_gap > 0) & (abs_gap <= gap_threshold)
    concordant = (np.sign(diff_actual) == np.sign(diff_pred))
    return concordant[comparable].mean(), comparable.sum()


def permutation_test_spearman(actual, pred, n_perm=2000, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    observed = spearmanr(actual, pred).correlation
    null = np.empty(n_perm)
    actual_arr = np.asarray(actual, dtype=float)
    for i in range(n_perm):
        null[i] = spearmanr(rng.permutation(actual_arr), pred).correlation
    p_value = (np.sum(np.abs(null) >= np.abs(observed)) + 1) / (n_perm + 1)
    return observed, null, p_value


def run(ctx, n_boot_valid=2000):
    df = ctx['df']
    y = ctx['y']
    stable_features = ctx['stable_features']
    sample_weight = ctx['sample_weight']
    best_xgb_params = ctx['best_xgb_params']

    gkf = GroupKFold(n_splits=5)
    groups_arr = df[PITCHER_COL].values
    oof_records = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups_arr)):
        val_pids = df.iloc[val_idx][PITCHER_COL].unique()
        arch_enc_fold = df.iloc[train_idx].groupby('mech_archetype')[TARGET].mean()
        global_train_mean = df.iloc[train_idx][TARGET].mean()

        X_fold = df[stable_features].copy()
        X_fold['archetype_target_enc'] = df['mech_archetype'].map(arch_enc_fold).fillna(global_train_mean)
        X_tr, X_val = X_fold.iloc[train_idx], X_fold.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        w_tr = sample_weight.iloc[train_idx]

        imp1 = SimpleImputer(strategy='median')
        X_tr_imp = imp1.fit_transform(X_tr)
        m1 = XGBRegressor(**best_xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
        m1.fit(X_tr_imp, y_tr, sample_weight=w_tr.values)
        val_pred1 = pd.Series(m1.predict(imp1.transform(X_val)), index=X_val.index)
        methodA_pitcher = val_pred1.groupby(df.loc[X_val.index, PITCHER_COL]).mean()

        train_agg = X_tr.groupby(df.loc[X_tr.index, PITCHER_COL]).mean()
        train_agg_y = y_tr.groupby(df.loc[X_tr.index, PITCHER_COL]).mean().reindex(train_agg.index)
        val_agg = X_val.groupby(df.loc[X_val.index, PITCHER_COL]).mean()

        imp2 = SimpleImputer(strategy='median')
        scaler2 = StandardScaler()
        train_agg_scaled = scaler2.fit_transform(imp2.fit_transform(train_agg[stable_features]))
        val_agg_scaled = scaler2.transform(imp2.transform(val_agg[stable_features]))

        nn = NearestNeighbors(n_neighbors=min(10, len(train_agg)))
        nn.fit(train_agg_scaled)
        dist, idx = nn.kneighbors(val_agg_scaled)
        train_y_arr = train_agg_y.values
        methodB_pitcher = pd.Series(train_y_arr[idx].mean(axis=1), index=val_agg.index)

        tr_X, tr_y = train_agg_scaled, train_agg_y.values
        n_tr = len(tr_y)
        pi, pj = np.meshgrid(np.arange(n_tr), np.arange(n_tr), indexing='ij')
        pi, pj = pi.ravel(), pj.ravel()
        keep = pi != pj
        pi, pj = pi[keep], pj[keep]
        clf3 = LogisticRegression(max_iter=2000)
        clf3.fit(tr_X[pi] - tr_X[pj], (tr_y[pi] > tr_y[pj]).astype(int))
        win_rates = [clf3.predict_proba(v[None, :] - tr_X)[:, 1].mean() for v in val_agg_scaled]
        methodC_pitcher = pd.Series(win_rates, index=val_agg.index)

        for pid in val_pids:
            oof_records.append({
                PITCHER_COL: pid, 'actual_velocity': df.loc[df[PITCHER_COL] == pid, TARGET].mean(),
                'methodA': methodA_pitcher.get(pid, np.nan), 'methodB': methodB_pitcher.get(pid, np.nan),
                'methodC': methodC_pitcher.get(pid, np.nan),
            })
        print(f'fold {fold + 1}/5 done', flush=True)

    oof_df = pd.DataFrame(oof_records).dropna()
    print(f'OOF validation set: {len(oof_df)} pitchers', flush=True)

    # "Close pairs" = the hardest 25% of comparisons to get right: pitcher pairs whose actual
    # velocity gap is small, where a ranking method has to do real discriminating work rather
    # than just separating two obviously-different throwers.
    all_actual = oof_df['actual_velocity'].values
    n_all = len(all_actual)
    diff_all_abs = np.abs(all_actual[:, None] - all_actual[None, :])
    upper_all = np.triu(np.ones((n_all, n_all), dtype=bool), k=1)
    close_gap_mph = float(np.quantile(diff_all_abs[upper_all & (diff_all_abs > 0)], 0.25))
    print(f'Close-pairs threshold (25th percentile of actual pairwise velocity gaps): {close_gap_mph:.2f} mph',
          flush=True)

    validation_rows = []
    for col, label in [('methodA', 'Method A: model prediction'), ('methodB', 'Method B: comparable pitchers'),
                        ('methodC', 'Method C: pairwise comparison')]:
        rho, null_dist, p_val = permutation_test_spearman(oof_df['actual_velocity'], oof_df[col])
        conc, n_pairs = pairwise_concordance(oof_df['actual_velocity'], oof_df[col])
        close_conc, n_close_pairs = close_pairs_concordance(oof_df['actual_velocity'], oof_df[col], close_gap_mph)
        validation_rows.append({'method': label, 'spearman_rho': rho, 'permutation_p_value': p_val,
                                 'pairwise_concordance': conc, 'n_pairs_compared': n_pairs,
                                 'close_pairs_concordance': close_conc, 'n_close_pairs': n_close_pairs})
    validation_df = pd.DataFrame(validation_rows)
    print(validation_df.round(4), flush=True)

    oof_corr = oof_df[['methodA', 'methodB', 'methodC']].rank(ascending=False).corr(method='spearman')
    print('OOF inter-method rank correlation:')
    print(oof_corr.round(3), flush=True)

    # Bootstrap CIs on rho and concordance, resampling whole pitchers (not pairs) so every
    # pairwise comparison within a draw stays internally consistent. The SAME resampled draw is
    # used for all three methods each iteration, making the paired differences below a valid test
    # of whether the methods differ from EACH OTHER, not just whether each beats chance.
    rng_v = np.random.RandomState(RANDOM_STATE)
    boot_cols = ['methodA', 'methodB', 'methodC']
    boot_rho = {c: np.empty(n_boot_valid) for c in boot_cols}
    boot_conc = {c: np.empty(n_boot_valid) for c in boot_cols}
    boot_close = {c: np.empty(n_boot_valid) for c in boot_cols}
    n_oof = len(oof_df)
    for b in range(n_boot_valid):
        samp_idx = rng_v.randint(0, n_oof, n_oof)
        samp = oof_df.iloc[samp_idx]
        for c in boot_cols:
            boot_rho[c][b] = spearmanr(samp['actual_velocity'], samp[c]).correlation
            boot_conc[c][b], _ = pairwise_concordance(samp['actual_velocity'], samp[c])
            boot_close[c][b], _ = close_pairs_concordance(samp['actual_velocity'], samp[c], close_gap_mph)

    ci_rows = []
    for c, label in zip(boot_cols, ['Method A', 'Method B', 'Method C']):
        ci_rows.append({
            'method': label,
            'rho_ci_lo': np.nanpercentile(boot_rho[c], 2.5), 'rho_ci_hi': np.nanpercentile(boot_rho[c], 97.5),
            'concordance_ci_lo': np.nanpercentile(boot_conc[c], 2.5),
            'concordance_ci_hi': np.nanpercentile(boot_conc[c], 97.5),
            'close_concordance_ci_lo': np.nanpercentile(boot_close[c], 2.5),
            'close_concordance_ci_hi': np.nanpercentile(boot_close[c], 97.5),
        })
    ci_df = pd.DataFrame(ci_rows)
    print('Bootstrap 95% CIs (2000 pitcher resamples):', flush=True)
    print(ci_df.round(4), flush=True)

    pair_tests = []
    for c1, c2 in [('methodA', 'methodC'), ('methodA', 'methodB'), ('methodC', 'methodB')]:
        diff_rho = boot_rho[c1] - boot_rho[c2]
        diff_conc = boot_conc[c1] - boot_conc[c2]
        pair_tests.append({
            'comparison': f'{c1} vs {c2}',
            'rho_diff_mean': diff_rho.mean(),
            'rho_diff_ci_lo': np.percentile(diff_rho, 2.5), 'rho_diff_ci_hi': np.percentile(diff_rho, 97.5),
            'conc_diff_mean': diff_conc.mean(),
            'conc_diff_ci_lo': np.percentile(diff_conc, 2.5), 'conc_diff_ci_hi': np.percentile(diff_conc, 97.5),
        })
    pair_test_df = pd.DataFrame(pair_tests)
    print('Paired bootstrap method comparisons (95% CI on the difference; CI excluding 0 = significant):',
          flush=True)
    print(pair_test_df.round(4), flush=True)

    safe_to_csv(validation_df, os.path.join(OUTPUT_DIR, 'ranking_validation.csv'), index=False)
    safe_to_csv(ci_df, os.path.join(OUTPUT_DIR, 'ranking_validation_bootstrap_ci.csv'), index=False)
    safe_to_csv(pair_test_df, os.path.join(OUTPUT_DIR, 'ranking_validation_pairwise_tests.csv'), index=False)

    method_labels = ['Method A\n(model prediction)', 'Method B\n(comparable pitchers)',
                      'Method C\n(pairwise comparison)']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    ax = axes[0]
    rho_vals = validation_df['spearman_rho'].values
    rho_err = [rho_vals - ci_df['rho_ci_lo'].values, ci_df['rho_ci_hi'].values - rho_vals]
    bars = ax.bar(method_labels, rho_vals, yerr=rho_err, capsize=5, color=[PHI_RED, PHI_BLUE, PHI_PURPLE])
    _, null_dist_ref, _ = permutation_test_spearman(oof_df['actual_velocity'], oof_df['methodA'], n_perm=500)
    null_rho_975 = np.percentile(np.abs(null_dist_ref), 97.5)
    ax.axhspan(-null_rho_975, null_rho_975, color='gray', alpha=0.15,
               label='chance-level range\n(H0: no real association between\nrank and actual velocity, alpha=0.05)')
    for bar, p_val in zip(bars, validation_df['permutation_p_value']):
        ax.annotate(f'rho={bar.get_height():.2f}\np={p_val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 14), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Spearman rank correlation (OOF)')
    ax.set_title('Rank Correlation (95% Bootstrap CI)')
    ax.legend(loc='lower right', fontsize=7.5)

    ax2 = axes[1]
    x = np.arange(3)
    width = 0.35
    conc_vals = validation_df['pairwise_concordance'].values
    close_vals = validation_df['close_pairs_concordance'].values
    conc_err = [conc_vals - ci_df['concordance_ci_lo'].values, ci_df['concordance_ci_hi'].values - conc_vals]
    close_err = [close_vals - ci_df['close_concordance_ci_lo'].values,
                 ci_df['close_concordance_ci_hi'].values - close_vals]
    ax2.bar(x - width / 2, conc_vals, width, yerr=conc_err, capsize=4, label='all pairs',
            color=[PHI_RED, PHI_BLUE, PHI_PURPLE], alpha=0.5)
    ax2.bar(x + width / 2, close_vals, width, yerr=close_err, capsize=4,
            label=f'close pairs only\n(actual gap <= {close_gap_mph:.1f} mph, hardest 25%)',
            color=[PHI_RED, PHI_BLUE, PHI_PURPLE])
    ax2.axhline(0.5, color='black', linestyle=':', linewidth=1, label='coin-flip baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels)
    ax2.set_ylabel('Pairwise concordance')
    ax2.set_title('Concordance: All Pairs vs. Close Pairs')
    ax2.legend(loc='lower right', fontsize=7.5)
    ax2.set_ylim(0.3, 1.0)

    fig.suptitle('Ranking Validation: Rank Correlation and Concordance', y=1.02, fontsize=12)
    savefig('19_validation_rank_metrics.png')

    ctx['oof_df'] = oof_df
    ctx['validation_df'] = validation_df
    return ctx
