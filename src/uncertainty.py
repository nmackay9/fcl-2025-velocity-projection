"""Stage 8: three explicit layers of uncertainty for "hardest pitch in five years."

Mirrors notebook Section 9. Ranking by predicted mean velocity only answers "who has the most
efficient mechanics right now" -- getting to "who throws the single hardest pitch" requires
propagating (1) model uncertainty, (2) player-development uncertainty, and (3) pitch-to-pitch
variability, each as a distribution, through a simulation (simulation.py).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import PITCHER_COL, RANDOM_STATE, TARGET


def run_posterior_ensemble(ctx, b_ensemble=200):
    """Layer 1 (model uncertainty) + Layer 2 (development uncertainty): bootstrap-resample the
    tuned model B_ENSEMBLE times and predict every FCL-2025 pitcher each time; independently
    bootstrap the small dual-year development sample the same number of times."""
    df = ctx['df']
    fcl = ctx['fcl']
    X_full_reduced = ctx['X_full_reduced']
    X_fcl_reduced = ctx['X_fcl_reduced']
    best_xgb_params = ctx['best_xgb_params']
    sample_weight = ctx['sample_weight']
    dev_compare = ctx['dev_compare']
    n_dev_pitchers = ctx['n_dev_pitchers']

    unique_pitchers = df[PITCHER_COL].unique()
    pitcher_row_indices = {p: df.index[df[PITCHER_COL] == p].to_numpy() for p in unique_pitchers}
    row_weight_full = sample_weight.copy()

    fcl_pred_matrix = np.zeros((b_ensemble, len(fcl)))
    rng4 = np.random.RandomState(RANDOM_STATE)
    for b in range(b_ensemble):
        boot_pitchers = rng4.choice(unique_pitchers, size=len(unique_pitchers), replace=True)
        boot_idx = np.concatenate([pitcher_row_indices[p] for p in boot_pitchers])
        X_boot = X_full_reduced.loc[boot_idx]
        y_boot = df.loc[boot_idx, TARGET]
        w_boot = row_weight_full.loc[boot_idx]
        imp_b = SimpleImputer(strategy='median')
        X_boot_imp = imp_b.fit_transform(X_boot)
        m = XGBRegressor(**best_xgb_params, random_state=int(b), n_jobs=-1)
        m.fit(X_boot_imp, y_boot, sample_weight=w_boot.values)
        X_fcl_imp = imp_b.transform(X_fcl_reduced)
        fcl_pred_matrix[b, :] = m.predict(X_fcl_imp)
        if (b + 1) % 50 == 0:
            print(f'ensemble {b + 1}/{b_ensemble}', flush=True)

    fcl_pitcher_id_arr = fcl[PITCHER_COL].values
    unique_fcl_pitchers = fcl[PITCHER_COL].unique()
    pitcher_boot_means = {p: fcl_pred_matrix[:, fcl_pitcher_id_arr == p].mean(axis=1) for p in unique_fcl_pitchers}
    print('Posterior ensemble built for', len(pitcher_boot_means), 'FCL-2025 pitchers', flush=True)

    growth_draws = np.array([rng4.choice(dev_compare['delta'].to_numpy(), size=n_dev_pitchers, replace=True).mean()
                              for _ in range(b_ensemble)])
    print(f'Growth-adjustment posterior: mean={growth_draws.mean():+.3f} mph, std={growth_draws.std():.3f} mph '
          f'(from n={n_dev_pitchers} dual-year pitchers)', flush=True)

    ctx['unique_pitchers'] = unique_pitchers
    ctx['pitcher_row_indices'] = pitcher_row_indices
    ctx['row_weight_full'] = row_weight_full
    ctx['unique_fcl_pitchers'] = unique_fcl_pitchers
    ctx['pitcher_boot_means'] = pitcher_boot_means
    ctx['growth_draws'] = growth_draws
    ctx['rng4'] = rng4
    return ctx


def run_sigma(ctx):
    """Layer 3 (pitch-to-pitch variability). First tries an archetype-pooled sigma, then an
    individualized mechanics-consistency model -- which fails to generalize (CV R^2 ~ 0) -- and
    settles on a single flat, population-average sigma applied identically to every pitcher."""
    df = ctx['df']
    fcl = ctx['fcl']

    pitcher_std_train = df.groupby(PITCHER_COL)[TARGET].std()
    pitcher_archetype_train = df.groupby(PITCHER_COL)['mech_archetype'].agg(lambda x: x.value_counts().idxmax())
    std_by_archetype = pitcher_std_train.groupby(pitcher_archetype_train).mean()
    print('Pitch-to-pitch sigma by archetype (labeled data):')
    print(std_by_archetype)
    fcl_pitcher_archetype = fcl.groupby(PITCHER_COL)['mech_archetype'].agg(lambda x: x.value_counts().idxmax())
    sigma_archetype = fcl_pitcher_archetype.map(std_by_archetype).fillna(pitcher_std_train.mean())

    # Attempt an individualized sigma model from mechanics-consistency markers -- no velocity
    # data required, so it can be trained on labeled pitchers and applied to FCL-2025 ones.
    consistency_features_raw = ['arm_slot', 'torso_rotation_velo_max', 'stride_length',
                                 'throw_elbow_flexion_at_ball_release', 'center_of_mass_velo_max']
    train_consistency = df.groupby(PITCHER_COL)[consistency_features_raw].std()
    train_consistency.columns = [f'{c}_std' for c in train_consistency.columns]
    train_drift = df.groupby(PITCHER_COL)[['arm_slot_outing_drift', 'stride_length_outing_drift']].first()
    sigma_feature_cols = train_consistency.columns.tolist() + ['arm_slot_outing_drift', 'stride_length_outing_drift']

    pitcher_velo_std = df.groupby(PITCHER_COL)[TARGET].std().rename('velo_std')
    n_pitches_train_s = df.groupby(PITCHER_COL).size().rename('n_pitches')
    sigma_model_data = train_consistency.join(train_drift).join(pitcher_velo_std).join(n_pitches_train_s)
    sigma_model_data = sigma_model_data[sigma_model_data['n_pitches'] >= 5].dropna()
    X_sigma = sigma_model_data[sigma_feature_cols].astype(float)
    y_sigma = sigma_model_data['velo_std']

    kf_sigma = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sigma_candidates = {
        'ElasticNet': Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                                 ('model', ElasticNet(random_state=RANDOM_STATE, max_iter=10000))]),
        'RandomForest': Pipeline([('impute', SimpleImputer(strategy='median')),
                                   ('model', RandomForestRegressor(n_estimators=200, max_depth=4,
                                                                     random_state=RANDOM_STATE, n_jobs=-1))]),
    }
    sigma_results = []
    for name, pipe in sigma_candidates.items():
        rmses, r2s = [], []
        for tr_idx, val_idx in kf_sigma.split(X_sigma):
            pipe.fit(X_sigma.iloc[tr_idx], y_sigma.iloc[tr_idx])
            pred = pipe.predict(X_sigma.iloc[val_idx])
            rmses.append(root_mean_squared_error(y_sigma.iloc[val_idx], pred))
            r2s.append(r2_score(y_sigma.iloc[val_idx], pred))
        sigma_results.append({'model': name, 'cv_rmse': np.mean(rmses), 'cv_r2': np.mean(r2s)})
    sigma_results_df = pd.DataFrame(sigma_results).sort_values('cv_rmse').reset_index(drop=True)
    print(sigma_results_df, flush=True)
    print(f'Population-mean-only baseline RMSE: {y_sigma.std():.3f}', flush=True)
    print('Individualized sigma model does not generalize (CV R^2 ~ 0) -- falling back to a flat, '
          'population-average sigma, applied identically to every FCL-2025 pitcher.', flush=True)

    sigma_flat = float(pitcher_std_train.mean())
    sigma_final = pd.Series(sigma_flat, index=fcl_pitcher_archetype.index)
    print(f'Flat population sigma applied to every FCL-2025 pitcher: {sigma_flat:.3f} mph '
          f'(vs archetype-pooled range {std_by_archetype.min():.3f}-{std_by_archetype.max():.3f} mph)', flush=True)

    ctx['sigma_final'] = sigma_final
    return ctx


def run(ctx):
    ctx = run_posterior_ensemble(ctx)
    ctx = run_sigma(ctx)
    return ctx
