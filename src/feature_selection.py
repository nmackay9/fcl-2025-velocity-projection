"""Stage 5: feature-set ablation, curated interactions, and bootstrap stability selection.

Mirrors notebook Sections 5-6: first a broad ablation testing whether the engineered/archetype
features actually help a model beyond raw mechanics, then a disciplined, validated reduction of
the full candidate pool (raw + engineered + curated interactions) down to a small, stable feature
set, using two independent selection criteria (Lasso for linear signal, XGBoost importance for
nonlinear signal) so a feature only has to clear one bar to survive.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LassoCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import PHI_BLUE, PHI_PURPLE, PHI_RED, PITCHER_COL, RANDOM_STATE, TARGET, savefig


def build_X(data, cols):
    X = data[cols].copy()
    cat_cols = [c for c in ['pitcher_handedness', 'mech_archetype'] if c in X.columns]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X.astype(float)


def run_initial_ablation(ctx):
    """Section 5: does the Section-3 engineering or the Section-4 archetype identity actually
    improve held-out prediction, beyond what raw mechanics already provide?"""
    df = ctx['df']
    raw_mechanics_cols = ctx['raw_mechanics_cols']
    engineered_extra_cols = ctx['engineered_extra_cols']

    feature_sets = {
        'raw_mechanics_only': raw_mechanics_cols + ['pitcher_handedness'],
        'plus_engineered': raw_mechanics_cols + engineered_extra_cols + ['pitcher_handedness'],
        'plus_archetype_cluster': raw_mechanics_cols + engineered_extra_cols +
            ['umap_1', 'umap_2', 'mech_archetype', 'pitcher_handedness'],
    }
    feature_sets = {k: [c for c in v if c != 'season'] for k, v in feature_sets.items()}

    models_initial = {
        'ElasticNet': Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                                 ('model', ElasticNet(random_state=RANDOM_STATE, max_iter=10000))]),
        'RandomForest': Pipeline([('impute', SimpleImputer(strategy='median')),
                                   ('model', RandomForestRegressor(n_estimators=300, max_depth=8,
                                                                     random_state=RANDOM_STATE, n_jobs=-1))]),
        'XGBoost': Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('model', XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                      subsample=0.8, colsample_bytree=0.8,
                                                      random_state=RANDOM_STATE, n_jobs=-1))]),
    }

    y = df[TARGET]
    groups = df[PITCHER_COL]
    pitch_counts_map = df.groupby(PITCHER_COL).size()
    sample_weight = (1.0 / df[PITCHER_COL].map(pitch_counts_map))
    sample_weight = sample_weight / sample_weight.mean()
    cv = GroupKFold(n_splits=5)

    initial_results = []
    for fs_name, cols in feature_sets.items():
        X = build_X(df, cols)
        for m_name, pipe in models_initial.items():
            rmses, r2s = [], []
            for train_idx, val_idx in cv.split(X, y, groups):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                w_tr = sample_weight.iloc[train_idx].values
                pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)
                pred = pipe.predict(X_val)
                rmses.append(root_mean_squared_error(y_val, pred))
                r2s.append(r2_score(y_val, pred))
            initial_results.append({'feature_set': fs_name, 'model': m_name,
                                     'cv_rmse': np.mean(rmses), 'cv_r2': np.mean(r2s)})

    initial_results_df = pd.DataFrame(initial_results).sort_values('cv_rmse').reset_index(drop=True)
    print(initial_results_df, flush=True)

    plt.figure(figsize=(10, 6))
    pivot = initial_results_df.pivot(index='feature_set', columns='model', values='cv_rmse')
    pivot = pivot.reindex(['raw_mechanics_only', 'plus_engineered', 'plus_archetype_cluster'])
    pivot.plot(kind='bar', ax=plt.gca(), color=[PHI_RED, PHI_BLUE, PHI_PURPLE])
    plt.ylabel('Grouped CV RMSE (mph)')
    plt.title('Do the Engineered / Archetype Features Improve Raw-Velocity Prediction?')
    plt.xticks(rotation=15, ha='right')
    savefig('11_initial_feature_ablation.png')

    best_initial = initial_results_df.iloc[0]
    print('BEST:', best_initial['feature_set'], best_initial['model'], round(best_initial['cv_rmse'], 4))

    ctx['y'] = y
    ctx['groups'] = groups
    ctx['sample_weight'] = sample_weight
    ctx['cv'] = cv
    return ctx


def add_interactions(data, squares, products):
    d = data.copy()
    for f in squares:
        d[f'{f}_sq'] = d[f] ** 2
    for f1, f2 in products:
        d[f'{f1}_x_{f2}'] = d[f1] * d[f2]
    return d


def run_interactions_and_candidates(ctx):
    """Section 6 (part 1): curated second-order interactions + physics-motivated transforms,
    on top of the raw and engineered features, as the candidate pool for stability selection."""
    df, fcl = ctx['df'], ctx['fcl']

    squares = ['torso_rotation_velo_max', 'stride_length', 'center_of_mass_velo_max',
               'throw_shoulder_external_rotation_max', 'pelvis_rotation_velo_max',
               'throw_shoulder_internal_rotation_velo_max']
    products = [
        ('torso_rotation_velo_max', 'arm_slot'), ('torso_rotation_velo_max', 'pelvis_rotation_velo_max'),
        ('pelvis_rotation_velo_max', 'throw_shoulder_internal_rotation_velo_max'),
        ('torso_rotation_velo_max', 'throw_shoulder_internal_rotation_velo_max'),
        ('stride_length', 'torso_rotation_velo_max'), ('stride_length', 'center_of_mass_velo_max'),
        ('torso_sidebend_at_mer', 'torso_rotation_velo_max'),
        ('hip_shoulder_separation_max', 'torso_rotation_velo_max'),
        ('arm_slot', 'throw_shoulder_external_rotation_max'), ('player_height', 'stride_length'),
        ('lead_leg_internal_rotation_velo_max', 'pelvis_rotation_velo_max'),
    ]

    df = add_interactions(df, squares, products)
    fcl = add_interactions(fcl, squares, products)
    interaction_cols = [f'{f}_sq' for f in squares] + [f'{f1}_x_{f2}' for f1, f2 in products]
    print(len(interaction_cols), 'curated interaction terms added')

    base_features = [
        'torso_rotation_velo_max', 'player_height', 'throw_shoulder_external_rotation_max', 'stride_length',
        'throw_elbow_extension_velo_max', 'center_of_mass_velo_max', 'torso_sidebend_at_mer',
        'lead_leg_internal_rotation_velo_max', 'throw_elbow_flexion_at_mer', 'lead_hip_flexion_at_mer',
        'throw_shoulder_internal_rotation_velo_max', 'back_knee_flexion_at_mer', 'pelvis_rotation_velo_max',
        'back_knee_flexion_at_ball_release', 'torso_forward_bend_at_foot_plant', 'torso_sidebend_at_ball_release',
        'arm_slot', 'torso_forward_bend_at_ball_release', 'glove_shoulder_abduction_at_mer',
        'hip_shoulder_separation_max', 'stride_angle',
    ]
    extra_engineered = ['stride_length_norm', 'stride_length_norm_log', 'stride_angle_dev',
                        'seq_ratio_pelvis_to_torso', 'seq_ratio_torso_to_arm',
                        'seq_logratio_pelvis_to_torso', 'seq_logratio_torso_to_arm',
                        'arm_slot_outing_drift', 'stride_length_outing_drift']

    archetype_mean_full = df.groupby('mech_archetype')[TARGET].mean()
    df['archetype_target_enc'] = df['mech_archetype'].map(archetype_mean_full)
    fcl['archetype_target_enc'] = fcl['mech_archetype'].map(archetype_mean_full)

    candidate_features = sorted(set(base_features + interaction_cols + extra_engineered +
                                     ['archetype_target_enc', 'pitcher_handedness_num', 'umap_1', 'umap_2']))
    print('n candidate features for stability selection:', len(candidate_features))

    ctx['df'] = df
    ctx['fcl'] = fcl
    ctx['interaction_cols'] = interaction_cols
    ctx['candidate_features'] = candidate_features
    return ctx


def run_stability_selection(ctx, n_bootstrap=60, top_k_xgb=15, stability_threshold=0.5):
    """Section 6 (part 2): dual-criterion (Lasso + XGBoost) bootstrap stability selection."""
    df = ctx['df']
    candidate_features = ctx['candidate_features']
    interaction_cols = ctx['interaction_cols']
    sample_weight = ctx['sample_weight']

    unique_pitchers = df[PITCHER_COL].unique()
    pitcher_row_indices = {p: df.index[df[PITCHER_COL] == p].to_numpy() for p in unique_pitchers}
    row_weight_full = sample_weight.copy()

    X_candidates_full = df[candidate_features].copy()
    imputer_cand = SimpleImputer(strategy='median')
    X_candidates_imputed = pd.DataFrame(imputer_cand.fit_transform(X_candidates_full),
                                         columns=candidate_features, index=df.index)

    rng = np.random.RandomState(RANDOM_STATE)
    selection_counts_lasso = pd.Series(0, index=candidate_features, dtype=float)
    selection_counts_xgb = pd.Series(0, index=candidate_features, dtype=float)
    for b in range(n_bootstrap):
        boot_pitchers = rng.choice(unique_pitchers, size=len(unique_pitchers), replace=True)
        boot_idx = np.concatenate([pitcher_row_indices[p] for p in boot_pitchers])
        X_boot = X_candidates_imputed.loc[boot_idx]
        y_boot = df.loc[boot_idx, TARGET]
        w_boot = row_weight_full.loc[boot_idx]

        X_boot_scaled = StandardScaler().fit_transform(X_boot)
        lasso = LassoCV(cv=3, random_state=RANDOM_STATE, max_iter=3000, n_alphas=15)
        lasso.fit(X_boot_scaled, y_boot, sample_weight=w_boot.values)
        selected_lasso = np.array(candidate_features)[np.abs(lasso.coef_) > 1e-6]
        selection_counts_lasso[selected_lasso] += 1

        xgb_boot = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.08,
                                 subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)
        xgb_boot.fit(X_boot, y_boot, sample_weight=w_boot.values)
        top_xgb = pd.Series(xgb_boot.feature_importances_, index=candidate_features).nlargest(top_k_xgb).index
        selection_counts_xgb[top_xgb] += 1

        if (b + 1) % 20 == 0:
            print(f'bootstrap {b + 1}/{n_bootstrap} done', flush=True)

    selection_freq_lasso = (selection_counts_lasso / n_bootstrap)
    selection_freq_xgb = (selection_counts_xgb / n_bootstrap)
    selection_compare = pd.DataFrame({'lasso_freq': selection_freq_lasso, 'xgb_freq': selection_freq_xgb})
    selection_compare['max_freq'] = selection_compare[['lasso_freq', 'xgb_freq']].max(axis=1)
    selection_compare = selection_compare.sort_values('max_freq', ascending=False)

    nonlinear_only = selection_compare[(selection_compare['xgb_freq'] >= stability_threshold) &
                                        (selection_compare['lasso_freq'] < stability_threshold)]
    linear_only = selection_compare[(selection_compare['lasso_freq'] >= stability_threshold) &
                                     (selection_compare['xgb_freq'] < stability_threshold)]
    print(f"Selected by XGBoost importance but NOT by Lasso (likely nonlinear-only signal, "
          f"{len(nonlinear_only)} features):")
    print(nonlinear_only.round(2), flush=True)
    print(f"\nSelected by Lasso but NOT by XGBoost top-{top_k_xgb} (likely linear-only signal, "
          f"{len(linear_only)} features):")
    print(linear_only.round(2), flush=True)

    top20_compare = selection_compare.head(20).sort_values('max_freq')
    fig, ax = plt.subplots(figsize=(9, 10))
    y_pos = np.arange(len(top20_compare))
    ax.barh(y_pos - 0.2, top20_compare['lasso_freq'], height=0.4, color=PHI_BLUE, label='Lasso (linear) selection freq')
    ax.barh(y_pos + 0.2, top20_compare['xgb_freq'], height=0.4, color=PHI_RED, label='XGBoost (nonlinear) selection freq')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20_compare.index)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='50% stability threshold')
    ax.set_xlabel(f'Selection frequency across {n_bootstrap} pitcher-level bootstrap resamples')
    ax.set_title('Feature Stability: Linear (Lasso) vs Nonlinear (XGBoost) Selection Criteria')
    ax.legend(fontsize=8, loc='lower right')
    savefig('12_stability_selection.png')

    stable_lasso = set(selection_freq_lasso[selection_freq_lasso >= stability_threshold].index)
    stable_xgb = set(selection_freq_xgb[selection_freq_xgb >= stability_threshold].index)
    stable_union = selection_compare.loc[list(stable_lasso | stable_xgb)].sort_values('max_freq', ascending=False)
    stable_features = stable_union.index.tolist()
    if len(stable_features) < 8:
        stable_features = selection_compare.head(10).index.tolist()
    elif len(stable_features) > 14:
        stable_features = selection_compare.head(14).index.tolist()
    print(f'Final feature set ({len(stable_features)} features, linear-or-nonlinear union): {stable_features}',
          flush=True)
    n_interactions_survived = sum(1 for f in stable_features if f in interaction_cols)
    n_rescued = sum(1 for f in stable_features if f in nonlinear_only.index)
    print(f'{n_interactions_survived} of {len(interaction_cols)} hand-picked interaction terms survived', flush=True)
    print(f'{n_rescued} feature(s) survived only because of the nonlinear (XGBoost) criterion', flush=True)

    ctx['stable_features'] = stable_features
    return ctx


def run(ctx):
    ctx = run_initial_ablation(ctx)
    ctx = run_interactions_and_candidates(ctx)
    ctx = run_stability_selection(ctx)
    return ctx
