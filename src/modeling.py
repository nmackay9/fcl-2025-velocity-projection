"""Stage 6: model comparison on the final, reduced feature set (incl. a neural net), then a
short hyperparameter search for the winning model.

Mirrors notebook Section 7. XGBoost wins again here, with a clear margin over the neural net,
consistent with gradient-boosted trees tending to outperform neural nets on small tabular
datasets (~200 independent pitchers' worth of effective sample size).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import PHI_BLUE, PHI_PURPLE, PHI_RED, RANDOM_STATE, TARGET, savefig

USE_ARCHETYPE_ENC = True  # reset properly inside run(), kept here only as a documented default


def build_X_reduced(data, stable_features, archetype_enc_map=None):
    reduced_features_no_enc = [f for f in stable_features if f != 'archetype_target_enc']
    use_enc = 'archetype_target_enc' in stable_features
    X = data[reduced_features_no_enc].copy()
    if use_enc:
        X['archetype_target_enc'] = data['mech_archetype'].map(archetype_enc_map)
    return X.astype(float)


def mlp_pipeline():
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                      ('model', MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', alpha=0.01,
                                              early_stopping=True, n_iter_no_change=15, max_iter=2000,
                                              random_state=RANDOM_STATE))])


def build_models_reduced():
    return {
        'ElasticNet': Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                                 ('model', ElasticNet(random_state=RANDOM_STATE, max_iter=10000))]),
        'RandomForest': Pipeline([('impute', SimpleImputer(strategy='median')),
                                   ('model', RandomForestRegressor(n_estimators=300, max_depth=8,
                                                                     random_state=RANDOM_STATE, n_jobs=-1))]),
        'XGBoost': Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('model', XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                      subsample=0.8, colsample_bytree=0.8,
                                                      random_state=RANDOM_STATE, n_jobs=-1))]),
        'Neural Net (MLP)': mlp_pipeline(),
    }


def run(ctx):
    df = ctx['df']
    fcl = ctx['fcl']
    stable_features = ctx['stable_features']
    y, groups, cv, sample_weight = ctx['y'], ctx['groups'], ctx['cv'], ctx['sample_weight']

    models_reduced = build_models_reduced()
    reduced_results = []
    rng2 = np.random.RandomState(RANDOM_STATE)
    for m_name, pipe in models_reduced.items():
        rmses, r2s = [], []
        for train_idx, val_idx in cv.split(df, y, groups):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            archetype_enc_map_fold = train_df.groupby('mech_archetype')[TARGET].mean()
            X_tr = build_X_reduced(train_df, stable_features, archetype_enc_map_fold)
            X_val = build_X_reduced(val_df, stable_features, archetype_enc_map_fold)
            if 'archetype_target_enc' in stable_features:
                X_val['archetype_target_enc'] = X_val['archetype_target_enc'].fillna(archetype_enc_map_fold.mean())
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_tr = sample_weight.iloc[train_idx].values
            if m_name == 'Neural Net (MLP)':
                boot_idx = rng2.choice(len(X_tr), size=len(X_tr), replace=True, p=w_tr / w_tr.sum())
                pipe.fit(X_tr.iloc[boot_idx], y_tr.iloc[boot_idx])
            else:
                pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)
            pred = pipe.predict(X_val)
            rmses.append(root_mean_squared_error(y_val, pred))
            r2s.append(r2_score(y_val, pred))
        reduced_results.append({'model': m_name, 'cv_rmse': np.mean(rmses), 'cv_r2': np.mean(r2s)})
        print(m_name, 'rmse=', round(np.mean(rmses), 4), 'r2=', round(np.mean(r2s), 4), flush=True)

    reduced_results_df = pd.DataFrame(reduced_results).sort_values('cv_rmse').reset_index(drop=True)
    print(reduced_results_df, flush=True)

    plt.figure(figsize=(8, 5))
    plt.bar(reduced_results_df['model'], reduced_results_df['cv_rmse'],
            color=[PHI_RED, PHI_BLUE, PHI_PURPLE, 'darkseagreen'])
    plt.ylabel('Grouped CV RMSE (mph)')
    plt.title(f'Model Comparison on the Final {len(stable_features)}-Feature Set')
    plt.xticks(rotation=15, ha='right')
    savefig('13_model_comparison_reduced.png')

    best_model_name = reduced_results_df.iloc[0]['model']
    print('BEST:', best_model_name, round(reduced_results_df.iloc[0]['cv_rmse'], 4))

    # --- Hyperparameter tuning of the winning model (XGBoost) on the full reduced feature set ---
    archetype_enc_map_full = df.groupby('mech_archetype')[TARGET].mean()
    df['archetype_target_enc'] = df['mech_archetype'].map(archetype_enc_map_full)
    fcl['archetype_target_enc'] = fcl['mech_archetype'].map(archetype_enc_map_full)

    X_full_reduced = df[stable_features].astype(float)
    X_fcl_reduced = fcl[stable_features].astype(float)

    param_dist_xgb = {
        'model__n_estimators': [200, 300, 500, 800], 'model__max_depth': [2, 3, 4, 5],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
        'model__subsample': [0.6, 0.8, 1.0], 'model__colsample_bytree': [0.6, 0.8, 1.0],
    }
    search = RandomizedSearchCV(models_reduced['XGBoost'], param_dist_xgb, n_iter=20,
                                 scoring='neg_root_mean_squared_error', cv=cv,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_full_reduced, y, groups=groups, model__sample_weight=sample_weight.values)
    best_xgb_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
    print('Tuned XGBoost params:', best_xgb_params, flush=True)
    print('Tuned CV RMSE:', -search.best_score_, flush=True)

    ctx['df'] = df
    ctx['fcl'] = fcl
    ctx['X_full_reduced'] = X_full_reduced
    ctx['X_fcl_reduced'] = X_fcl_reduced
    ctx['best_xgb_params'] = best_xgb_params
    return ctx
