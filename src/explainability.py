"""Stage 7: explainability -- what's actually driving the model.

Mirrors notebook Section 8. Four complementary views: out-of-fold calibration/residuals,
native gain importance + TreeSHAP (direction and magnitude), partial dependence (shape of the
top effects), and a fully-transparent linear baseline for comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix, XGBRegressor

from .config import PHI_BLUE, PHI_RED, RANDOM_STATE, TARGET, savefig
from .modeling import build_X_reduced


def run(ctx):
    df = ctx['df']
    y, groups, cv, sample_weight = ctx['y'], ctx['groups'], ctx['cv'], ctx['sample_weight']
    stable_features = ctx['stable_features']
    best_xgb_params = ctx['best_xgb_params']
    X_full_reduced = ctx['X_full_reduced']

    # --- View 1: out-of-fold calibration + residuals ---
    oof_pred = pd.Series(index=df.index, dtype=float)
    for train_idx, val_idx in cv.split(df, y, groups):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        archetype_enc_map_fold = train_df.groupby('mech_archetype')[TARGET].mean()
        X_tr = build_X_reduced(train_df, stable_features, archetype_enc_map_fold)
        X_val = build_X_reduced(val_df, stable_features, archetype_enc_map_fold)
        X_val['archetype_target_enc'] = X_val['archetype_target_enc'].fillna(archetype_enc_map_fold.mean())
        y_tr = y.iloc[train_idx]
        w_tr = sample_weight.iloc[train_idx].values
        imp_fold = SimpleImputer(strategy='median')
        X_tr_imp = imp_fold.fit_transform(X_tr)
        X_val_imp = imp_fold.transform(X_val)
        m = XGBRegressor(**best_xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_tr_imp, y_tr, sample_weight=w_tr)
        oof_pred.iloc[val_idx] = m.predict(X_val_imp)

    resid = y - oof_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(oof_pred, y, alpha=0.15, color=PHI_BLUE)
    lims = [y.min(), y.max()]
    axes[0].plot(lims, lims, color=PHI_RED, linestyle='--')
    axes[0].set_xlabel('Out-of-fold predicted velocity')
    axes[0].set_ylabel('Actual velocity')
    axes[0].set_title('Calibration: Predicted vs Actual (Out-of-Fold)')
    import seaborn as sns
    sns.histplot(resid, kde=True, ax=axes[1], color=PHI_RED)
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set_title(f'Residuals (mean={resid.mean():.3f}, std={resid.std():.3f})')
    savefig('14_calibration_residuals.png')
    print('OOF RMSE check:', root_mean_squared_error(y, oof_pred), flush=True)

    # --- Fit the final model on all labeled data ---
    imp_full = SimpleImputer(strategy='median')
    X_full_imp = imp_full.fit_transform(X_full_reduced)
    final_model = XGBRegressor(**best_xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(X_full_imp, y, sample_weight=sample_weight.values)

    native_imp = pd.Series(final_model.feature_importances_, index=stable_features).sort_values(ascending=False)
    print(native_imp, flush=True)
    X_full_imp_df = pd.DataFrame(X_full_imp, columns=stable_features)

    # --- View 2: SHAP (TreeSHAP, via XGBoost's native pred_contribs path) ---
    booster_dm = DMatrix(X_full_imp_df, feature_names=stable_features)
    shap_contribs = final_model.get_booster().predict(booster_dm, pred_contribs=True)
    shap_values = shap_contribs[:, :-1]

    shap.summary_plot(shap_values, X_full_imp_df, show=False, plot_size=(10, 7))
    plt.gcf().suptitle('SHAP Summary: Direction and Magnitude of Each Feature\'s Effect on Predicted Velocity', y=1.02)
    savefig('15b_shap_summary.png')

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=stable_features).sort_values(ascending=False)
    plt.figure(figsize=(9, 7))
    mean_abs_shap.sort_values().plot(kind='barh', color=PHI_BLUE)
    plt.xlabel('Mean |SHAP value| (mph -- average impact on the model\'s predicted velocity)')
    plt.title('SHAP Feature Importance, Final XGBoost Model')
    savefig('15c_shap_importance.png')

    rank_corr_shap_native = spearmanr(mean_abs_shap.reindex(stable_features),
                                       native_imp.reindex(stable_features)).correlation
    print('SHAP mean |value| ranking:')
    print(mean_abs_shap, flush=True)
    print(f'\nSpearman rank correlation between SHAP and native gain importance: {rank_corr_shap_native:.3f}',
          flush=True)

    # --- View 3: partial dependence ---
    top6_for_pdp = native_imp.head(6).index.tolist()
    fig, ax = plt.subplots(figsize=(13, 8))
    PartialDependenceDisplay.from_estimator(final_model, X_full_imp_df, features=top6_for_pdp, ax=ax, n_cols=3)
    savefig('15_partial_dependence.png')

    # --- View 4: fully-transparent linear baseline, for comparison ---
    scaler_lin = StandardScaler()
    X_full_scaled_lin = scaler_lin.fit_transform(X_full_imp)
    linear_baseline = ElasticNet(random_state=RANDOM_STATE, max_iter=10000)
    linear_baseline.fit(X_full_scaled_lin, y, sample_weight=sample_weight.values)
    lin_coefs = pd.Series(linear_baseline.coef_, index=stable_features).sort_values(key=np.abs, ascending=False)
    n_zeroed = int((lin_coefs.abs() < 1e-9).sum())
    print(f'\nLinear baseline zeroed out {n_zeroed} of {len(stable_features)} features:', flush=True)
    print(lin_coefs, flush=True)

    plt.figure(figsize=(9, 7))
    lin_coefs.sort_values().plot(kind='barh', color=[PHI_RED if v < 0 else PHI_BLUE for v in lin_coefs.sort_values()])
    plt.title('Fully-Transparent Baseline: Linear Model Coefficients')
    plt.xlabel('Standardized coefficient (sign = direction of effect on velocity)')
    savefig('16_linear_baseline_coefs.png')

    lin_pred = linear_baseline.predict(X_full_scaled_lin)
    xgb_pred = final_model.predict(X_full_imp)
    lin_vs_xgb_corr = np.corrcoef(lin_pred, xgb_pred)[0, 1]
    plt.figure(figsize=(7, 7))
    plt.scatter(lin_pred, xgb_pred, alpha=0.15, color=PHI_BLUE)
    lims2 = [min(lin_pred.min(), xgb_pred.min()), max(lin_pred.max(), xgb_pred.max())]
    plt.plot(lims2, lims2, color=PHI_RED, linestyle='--')
    plt.xlabel('Simple linear-model prediction')
    plt.ylabel('XGBoost prediction')
    plt.title('Where the Complex Model Agrees and Disagrees with the Simple Baseline')
    savefig('17_simple_vs_complex.png')
    print('Linear vs XGBoost prediction correlation:', lin_vs_xgb_corr, flush=True)

    ctx['final_model'] = final_model
    ctx['X_full_imp_df'] = X_full_imp_df
    return ctx
