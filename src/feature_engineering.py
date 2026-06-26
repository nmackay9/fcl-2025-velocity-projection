"""Stage 3: engineer features that encode a biomechanical idea, not just a raw angle.

Mirrors notebook Section 3 (kinetic-chain hypothesis -> engineered features) plus the
within-outing drift helper that EDA's fatigue-drift figure also depends on.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import DATE_COL, PHI_BLUE, PHI_RED, PITCHER_COL, TARGET, savefig


def compute_outing_drift(data, value_col, pitcher_col=PITCHER_COL, date_col=DATE_COL,
                          order_col='orig_order', min_pitches=3):
    """Mean within-outing linear slope of `value_col` over the course of each outing."""
    drifts = {}
    sorted_data = data.sort_values(order_col)
    for (pid, date), g in sorted_data.groupby([pitcher_col, date_col]):
        if len(g) >= min_pitches:
            yv = g[value_col].to_numpy()
            if np.std(yv) > 0:
                xv = np.arange(len(yv))
                slope = np.polyfit(xv, yv, 1)[0]
                drifts.setdefault(pid, []).append(slope)
    import pandas as pd
    return pd.Series({pid: float(np.mean(v)) for pid, v in drifts.items()}, name=f'{value_col}_outing_drift')


def engineer_mechanics_features(data):
    """Kinetic-chain-motivated engineered features: normalized stride, rotational/linear
    kinetic-energy proxies, and proximal-to-distal sequencing ratios (raw and log-ratio forms)."""
    d = data.copy()
    eps = 1e-6
    d['stride_length_norm'] = d['stride_length'] / d['player_height']
    d['stride_angle_dev'] = d['stride_angle'].abs()
    # Allometric scaling (log-log linearizes power-law body-size relationships -- standard
    # practice for size-normalized biomechanical variables) gives a second, literature-motivated
    # functional form of the same normalized-stride idea.
    d['stride_length_norm_log'] = np.log(d['stride_length_norm'].clip(lower=eps))

    rot_weights = {
        'pelvis_rotation_velo_max': 0.22, 'torso_rotation_velo_max': 0.18,
        'back_leg_external_rotation_velo_max': 0.12, 'lead_leg_internal_rotation_velo_max': 0.12,
        'lead_knee_extension_velo_max': 0.10, 'throw_shoulder_internal_rotation_velo_max': 0.16,
        'throw_elbow_extension_velo_max': 0.10,
    }
    rot_energy = np.zeros(len(d))
    for col, w in rot_weights.items():
        omega_rad = np.deg2rad(d[col])
        rot_energy += w * (omega_rad ** 2)
    d['rotational_ke_proxy'] = rot_energy
    d['linear_ke_proxy'] = (d['center_of_mass_velo_max'] ** 2) * d['player_height']

    # Raw-ratio sequencing proxy (magnitude only, clipped to bound outliers from near-zero
    # denominators) ...
    seq1 = (d['torso_rotation_velo_max'].abs() / (d['pelvis_rotation_velo_max'].abs() + eps)).clip(0, 5)
    seq2 = (d['throw_shoulder_internal_rotation_velo_max'].abs() / (d['torso_rotation_velo_max'].abs() + eps)).clip(0, 5)
    d['seq_ratio_pelvis_to_torso'] = seq1
    d['seq_ratio_torso_to_arm'] = seq2
    d['kinetic_chain_efficiency'] = d[['seq_ratio_pelvis_to_torso', 'seq_ratio_torso_to_arm']].mean(axis=1)
    # ... and a log-ratio version of the same idea. A log-ratio is the standard form for comparing
    # two positive, multiplicatively-related quantities (symmetric around zero, not bounded below
    # by zero the way a raw ratio is) -- both forms are kept as separate candidates and stability
    # selection (feature_selection.py) decides empirically which functional form, if either,
    # actually carries signal.
    d['seq_logratio_pelvis_to_torso'] = np.log((d['torso_rotation_velo_max'].abs() + eps) /
                                                (d['pelvis_rotation_velo_max'].abs() + eps))
    d['seq_logratio_torso_to_arm'] = np.log((d['throw_shoulder_internal_rotation_velo_max'].abs() + eps) /
                                             (d['torso_rotation_velo_max'].abs() + eps))
    return d


def run(ctx):
    df, fcl = ctx['df'], ctx['fcl']

    arm_slot_drift_df = compute_outing_drift(df, 'arm_slot')
    stride_drift_df = compute_outing_drift(df, 'stride_length')
    velo_drift_df = compute_outing_drift(df, 'velocity')
    arm_slot_drift_fcl = compute_outing_drift(fcl, 'arm_slot')
    stride_drift_fcl = compute_outing_drift(fcl, 'stride_length')

    df['arm_slot_outing_drift'] = df[PITCHER_COL].map(arm_slot_drift_df).fillna(0)
    df['stride_length_outing_drift'] = df[PITCHER_COL].map(stride_drift_df).fillna(0)
    fcl['arm_slot_outing_drift'] = fcl[PITCHER_COL].map(arm_slot_drift_fcl).fillna(0)
    fcl['stride_length_outing_drift'] = fcl[PITCHER_COL].map(stride_drift_fcl).fillna(0)

    df = engineer_mechanics_features(df)
    fcl = engineer_mechanics_features(fcl)

    engineered_extra_cols = [
        'stride_length_norm', 'stride_length_norm_log', 'stride_angle_dev', 'rotational_ke_proxy',
        'linear_ke_proxy', 'seq_ratio_pelvis_to_torso', 'seq_ratio_torso_to_arm',
        'seq_logratio_pelvis_to_torso', 'seq_logratio_torso_to_arm', 'kinetic_chain_efficiency',
        'arm_slot_outing_drift', 'stride_length_outing_drift',
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.regplot(data=df, x='stride_length_norm', y=TARGET, ax=axes[0],
                scatter_kws={'alpha': 0.15, 'color': PHI_BLUE}, line_kws={'color': PHI_RED})
    axes[0].set_title('Stride Length (% of height) vs Velocity')
    sns.regplot(data=df, x='kinetic_chain_efficiency', y=TARGET, ax=axes[1],
                scatter_kws={'alpha': 0.15, 'color': PHI_BLUE}, line_kws={'color': PHI_RED})
    axes[1].set_title('Kinetic-Chain Sequencing Efficiency vs Velocity')
    savefig('06_stride_efficiency.png')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.regplot(data=df, x='rotational_ke_proxy', y=TARGET, ax=axes[0],
                scatter_kws={'alpha': 0.15, 'color': PHI_BLUE}, line_kws={'color': PHI_RED})
    axes[0].set_title('Rotational Kinetic-Energy Proxy vs Velocity')
    sns.regplot(data=df, x='torso_rotation_velo_max', y=TARGET, ax=axes[1],
                scatter_kws={'alpha': 0.15, 'color': PHI_BLUE}, line_kws={'color': PHI_RED})
    axes[1].set_title('Raw Torso Rotation Velocity vs Velocity')
    savefig('07_kinetic_chain.png')

    print('corr stride_length_norm:', round(df['stride_length_norm'].corr(df[TARGET]), 3))
    print('corr kinetic_chain_efficiency:', round(df['kinetic_chain_efficiency'].corr(df[TARGET]), 3))
    print('corr rotational_ke_proxy:', round(df['rotational_ke_proxy'].corr(df[TARGET]), 3))
    print('corr torso_rotation_velo_max (raw ingredient):', round(df['torso_rotation_velo_max'].corr(df[TARGET]), 3))

    ctx['df'] = df
    ctx['fcl'] = fcl
    ctx['velo_drift_df'] = velo_drift_df
    ctx['engineered_extra_cols'] = engineered_extra_cols
    return ctx
