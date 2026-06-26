"""Stage 4: discover mechanical archetypes via a shared UMAP embedding + KMeans.

Mirrors notebook Section 4. Fit on the pooled labeled + FCL-2025 data with no velocity
information at all, so both cohorts land in one shared space and an FCL-2025 pitcher can be
meaningfully compared to the labeled population.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import PALETTE, PHI_BLUE, PHI_RED, RANDOM_STATE, TARGET, savefig


def run(ctx):
    df, fcl = ctx['df'], ctx['fcl']
    raw_mechanics_cols = ctx['raw_mechanics_cols']

    archetype_feature_cols = ['stride_length_norm', 'stride_angle_dev', 'arm_slot_outing_drift',
                               'stride_length_outing_drift']
    shared_feature_cols = [c for c in raw_mechanics_cols if c != 'season'] + archetype_feature_cols
    shared_feature_cols = [c for c in shared_feature_cols if c in df.columns and c in fcl.columns]
    print('n shared feature cols for UMAP:', len(shared_feature_cols))

    df_pool = df[shared_feature_cols].copy()
    df_pool['__source__'] = 'labeled'
    fcl_pool = fcl[shared_feature_cols].copy()
    fcl_pool['__source__'] = 'FCL-2025 target'
    pooled = pd.concat([df_pool, fcl_pool], ignore_index=True)

    imputer_pool = SimpleImputer(strategy='median')
    scaler_pool = StandardScaler()
    X_pool_scaled = scaler_pool.fit_transform(imputer_pool.fit_transform(pooled[shared_feature_cols]))

    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=20, min_dist=0.3)
    embedding = reducer.fit_transform(X_pool_scaled)
    pooled['umap_1'] = embedding[:, 0]
    pooled['umap_2'] = embedding[:, 1]

    best_k, best_score, best_labels = None, -1, None
    sil_scores = {}
    for k in range(3, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(embedding)
        score = silhouette_score(embedding, labels)
        sil_scores[k] = score
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    pooled['mech_archetype'] = best_labels.astype(str)
    print(f'Silhouette by k: {sil_scores}')
    print(f'Chosen k={best_k} clusters (silhouette={best_score:.3f})')

    n_train = len(df)
    df['umap_1'] = pooled['umap_1'].values[:n_train]
    df['umap_2'] = pooled['umap_2'].values[:n_train]
    df['mech_archetype'] = pooled['mech_archetype'].values[:n_train]
    fcl['umap_1'] = pooled['umap_1'].values[n_train:]
    fcl['umap_2'] = pooled['umap_2'].values[n_train:]
    fcl['mech_archetype'] = pooled['mech_archetype'].values[n_train:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=pooled, x='umap_1', y='umap_2', hue='mech_archetype', palette=PALETTE,
                     alpha=0.6, s=20, ax=axes[0], legend='full')
    axes[0].set_title(f'Mechanical Archetypes (k={best_k} clusters)')
    sns.scatterplot(data=pooled, x='umap_1', y='umap_2', hue='__source__',
                     palette=[PHI_BLUE, PHI_RED], alpha=0.5, s=20, ax=axes[1])
    axes[1].set_title('Labeled Pitches vs FCL-2025 Target Pitches (Shared Embedding)')
    savefig('10_umap_archetypes.png')

    archetype_velo = df.groupby('mech_archetype')[TARGET].mean().sort_values(ascending=False)
    print('Mean velocity by mechanical archetype (labeled data only):')
    print(archetype_velo)

    df['pitcher_handedness_num'] = (df['pitcher_handedness'] == 'r').astype(float)
    fcl['pitcher_handedness_num'] = (fcl['pitcher_handedness'] == 'r').astype(float)

    ctx['df'] = df
    ctx['fcl'] = fcl
    return ctx
