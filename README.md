# FCL-2025 Velocity Projection: Who Throws the Hardest Pitch in Five Years?

**Question:** of the pitchers in the Phillies' 2025 Florida Complex League (FCL) cohort, who is most likely to throw the hardest pitch five years from now?

Future velocity can't be observed directly, so this isn't a simple regression problem. The approach taken here: understand what mechanical dynamics actually relate to velocity and why; discover natural mechanical groupings (archetypes) among pitchers; build a disciplined, validated model on top of that understanding; and rank pitchers from a **consensus of three independent methods** plus an explicit, simulation-based treatment of uncertainty -- rather than a single point prediction.

A full write-up of the methodology, findings, and limitations is in [`report/FCL_2025_Velocity_Projection_Report.pdf`](report/FCL_2025_Velocity_Projection_Report.pdf).

## Headline result

Three structurally independent ranking methods (a biomechanical XGBoost prediction model, a nearest-comparable-pitcher framework, and a Bradley-Terry-style pairwise ranking model) converge on the same top prospects. Their consensus rank, propagated through a posterior simulation and extreme-value framework, projects:

| Pitcher | Projected ceiling (expected max, mph) | 90% interval (mph) |
|---|---:|---:|
| 212 | 98.66 | 95.78 -- 103.23 |
| 132 | 97.91 | 96.58 -- 99.95 |
| 154 | 98.57 | 96.86 -- 101.22 |

The more important finding isn't any single pitcher's number -- it's that the same biomechanical theme (lower-body force generation, rotational acceleration, and hip-shoulder/torso separation) kept surfacing independently across archetype discovery, feature selection, model explainability, comparable-pitcher matching, and three separately-built ranking methods.

## Repo structure

```
.
├── README.md
├── requirements.txt
├── run_pipeline.py              <- run the whole analysis end to end
├── data/                        <- not included; see "Data" below
├── src/                         <- the analysis, broken into one module per pipeline stage
│   ├── config.py                   shared constants, paths, color palette, I/O helpers
│   ├── data_loading.py             Stage 1 -- load labeled + FCL-2025 datasets
│   ├── eda.py                      Stage 2 -- exploratory analysis
│   ├── feature_engineering.py      Stage 3 -- kinetic-chain-motivated engineered features
│   ├── archetypes.py               Stage 4 -- UMAP + KMeans mechanical archetypes
│   ├── feature_selection.py        Stage 5 -- ablation + dual-criterion stability selection
│   ├── modeling.py                 Stage 6 -- model comparison + hyperparameter tuning
│   ├── explainability.py           Stage 7 -- calibration, SHAP, partial dependence
│   ├── uncertainty.py               Stage 8 -- model / development / pitch-to-pitch uncertainty
│   ├── simulation.py                Stage 9 -- Monte Carlo + extreme-value simulation
│   ├── validation.py                Stage 10 -- out-of-fold ranking validation (3 methods)
│   ├── consensus_ranking.py         Stage 11 -- combine methods into one consensus rank
│   ├── case_studies.py              Stage 12 -- what makes the top pitchers stand out
│   ├── final_predictions.py         Stage 13 -- explicit top-3 ceiling projections
│   └── final_ranking.py             Stage 14 -- final ranking + CSV output
├── notebooks/
│   └── phillies_velocity_model.ipynb   <- the original exploratory analysis, in notebook form
├── figures/                      <- generated charts (populated by running the pipeline)
├── output/                       <- generated CSVs (populated by running the pipeline)
└── report/
    ├── FCL_2025_Velocity_Projection_Report.pdf
    └── FCL_2025_Velocity_Projection_Report.md
```

`src/` and `notebooks/` contain the same analysis in two forms: the notebook is the original, exploratory, cell-by-cell version (with inline commentary on every decision); `src/` is a refactored, importable, end-to-end-runnable version of the identical logic, organized so each pipeline stage can be read (or rerun) on its own.

## Methodology, in short

1. **EDA** -- correlation structure, a level/promotion confound between the two labeled seasons, and a check for pitcher overrepresentation that motivates pitcher-grouped cross-validation and inverse-pitch-count sample weighting throughout.
2. **Feature engineering** -- kinetic-chain-motivated transforms (normalized stride length, rotational/linear kinetic-energy proxies, proximal-to-distal sequencing ratios), tested rather than assumed to help.
3. **Mechanical archetypes** -- a shared UMAP + KMeans embedding fit on *pooled* labeled + FCL-2025 data (no velocity information), so archetypes describe movement patterns alone and both cohorts land in one comparable space.
4. **Disciplined feature selection** -- curated second-order interactions and physics-motivated transforms, reduced via **dual-criterion bootstrap stability selection** (Lasso for linear signal, XGBoost importance for nonlinear signal -- a feature only needs to clear one bar).
5. **Modeling** -- XGBoost beats Random Forest, Elastic Net, and a tuned neural net on the final feature set, consistent with gradient-boosted trees outperforming neural nets on small tabular data.
6. **Explainability** -- calibration/residuals, TreeSHAP, partial dependence, and a fully-transparent linear baseline, used together rather than relying on any one view.
7. **Uncertainty, modeled explicitly in three layers** -- model uncertainty (200 bootstrap refits), development uncertainty (bootstrapped year-over-year gain from a small dual-year sample), and pitch-to-pitch variability (a flat population-average sigma, after an individualized sigma model failed to generalize).
8. **Monte Carlo + extreme-value simulation** -- each posterior draw is propagated through thousands of simulated future pitches; the ranking metric is the *mean* of each pitcher's simulated-maximum distribution, checked (not assumed) against a fitted Gumbel distribution.
9. **Three independent ranking methods, validated out-of-fold** -- a prediction model, a comparable-pitcher framework, and a pairwise classifier, each scored on Spearman rank correlation and pairwise concordance (including a "close pairs only" check on the hardest 25% of comparisons), with permutation tests and bootstrap confidence intervals.
10. **Consensus ranking** -- the simple average of each method's independently-computed rank, specifically designed to penalize a pitcher who looks great under one method but has no real precedent under another, rather than let one method's optimism carry the ranking.

See the [full report](report/FCL_2025_Velocity_Projection_Report.pdf) for the complete findings, validation numbers, and stated limitations.

## Running it

```bash
pip install -r requirements.txt
python run_pipeline.py
```

This regenerates every figure (`figures/`) and the final ranking CSV (`output/fcl_2025_pitcher_rankings.csv`) from raw data through to the final ranking.

## Data

The underlying biomechanics data (`velo_and_mechanics.csv`, `fcl_mechanics_2025.csv`, `metric_descriptions.csv`) was provided privately for a Phillies take-home assignment and is **not included in this repository**. To run the pipeline yourself, place those three files in `data/`. Without them, `run_pipeline.py` will fail at Stage 1 with a clear `FileNotFoundError`.

## Limitations

Briefly (see the report for full detail): no age/maturation data, so five-year development is a population-level adjustment rather than an individualized growth curve; the multi-season development sample is small (10 pitchers); pitch-to-pitch variability is modeled at the population level after an individualized model failed to generalize; some FCL-2025 pitchers' mechanics are only sparsely represented in the labeled training population; and biomechanics explain only part of velocity -- strength, intent, fatigue, and measurement noise are unobserved.
