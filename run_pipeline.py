"""End-to-end pipeline: FCL-2025 "hardest pitch in five years" velocity projection.

Runs every stage of the analysis in order, threading a single shared `ctx` dict through each
stage (each stage reads what it needs from `ctx` and adds its own results back into it). This
mirrors the original exploratory notebook (see notebooks/phillies_velocity_model.ipynb) section
by section, but as plain, importable, runnable Python modules.

Usage:
    python run_pipeline.py

Requires the data files described in README.md to be present under ./data/ (not included in
this repository -- the underlying biomechanics data was provided privately for a Phillies
take-home assignment and is not mine to publish).

Outputs:
    figures/   -- every chart the analysis produces (24 PNGs)
    output/    -- fcl_2025_pitcher_rankings.csv and the ranking-validation CSVs
"""
from src import (archetypes, case_studies, consensus_ranking, data_loading, eda, explainability,
                  feature_engineering, feature_selection, final_predictions, final_ranking, modeling,
                  simulation, uncertainty, validation)


def main():
    ctx = {}

    print('\n=== Stage 1: Data loading ===', flush=True)
    ctx = data_loading.run(ctx)

    print('\n=== Stage 2: Exploratory analysis ===', flush=True)
    ctx = eda.run(ctx)

    print('\n=== Stage 3: Feature engineering ===', flush=True)
    ctx = feature_engineering.run(ctx)
    ctx = eda.run_fatigue_and_development(ctx)

    print('\n=== Stage 4: Mechanical archetypes (UMAP + KMeans) ===', flush=True)
    ctx = archetypes.run(ctx)

    print('\n=== Stage 5: Feature selection ===', flush=True)
    ctx = feature_selection.run(ctx)

    print('\n=== Stage 6: Model comparison + tuning ===', flush=True)
    ctx = modeling.run(ctx)

    print('\n=== Stage 7: Explainability ===', flush=True)
    ctx = explainability.run(ctx)

    print('\n=== Stage 8: Uncertainty layers ===', flush=True)
    ctx = uncertainty.run(ctx)

    print('\n=== Stage 9: Monte Carlo + extreme-value simulation ===', flush=True)
    ctx = simulation.run(ctx)

    print('\n=== Stage 10: Ranking validation ===', flush=True)
    ctx = validation.run(ctx)

    print('\n=== Stage 11: Consensus ranking ===', flush=True)
    ctx = consensus_ranking.run(ctx)

    print('\n=== Stage 12: Case studies ===', flush=True)
    ctx = case_studies.run(ctx)

    print('\n=== Stage 13: Explicit top-3 predictions ===', flush=True)
    ctx = final_predictions.run(ctx)

    print('\n=== Stage 14: Final ranking ===', flush=True)
    ctx = final_ranking.run(ctx)

    print('\nDone. See figures/ for all charts and output/ for the final ranking CSV.', flush=True)
    return ctx


if __name__ == '__main__':
    main()
