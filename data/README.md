# Data

This folder is intentionally empty in the public repository.

The underlying biomechanics data was provided privately for a Phillies take-home assignment and
is not included here. To run `run_pipeline.py`, place the following three files in this folder:

- `velo_and_mechanics.csv` -- labeled dataset (203 pitchers, observed velocity known)
- `fcl_mechanics_2025.csv` -- FCL-2025 target dataset (71 pitchers, velocity unknown)
- `metric_descriptions.csv` -- column name -> description/units lookup for both files above

Without these files, `run_pipeline.py` will fail at Stage 1 with a `FileNotFoundError` pointing
at whichever file is missing.
