# Contributing

## Development workflow

1. Create a fresh virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run the public verification commands before opening a PR:
   - `python3 -m unittest tests.test_policy_behavior -v`
   - `python3 -m pytest tests/ -v`
4. Keep the public workflow centered on local Visual Layer JSON exports.

## Pull request expectations

- Do not change public artifact names without updating:
  - `README.md`
  - `clean_imagenet1k/README.md`
  - `clean_imagenet1k/imagenet1k_cleaning_chunked.py`
  - tests under `tests/`
- Keep optional outputs clearly separated from guaranteed outputs.
- Preserve fixture-based reproducibility for the public CLI.
