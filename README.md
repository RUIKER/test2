# NGAFID Maintenance Event Detection (2days Binary Benchmark)

This repository reproduces maintenance event binary detection from the NGAFID dataset benchmark subset (`2days`):

- Task: `before maintenance` vs `after maintenance`
- Validation: fixed 5-Fold Stratified Cross-Validation
- Model: MiniRocket + RidgeClassifierCV
- Input shape: `[samples, timesteps, features]`
- Default timesteps: `4096`

## 1. Environment Setup (Clean Environment)

Python version requirement:

- Supported: Python 3.10-3.12
- Recommended: Python 3.11
- Not supported: Python 3.14 (some core dependencies do not provide compatible wheels yet)

macOS clean setup example:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
pip install -r requirements.txt
```

## 2. One-Command Run

```bash
python main.py
```

What `main.py` does:

1. Auto-downloads `2days` dataset (Zenodo first, Google Drive fallback)
2. Loads local benchmark subset from `data/subset_data/2days`
3. Runs fixed 5-fold CV training/evaluation
4. Saves figures to `results/`

## 3. Output and Metrics

During execution, the script prints:

- Per-fold accuracy/F1/ROC-AUC
- Final mean ± std for:
  - Accuracy
  - Weighted F1
  - ROC-AUC

Saved figures in `results/`:

- `confusion_matrix.png`
- `roc_curve.png`
- `fold_metrics_comparison.png`

## 4. Reproducibility Notes

- Cross-validation is fixed to 5 folds in code.
- Random seed is fixed to `42`.
- No hard-coded absolute local paths are required.
- Default run does not require extra environment variables.

Optional local tuning (not required for official reproduction):

- `PM_MAX_LENGTH` (example: `1024`) to reduce sequence length for resource-limited machines.

## 5. Data Source

Paper:

- `A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID`

Dataset source links:

- Zenodo: `https://doi.org/10.5281/zenodo.6624956`
- Kaggle: `https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid`

## 6. Troubleshooting

- If dependency error occurs: rerun `pip install -r requirements.txt`.
- If download times out: rerun command (fallback logic is enabled).
- If data loading fails: verify `data/subset_data/2days` contains:
  - `flight_data.pkl`
  - `flight_header.csv`
  - `stats.csv`

## 7. Clean Environment Verification

This repository includes GitHub Actions workflow:

- `.github/workflows/clean-env-verify.yml`

What CI verifies automatically:

1. New clean Ubuntu runner
2. Python 3.11 setup
3. `pip install -r requirements.txt`
4. 5-fold smoke validation using synthetic data
5. Upload result plots as workflow artifacts

Why smoke data is used in CI:

- Public runners may be unstable for long external dataset downloads.
- Smoke validation proves dependency completeness and pipeline executability in a clean environment.

For full benchmark reproduction (teacher review):

```bash
pip install -r requirements.txt
python main.py
```
