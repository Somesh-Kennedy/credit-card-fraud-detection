# Credit Card Fraud Detection

An end-to-end fraud detection pipeline using the provided Kaggle credit card transaction dataset.

## Project structure

- `src/`
  - `data.py` - data loading and preprocessing
  - `model.py` - training and evaluation pipeline
  - `utils.py` - helper functions for feature engineering and metrics
- `notebooks/`
  - `eda.ipynb` - exploratory data analysis and visualization
- `requirements.txt` - Python dependencies
- `README.md` - project overview and setup instructions
- `credit_card_transactions.csv` - dataset

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Run training:

   ```powershell
   python src/model.py
   ```

## Notes

- Uses SMOTE for class balancing.
- Contains numeric and categorical feature engineering.
- Targets `is_fraud` for binary classification.
