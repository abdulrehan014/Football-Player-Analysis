# Football Player Data Analysis

Statistical analysis and market-value prediction for football player attributes, delivered as an interactive **Streamlit** web app.


## What it does

Works on `players.csv` (3,120 player records, 65 raw columns), cleans the messy text-formatted fields, and lets you explore the data through eight tabs:

| Tab | Contents |
|---|---|
| **Overview** | Key metrics, data preview, missing values, dtypes, cleaned-data download |
| **Descriptive stats** | Mean, median, mode, variance, std, quartiles, IQR, skewness, kurtosis |
| **Visualizations** | Histogram, box plot, pie/bar, scatter (with trendline), average rating by age |
| **Sampling & CLT** | Simple random, stratified and systematic sampling vs the population; Central Limit Theorem demo |
| **Hypothesis tests** | Confidence interval, one-sample Z-test and t-test |
| **Correlation & ANOVA** | Pearson/Spearman heatmaps, covariance matrix, correlation tests, two-way ANOVA (foot × age group) |
| **Value prediction** | Linear, Ridge, Lasso, Decision Tree and Random Forest regressors with R², RMSE, actual-vs-predicted plot and feature importance |
| **Player explorer** | Name search and radar-chart comparison of up to four players |

Sidebar filters (age, overall rating, best position, foot) apply to every tab.

## Key results

| Model | R² | RMSE (€) | MAE (€) |
|---|---|---|---|
| Multiple Linear Regression | 0.7498 | 8,918,305 | 5,791,151 |
| Ridge Regression | 0.7498 | 8,918,312 | 5,791,123 |
| Lasso Regression | 0.7498 | 8,918,305 | 5,791,151 |
| Decision Tree (max_depth=5) | 0.9082 | 5,402,629 | 2,442,262 |
| **Random Forest (100 trees, max_depth=5)** | **0.9519** | **3,908,287** | **1,733,222** |

Predictors: Overall rating, Potential, Age, Wage → target: Value. 80/20 split, `random_state=42`.

Other findings:
- Mean overall rating is 73.18 (95% CI 72.95–73.42); H₀: μ = 73 is not rejected (p ≈ 0.13).
- Age group significantly affects overall rating (two-way ANOVA); foot preference and the interaction do not.
- Overall rating vs Value: Pearson r = 0.68 but Spearman ρ = 0.88, a strongly monotonic but non-linear relationship.
- The data contains 371 exact duplicate rows; results are unchanged in substance when they are removed (Random Forest R² = 0.962).


## Getting started

```bash
git clone <your-repo-url>
cd <repo-folder>

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The app opens at <http://localhost:8501>. Place `players.csv` next to `app.py`, or upload it from the sidebar. The first load can take up to a minute while the libraries import.

To regenerate the report figures and numbers without the GUI: `python analysis.py` (writes to `figs/` and `results.json`).

## Project structure

```
├── app.py              # Streamlit GUI
├── analysis.py         # Reproduces the report's tables, figures and results
├── players.csv         # Dataset
├── requirements.txt
├── screenshots/        # README images
└── Football_Player_Data_Analysis.ipynb   # Exploratory notebook
```

## Data cleaning

- Dropped the empty `Unnamed: 64` column.
- Parsed money fields (`€1.5M`, `€900K`) into euro amounts with a regex (no `eval`).
- Reduced rating entries like `87+1` to the base rating; values above 100 become NaN.
- Extracted height (cm) and weight (kg) into numeric columns; split team name from contract.
- Rows with missing values are excluded per analysis; no imputation.

## Limitations

- Only four numeric predictors, a single train–test split, no cross-validation or hyper-parameter tuning.
- Wage is closely tied to Value, so the models show association, not causation.
- Duplicate rows are kept in the main pipeline.

## Tech stack

Python · pandas · NumPy · SciPy · statsmodels · scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit

## Author

Abdul Rehan
