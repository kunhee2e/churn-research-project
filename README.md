# Customer Segmentation and Segment-Specific Churn Prediction

Does training a separate churn model inside each customer segment beat a single model
trained on everyone? This project tests that question on the IBM Telco Customer Churn
dataset using K-Means segmentation, three classifiers, and SHAP for interpretation.

## Research questions

- **RQ1** How can customers be segmented using RFM-style features and K-Means?
- **RQ2** Do segment-specific classifiers outperform a unified model?
- **RQ3** Which features drive churn within each segment?

## Data

IBM Telco Customer Churn (7,043 records, 21 attributes), available on
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
The CSV is not tracked in this repository. Download it and place it at:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the analysis

Run the notebooks in order. Each one saves outputs that the next one reads.

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Clean the data, explore churn by subgroup |
| `02_segmentation.ipynb` | Build RFM proxies, K-Means clustering, choose *k* |
| `03_unified_models.ipynb` | Train LR / RF / XGBoost on all customers, apply SMOTE |
| `04_segment_models.ipynb` | Train a model per segment, compare against the baseline |
| `05_shap.ipynb` | Global and per-segment SHAP feature attribution |

Figures land in `results/figures/`, tables in `results/tables/`.

## Method summary

1. Clean data (drop 11 rows with blank `TotalCharges`), one-hot encode
2. RFM proxies: tenure → Recency, service count → Frequency, `MonthlyCharges` → Monetary
3. K-Means with *k* chosen by the Elbow method and silhouette score
4. Stratified 80/20 split, SMOTE applied to the training set only
5. Two configurations: one unified model, and one model per segment
6. Evaluate with accuracy, precision, recall, F1 and AUC-ROC
7. SHAP at global and segment level

## Project structure

- `data/raw/` — source CSV, not tracked in the repository
- `data/processed/` — cleaned and segmented data written by notebooks 01 and 02
- `notebooks/` — the five analysis notebooks, run in order
- `results/figures/` — PNG figures
- `results/tables/` — CSV tables
- `requirements.txt` — pinned dependencies
- `README.md` — this file

## Notes

- `SEED = 42` throughout, but exact scores can shift slightly with library versions.
- K-Means assigns cluster numbers arbitrarily. Notebook 02 names segments from their
  profile (tenure, service count, spend) rather than from the cluster index, so the
  labels stay correct across runs.
- macOS users may hit an XGBoost import error; `brew install libomp` fixes it.
