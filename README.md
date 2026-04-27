# Walmart Sales Prediction

This project uses data from Walmart, an American multinational retailer.

The goal is to develop a machine learning model capable of evaluating the weekly sales of their stores.

This model should also allow them to better understand the influence of economic indicators on sales and could be used to plan future marketing campaigns.

## Results

Final model: Ridge regression (alpha = 0.013) with cyclical encoding for time features and one-hot encoded store IDs.

|    | R² | MAE | RMSE | MAPE |
|----|----|----|----|----|
| Train | 0.964 | $94,067 | $127,671 | 8.7% |
| Test | 0.944 | $119,899 | $139,172 | 12.9% |

The gap between train and test is small, which suggests the model generalizes reasonably well

## Key takeaways

* Store is the fundamental variable of this dataset. According to the CPI distribution, it appears that there are two distinct categories of Walmart stores, probably urban and rural.
* Unemployment in the area surrounding the stores is the primary factor influencing sales. Walmart stores are sensitive to the geo-economic situation of their region.
* Public holidays have a curiously negative impact on sales, which is why the variable was removed in version 05. This seems to indicate that customers anticipate the holidays and make purchases in advance, or that some stores close on public holidays.

## Approach

I built the model in 6 iterations, each isolating a single change:







1. **Baseline** — Linear regression, no Store, no Date. R² is negative.
2. **V1** — add Store as one-hot encoded categorical. R² to 0.92.
3. **V2** — add date features with cyclical encoding (sin/cos for month, day, dayofweek). Slight gain.
4. **V3** — drop dayofweek (it was noise). Same metrics.
5. **V4** — add binary `cold` / `hot` features for temperature. Metrics drop, reverted.
6. **V5** — drop `Holiday_Flag`. Significant improvement.
7. **V6** — Ridge regression with cross-validated alpha. Less overfit, more robust.

The notebook walks through the reasoning behind each version, including the dead ends (V4).

## Limitations

A few things worth being upfront about:

* It's important to understand that the model was initially trained on a small dataset of 136 rows. Therefore, the coefficients for the stores encoded using one-hot methods are likely less stable than the Ridge regularization would suggest.
* A MAPE of approximately 13% on the test set is acceptable, but insufficient for production use. The next step to optimize this model would therefore be to increase the size of the dataset.

## Installation

```bash
git clone https://github.com/<your-username>/walmart-sales-prediction.git
cd walmart-sales-prediction
pip install -r requirements.txt
jupyter notebook
```

Then open `walmart_sales_prediction.ipynb` and run the cells.

## Using the saved model

The final pipeline (preprocessing + Ridge model) is serialized in `models/walmart_sales_model_v6.joblib`. To use it on new data:

```
import joblib
import pandas as pd

pipeline = joblib.load("models/walmart_sales_model_v6.joblib")

predictions = pipeline.predict(X_new)
```

## Dataset

Source: [Kaggle Walmart Sales Forecasting](https://www.kaggle.com/datasets/yasserh/walmart-dataset), modified by Jedha 

## Stack

Python 3, pandas, scikit-learn, joblib. Full versions in `requirements.txt`.