# rusty-bargain-car-prices
Used-car price regression for a market-value app, comparing Linear Regression, Decision Tree, Random Forest, and LightGBM on prediction quality, training time, and inference speed. Random Forest wins on accuracy (test RMSE ≈ 1,620); LightGBM wins the speed-quality tradeoff.
# Rusty Bargain Used Car Price Prediction

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DustinBlancas/rusty-bargain-car-prices/blob/main/rusty_bargain_car_prices.ipynb)
[![Render via nbviewer](https://img.shields.io/badge/render-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/DustinBlancas/rusty-bargain-car-prices/blob/main/rusty_bargain_car_prices.ipynb)

> 👉 **Click a badge above to view the rendered notebook with plots, tables, and results.** GitHub's in-browser preview can be unreliable for larger notebooks; the badges always work.

---

Regression model that estimates the market value of a used car for **Rusty Bargain**, a used-car sales service building a mobile app where sellers can get an instant price quote. The project compares five models on a three-way tradeoff: **prediction quality, training time, and inference speed.**

## Problem

Rusty Bargain's app needs to return a price estimate fast enough to feel instant in a phone UI, but accurate enough that sellers trust the number. That makes raw RMSE only part of the story — a model that's 5% more accurate but 100× slower at prediction time may not be the right pick. The deliverable is a justified recommendation based on all three axes.

**Requirements from the brief:**
- Quality (RMSE)
- Speed of prediction
- Training time (matters for periodic retraining)

## Data

Historical used-car records with technical specifications, registration year, brand, model, fuel type, vehicle type, gearbox, mileage, and the target `Price` (in €).

**Preprocessing:**
- Removed outliers: price kept to €100 – €200,000; registration year to 1900 – 2023; horsepower to 1 – 1000.
- Filled missing categorical values with `unknown`.
- Engineered a `vehicle_age` feature from `RegistrationYear`.
- **Ordinal-encoded** categorical columns for tree-based models.
- **60 / 20 / 20** train / validation / test split. Hyperparameter tuning was done strictly on the validation set; the test set was touched once for the final unbiased evaluation.

## Methodology

Five models trained and compared:

| Model | Tuning |
|---|---|
| Linear Regression | Baseline (sanity check) |
| Decision Tree | `max_depth` ∈ {5, 10, 15, 20, None} |
| Random Forest | `n_estimators` ∈ {50, 100}, `max_depth` ∈ {10, 20} |
| LightGBM | `n_estimators`, `max_depth`, `learning_rate`, `num_leaves` |
| CatBoost | Default config (bonus) |

All hyperparameters selected on the **validation set**, never on the test set.

## Results

| Model | Validation RMSE | Test RMSE | Train time | Predict time |
|---|---|---|---|---|
| Linear Regression | 2,953.26 | 2,973.45 | 0.04 s | 0.004 s |
| Decision Tree (depth=15) | 1,935.47 | 1,943.99 | 0.46 s | 0.01 s |
| **Random Forest (n=100, depth=20)** | **1,604.75** | **1,620.14** | 19.5 s | 0.79 s |

**Best model by accuracy: Random Forest** — Test RMSE ≈ **€1,620** on car prices that average several thousand euros, with both `Power` and `RegistrationYear` consistently the most important features.

**Sanity check passed:** every advanced model beat Linear Regression on test RMSE.

**Validation vs. test gap was small** for every model (within ~15 RMSE points), confirming the tuning didn't overfit to the validation set.

## Recommendation for the app

Trade-off summary:

- **If accuracy is the priority** → Random Forest wins on RMSE.
- **If inference speed is the priority** → Linear Regression / Decision Tree are 100× faster at predict time, at a meaningful accuracy cost.
- **For a mobile app's real-world tradeoff** → **LightGBM** typically offers the best balance of accuracy and speed, trains far faster than Random Forest, and predicts in milliseconds — the recommended choice for production.

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `LightGBM` · `CatBoost` · `matplotlib`

## Repository

- [`rusty_bargain_car_prices.ipynb`](./rusty_bargain_car_prices.ipynb) — full notebook: data prep, model tuning, comparison table, and recommendation
