# Electricity Price Forecasting Project

This project predicts electricity prices using neural network models.

The main FFNN model is inside:

```text
FFNN_workspace/ffnn_code.py
```

## What This Project Does

The FFNN script trains a Feed Forward Neural Network to predict the next electricity price value.

The target column is:

```text
target
```

The target represents the next-hour ENTSO-E electricity price.

The `timestamp` column is only used for graph labels. It is not used as a model input.

## Dataset Context

This project currently uses a working sample dataset from the rich Plan B preprocessing pipeline.

The dataset covers the period from December 2025 to February 2026. The time window is small, but the dataset is already cleaned, scaled, engineered, and split chronologically.

Current split sizes:

```text
Train: 1006 rows
Validation: 215 rows
Test: 217 rows
```

Each file has 75 columns in total, including the target column. The model uses 74 engineered input features.

The features include:

- lagged price features
- 1 hour, 24 hour, and 168 hour price lags
- rolling statistics
- cyclical time encoding
- weather data
- wind data
- solar data
- calendar features
- load forecast and load actual values
- market variables such as gas and carbon prices
- cross-border electricity flows
- other engineered numerical features

The project also includes scalers for X and Y in the preprocessing pipeline, so predictions can be converted back to the original price scale.

Important: this dataset only covers around 3 months of data. It is useful for checking the code, testing the model architecture, and confirming that the pipeline works. It should not be used to judge final model performance.

For proper long-history training, the project will still need Lochan's Option A pipeline with around 17 months of basic features.

## Files

```text
FFNN_workspace/
  ffnn_code.py
  train_raw.csv
  val_raw.csv
  test_raw.csv
```

When the code runs, it also creates report images:

```text
image_01_training_validation_loss.png
image_02_prediction_vs_actual.png
image_03_residual_plot.png
image_04_scatter_actual_vs_predicted.png
image_05_evaluation_metrics.png
```

These images are overwritten every time the script runs.

## How To Run

Open a terminal in the project folder.

Then go into the FFNN workspace:

```bash
cd FFNN_workspace
```

Run the script:

```bash
python ffnn_code.py
```

If you are using the included virtual environment from the local workspace, you can run:

```bash
..\venv\Scripts\python.exe ffnn_code.py
```

## Output

The script prints:

- training loss
- validation loss
- learning rate
- MAE
- RMSE
- MAPE
- R2 score
- sample predictions

It also saves the graphs listed above.

## Model Features

The FFNN implementation includes:

- data scaling
- early stopping
- learning rate scheduling
- test evaluation
- professional matplotlib graphs
- residual error analysis
- actual vs predicted comparison

## Notes

Electricity prices are difficult to predict because they can change suddenly due to demand, weather, market conditions, and energy supply changes.

The FFNN model is useful for learning overall trends and cyclical patterns, but sudden price spikes may still be difficult to predict.
