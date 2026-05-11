# =========================================================
# ENERGY PRICE PREDICTION USING FFNN (PYTORCH)
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. CONFIGURATION
# =========================================================

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15

HIDDEN_1 = 128
HIDDEN_2 = 64
HIDDEN_3 = 32
DROPOUT = 0.30

BEST_MODEL_PATH = "best_ffnn_model.pt"
FINAL_MODEL_PATH = "ffnn_energy_model.pt"

PREDICTION_SAMPLES = 150

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. DATA PREPARATION
# =========================================================

def load_data():
    train_df = pd.read_csv("train_raw.csv")
    val_df = pd.read_csv("val_raw.csv")
    test_df = pd.read_csv("test_raw.csv")

    print("Datasets loaded successfully")
    print("Train Shape:", train_df.shape)
    print("Validation Shape:", val_df.shape)
    print("Test Shape:", test_df.shape)

    return train_df, val_df, test_df


def split_features_and_target(train_df, val_df, test_df):
    test_timestamps = pd.to_datetime(test_df["timestamp"]) + pd.Timedelta(hours=1)

    # The timestamp is useful for indexing, but it should not be used as a model input.
    train_df = train_df.drop(columns=["timestamp"])
    val_df = val_df.drop(columns=["timestamp"])
    test_df = test_df.drop(columns=["timestamp"])

    X_train = train_df.drop(columns=["target"]).values
    y_train = train_df["target"].values.reshape(-1, 1)

    X_val = val_df.drop(columns=["target"]).values
    y_val = val_df["target"].values.reshape(-1, 1)

    X_test = test_df.drop(columns=["target"]).values
    y_test = test_df["target"].values.reshape(-1, 1)

    print("\nFeature Matrix Shape:", X_train.shape)
    print("Target Shape:", y_train.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, test_timestamps


def scale_data(X_train, y_train, X_val, y_val, X_test, y_test):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_y


def create_loader(X, y, shuffle=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


# =========================================================
# 3. MODEL DEFINITION
# =========================================================

class FFNN(nn.Module):
    def __init__(self, input_size):
        super(FFNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_2, HIDDEN_3),
            nn.ReLU(),

            nn.Linear(HIDDEN_3, 1)
        )

    def forward(self, x):
        return self.network(x)


# =========================================================
# 4. TRAINING AND EVALUATION HELPERS
# =========================================================

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate_loss(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            running_loss += loss.item()

    return running_loss / len(data_loader)


def train_model(model, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = 0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate_loss(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    return train_losses, val_losses


def predict(model, data_loader, scaler_y):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)

            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    predictions = scaler_y.inverse_transform(np.array(predictions))
    actuals = scaler_y.inverse_transform(np.array(actuals))

    return actuals.flatten(), predictions.flatten()


def calculate_metrics(actuals, predictions):
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # Small epsilon avoids division by zero if electricity prices are exactly zero.
    epsilon = 1e-8
    mape = np.mean(np.abs((actuals - predictions) / np.maximum(np.abs(actuals), epsilon))) * 100

    r2 = r2_score(actuals, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


# =========================================================
# 5. VISUALIZATION HELPERS
# =========================================================

def apply_plot_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 120
    })


def save_and_close(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {filename}")


def format_time_axis():
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.DateFormatter("%a %d %b\n%H:%M")

    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=0)


def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2.5, color="#1f4e79")
    plt.plot(val_losses, label="Validation Loss", linewidth=2.5, color="#c00000")

    plt.title("Training vs Validation Loss", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_and_close("image_01_training_validation_loss.png")


def plot_actual_vs_predicted(actuals, predictions, timestamps):
    samples = min(PREDICTION_SAMPLES, len(actuals))
    x_axis = timestamps.iloc[:samples]

    plt.figure(figsize=(14, 6))
    plt.plot(
        x_axis,
        actuals[:samples],
        label="Actual Price",
        color="#111111",
        linewidth=2.7
    )
    plt.plot(
        x_axis,
        predictions[:samples],
        label="Predicted Price",
        color="#0072b2",
        linewidth=2.5,
        linestyle="--"
    )

    plt.title("Electricity Price Prediction", fontweight="bold")
    plt.xlabel("Forecast Time")
    plt.ylabel("Price (€/MWh)")
    format_time_axis()
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_and_close("image_02_prediction_vs_actual.png")


def plot_residuals(actuals, predictions, timestamps):
    samples = min(PREDICTION_SAMPLES, len(actuals))
    residuals = actuals - predictions
    x_axis = timestamps.iloc[:samples]

    plt.figure(figsize=(14, 6))
    plt.plot(
        x_axis,
        residuals[:samples],
        label="Residual Error",
        color="#7030a0",
        linewidth=2.3
    )
    plt.axhline(0, color="#111111", linewidth=1.8, linestyle="--", label="Zero Error")

    plt.title("Residual Error Plot", fontweight="bold")
    plt.xlabel("Forecast Time")
    plt.ylabel("Residual: Actual - Predicted")
    format_time_axis()
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_and_close("image_03_residual_plot.png")


def plot_scatter(actuals, predictions):
    min_value = min(actuals.min(), predictions.min())
    max_value = max(actuals.max(), predictions.max())

    plt.figure(figsize=(8, 8))
    plt.scatter(
        actuals,
        predictions,
        alpha=0.70,
        s=45,
        color="#0072b2",
        edgecolors="#222222",
        linewidths=0.3
    )
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="#c00000",
        linewidth=2.5,
        linestyle="--",
        label="Perfect Fit"
    )

    plt.title("Actual vs Predicted Prices", fontweight="bold")
    plt.xlabel("Actual Price (€/MWh)")
    plt.ylabel("Predicted Price (€/MWh)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_and_close("image_04_scatter_actual_vs_predicted.png")


def plot_metrics(metrics):
    metric_names = ["MAE", "RMSE", "MAPE"]
    metric_values = [metrics[name] for name in metric_names]
    colors = ["#1f4e79", "#0072b2", "#70ad47"]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(metric_names, metric_values, color=colors, edgecolor="#222222", linewidth=0.8)

    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

    plt.title("Evaluation Metrics", fontweight="bold")
    plt.ylabel("Metric Value")
    plt.grid(True, axis="y", alpha=0.3)

    save_and_close("image_05_evaluation_metrics.png")


# =========================================================
# 6. MAIN PROGRAM
# =========================================================

def main():
    print("Using device:", device)

    apply_plot_style()

    train_df, val_df, test_df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, test_timestamps = split_features_and_target(
        train_df,
        val_df,
        test_df
    )

    X_train, y_train, X_val, y_val, X_test, y_test, scaler_y = scale_data(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )

    train_loader = create_loader(X_train, y_train, shuffle=True)
    val_loader = create_loader(X_val, y_val, shuffle=False)
    test_loader = create_loader(X_test, y_test, shuffle=False)

    input_size = X_train.shape[1]
    model = FFNN(input_size).to(device)

    print("\nModel Architecture:\n")
    print(model)

    train_losses, val_losses = train_model(model, train_loader, val_loader)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print("\nBest model loaded successfully.")

    actuals, predictions = predict(model, test_loader, scaler_y)
    metrics = calculate_metrics(actuals, predictions)

    print("\n================================")
    print("TEST PERFORMANCE")
    print("================================")
    print(f"MAE   : {metrics['MAE']:.4f}")
    print(f"RMSE  : {metrics['RMSE']:.4f}")
    print(f"MAPE  : {metrics['MAPE']:.2f}%")
    print(f"R2    : {metrics['R2']:.4f}")

    plot_training_history(train_losses, val_losses)
    plot_actual_vs_predicted(actuals, predictions, test_timestamps)
    plot_residuals(actuals, predictions, test_timestamps)
    plot_scatter(actuals, predictions)
    plot_metrics(metrics)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\nFinal model saved successfully: {FINAL_MODEL_PATH}")

    print("\nSample Predictions:\n")
    for i in range(min(10, len(actuals))):
        print(
            f"Sample {i + 1:02d} | "
            f"Actual: {actuals[i]:.2f} | "
            f"Predicted: {predictions[i]:.2f} | "
            f"Residual: {actuals[i] - predictions[i]:.2f}"
        )


if __name__ == "__main__":
    main()
