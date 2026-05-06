"""
Bi-directioneel LSTM (Bi-LSTM) met PyTorch
"""
# %% Section 1: Prepare the workspace
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


# %% Section 2: Configuration
SEQ_LEN     = 24        # Timesteps per sequence (window size)
HIDDEN_SIZE = 64        # Number of neurons in the LSTM hidden layers
NUM_LAYERS  = 2         # Number of LSTM layers
BATCH_SIZE  = 32        # Number of data samples per learning batch
EPOCHS      = 10        # Number of training rounds
LR          = 1e-3      # Learning rate
DROPOUT     = 0.3       # Dropout rate between LSTM layers
SEED        = 42        # make reproducible
NUM_CLASSES = 72        # Number of output classes (24 hours * 3 days = 72 hours to predict)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Use GPU if available


# %% Section 3: Definitions
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        # Linear layer output size is hidden_size * 2 due to bidirectionality
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        
        # We take the last time step of the sequence for prediction
        out = out[:, -1, :] 
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

def create_sequences(X, y, seq_length, pred_steps):                 # Convert flat data into 3D sequences for LSTM input
    xs, ys = [], []
    for i in range(len(X) - seq_length - pred_steps):
        x_block = X[i : i + seq_length]
        y_target = y[i + seq_length : i + seq_length + pred_steps]  # Next pred_steps hours as target
        xs.append(x_block)
        ys.append(y_target)
    return np.array(xs), np.array(ys)

def train_epoch(model, loader, optimizer, criterion, device):   # Handles the training logic for one single epoch.
    model.train()
    total_loss, total_samples = 0.0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        
        # Ensure outputs and targets have the same shape [batch, 1]
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)

    return total_loss / total_samples

def evaluate(model, loader, criterion, device):         # Handles the validation logic.
    model.eval()
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * len(y_batch)
            total_samples += len(y_batch)

    return total_loss / total_samples


# %% Section 4: Data Preparation
data = pd.read_csv("merged_hourly.csv")

X = data.iloc[:, 0:10].values                   # Feature selection
y = data.iloc[:, -1].values                     # Target as column-vector for scaler
m, n = X.shape

total_days = len(data) // 24
val_days = 7
train_days = total_days - val_days

m_train = train_days * 24
m_val = val_days * 24

X_train = X[:m_train, :]
y_train = y[:m_train]
X_val = X[m_train:m_train + m_val, :]
y_val = y[m_train:m_train + m_val]

# Normalisation of features (X)
mu_x = np.mean(X_train, axis=0)
sigma_x = np.std(X_train, axis=0)
X_train = (X_train - mu_x) / sigma_x
X_val = (X_val - mu_x) / sigma_x

# Normalisation of Target (y)
mu_y = np.mean(y_train)
sigma_y = np.std(y_train)
y_train = (y_train - mu_y) / sigma_y
y_val = (y_val - mu_y) / sigma_y

# Maak de 3D blokken voor training en validatie
X_train_3D, y_train_3D = create_sequences(X_train, y_train, SEQ_LEN, NUM_CLASSES)
X_val_3D, y_val_3D     = create_sequences(X_val, y_val, SEQ_LEN, NUM_CLASSES)

# Converteren naar PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_3D).float()
y_train_tensor = torch.from_numpy(y_train_3D).float()
X_val_tensor   = torch.from_numpy(X_val_3D).float()
y_val_tensor   = torch.from_numpy(y_val_3D).float()

INPUT_SIZE = X_train_tensor.shape[2]                    # Number of features per timestep (input dimension)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
val_ds   = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)


# %% Section 5: Define the BiLSTM model
model = BiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
               NUM_CLASSES, DROPOUT).to(device)
print(model)
print(f"\nAantal parameters: {sum(p.numel() for p in model.parameters()):,}\n")


# %% Section 6: Loss, Optimizer & Scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# %% Section 7: Training Loop
print(f"{'Epoch':>6} | {'Train MSE':>13} | {'Val MSE':>11}")
print("-" * 40)

train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    trn_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    train_losses.append(trn_loss)
    val_losses.append(val_loss)
    scheduler.step()

    print(f"{epoch:>6} | {trn_loss:>13.4f} | {val_loss:>11.4f}")


# %% Section 8: Save the model
PAD_PATH = "bi_lstm_model.pt"
torch.save(model.state_dict(), PAD_PATH)
print(f"\nModel saved as '{PAD_PATH}'")


# %% Section 9: Example Prediction
model.eval()
example = X_val_tensor[0].unsqueeze(0).to(device)  # Select the first validation sample

with torch.no_grad():
    prediction = model(example)
    predicted_prices = (prediction.squeeze().cpu().numpy() * sigma_y) + mu_y  # Convert back to original scale
    for uur, prijs in enumerate(predicted_prices, start=1):
        print(f"  Uur {uur:>3}: €{prijs:.2f}")


# %% Section 10: Plot training and validation loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# %% Section 11: Plot predictions vs actuals for the validation set
model.eval()
all_preds = []
all_actuals = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        
        # Terugrekenen naar echte prijzen
        preds = (outputs.cpu().numpy() * sigma_y) + mu_y
        actuals = (y_batch.numpy() * sigma_y) + mu_y
        
        all_preds.extend(preds)
        all_actuals.extend(actuals)

all_preds = np.array(all_preds).flatten()
all_actuals = np.array(all_actuals).flatten()

# Plot the validation set
plt.figure(figsize=(15, 6))
plt.plot(all_actuals[:m_val], label='Real Price', color='blue')
plt.plot(all_preds[:m_val], label='Predicted Price', color='red', linestyle='--')
plt.title('Energy Price Prediction vs Actuals')
plt.ylabel('Price (€)')
plt.xlabel('Time (hours)')
plt.legend()
plt.show()