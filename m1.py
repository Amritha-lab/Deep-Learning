"""
This mock test requires knowledge of:
1. Preprocessing regression data with oversampling------(5 marks)
2. Creating a neural network with specific optimizers, loss functions, and L1 regularization------(5 marks)
3. Logging training details------(3 marks)
4. Evaluating the model with multiple performance metrics------(4 marks)
5. Plotting accuracy and loss curves------(3 marks)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE
import os
import datetime
import csv
import time

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    return X, y

def preprocess_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y_train.reshape(-1, 1)).ravel()

    smote = SMOTE(random_state=42)
    X_train_resampled, y_binned_resampled = smote.fit_resample(X_train_scaled, y_binned)

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_train_scaled)
    _, indices = nn.kneighbors(X_train_resampled)
    y_train_resampled = y_train[indices.ravel()]

    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    train_dataset = RegressionDataset(X_train, y_train)
    val_dataset = RegressionDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def add_l1_regularization(model, loss, lambda_l1=0.001):
    l1_reg = torch.tensor(0., requires_grad=True).to(device)
    for param in model.parameters():
        l1_reg = l1_reg + torch.sum(torch.abs(param))
    return loss + lambda_l1 * l1_reg

class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self.rows = []

    def append(self, metrics):
        self.rows.append(metrics)

    def save(self):
        with open(self.filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

def train_model(model, train_loader, val_loader, num_epochs=100):
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLogger(os.path.join(log_dir, 'training.log'))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_mae = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss = add_l1_regularization(model, loss)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_mae += torch.sum(torch.abs(outputs - y_batch)).item()

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_mae += torch.sum(torch.abs(outputs - y_batch)).item()

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        csv_logger.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "train_mae": train_mae, "val_mae": val_mae})

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    csv_logger.save()
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth')))
    return model, history

def evaluate_model(model, X_test, y_test):
    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)
    with torch.no_grad():
        predictions = model(X_test)
    predictions = predictions.cpu().numpy()
    y_true = y_test.cpu().numpy()
    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")
    return {"mse": mse, "mae": mae, "r2": r2}

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('MAE over Epochs')

    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.show()

def main():
    print("Loading data...")
    X, y = load_data()

    print("Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)

    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val)

    print("Creating model...")
    input_size = X_train.shape[1]
    model = RegressionModel(input_size).to(device)
    print(model)

    print("Training model...")
    model, history = train_model(model, train_loader, val_loader)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("Plotting training history...")
    plot_training_history(history)

    print("Assignment completed!")

main()