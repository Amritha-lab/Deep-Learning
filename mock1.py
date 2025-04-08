"""
This mock test requires knowledge of:
1. Preprocessing regression data with oversampling------(5 marks)
2. Creating a neural network with specific optimizers, loss functions, and L1 regularization------(5 marks)
3. Logging training details------(3 marks)
4. Evaluating the model with multiple performance metrics------(4 marks)
5. Plotting accuracy and loss curves------(3 marks)

Instructions:
- Fill in the sections marked with "TODO" comments
- Do not modify the structure of the code or function signatures
- Submit your completed script
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
    """
    Load and return a regression dataset.
    we'll use the California Housing dataset.
    """
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    return X, y

def preprocess_data(X, y):
    """
    Preprocess the data:
    1. Split into train/validation/test sets
    2. Scale features
    3. Apply oversampling to training data

    Returns preprocessed data splits.
    """
    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Further split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # TODO: Standardize the features
    # Create a StandardScaler and fit it on the training data only
    # Then apply the transformation to train, validation, and test data
     # TODO: Apply oversampling to the training data using SMOTE
    # This is somewhat unusual for regression, but can help with imbalanced regression problems
    # Hint: For regression, you'll need to create bins of the target variable

    # TODO: Convert the numpy arrays to PyTorch tensors
    # Make sure to convert them to torch.float32
    # 3. Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Apply SMOTE for regression by binning target
    '''discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y_train.reshape(-1, 1)).ravel()

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train, y_binned)'''

     # 5. Convert to PyTorch tensors
    '''X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)'''

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


class RegressionDataset(Dataset):
    """
    Dataset class for regression data.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create PyTorch DataLoaders for training and validation.
    """
    # TODO: Create Dataset objects and DataLoaders for training and validation sets
    # Use the RegressionDataset class defined above
    # Set appropriate batch sizes
    train_dataset = RegressionDataset(X_train, y_train)
    val_dataset = RegressionDataset(X_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

class RegressionModel(nn.Module):
    """
    Neural network model for regression.
    """
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)  # Output layer (no activation for regression)
            )




        # TODO: Define the layers of the network
        # - At least 3 fully-connected layers with appropriate activations
        # - Dropout layers for regularization
        # - Output layer with no activation (for regression)


    def forward(self, x):
        # TODO: Implement the forward pass
        return self.model(x)


        return x

def add_l1_regularization(model, loss, lambda_l1=0.001):
    """
    Add L1 regularization to the loss function.
    """
    l1_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l1_reg = l1_reg + torch.sum(torch.abs(param))
    return loss + lambda_l1 * l1_reg


class CSVLogger:
    """
    Simple CSV logger for PyTorch training.
    """
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


def train_model(model, train_loader, val_loader, num_epochs=100, patience=None):
    """
    Train the model and log the training details.
    Returns training history.
    """
    # Setup logging
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_mae', 'val_mae'])

    # TODO: Define loss function and optimizer
    # Use an appropriate loss function for regression
    # Use Adam optimizer with learning rate 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# Setup tracking variables for history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }

    # TODO: Implement the training loop
    # - Iterate through epochs and batches
    # - Calculate and log loss and metrics
    # - Implement early stopping
    # - Save the best model

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(log_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        model.train()
        train_losses, train_maes = [], []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_maes.append(torch.mean(torch.abs(outputs - batch_y)).item())

        # Validation
        model.eval()
        val_losses, val_maes = [], []
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_outputs = model(val_X).squeeze()
                val_loss = criterion(val_outputs, val_y)
                val_mae = torch.mean(torch.abs(val_outputs - val_y))

                val_losses.append(val_loss.item())
                val_maes.append(val_mae.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        train_mae = sum(train_maes) / len(train_maes)
        val_mae = sum(val_maes) / len(val_maes)

        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        csv_writer.writerow([epoch + 1, train_loss, val_loss, train_mae, val_mae])
        csv_file.flush()

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
              f"Train MAE: {train_mae:.4f} - Val MAE: {val_mae:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    csv_file.close()

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    return model, history



def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using multiple performance metrics.
    """
    model.eval()  # Set model to evaluation mode

    # TODO: Convert X_test and y_test to PyTorch tensors if they aren't already



    # TODO: Make predictions on test data
    # Don't forget to use torch.no_grad()
    model.eval()  # Set model to evaluation mode

    # Convert to PyTorch tensors if they aren't already
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32)

    X_test, y_test = X_test.to(device), y_test.to(device)

    # Disable gradient tracking for evaluation
    with torch.no_grad():
        outputs = model(X_test).squeeze()

    # Move results to CPU for metric computation
    y_true = y_test.cpu().numpy()
    y_pred = outputs.cpu().numpy()

    # Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    metrics = {
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }


    # TODO: Calculate and print at least 3 different regression metrics
    # (e.g., MSE, MAE, R²)
    return metrics

def plot_training_history(history):
    """
    Plot the training and validation loss curves.
    """
    # TODO: Create a figure with two subplots
    # - First subplot: Plot training and validation loss over epochs
    # - Second subplot: Plot any metrics tracked during training

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_mae'], label='Train MAE', color='green')
    plt.plot(epochs, history['val_mae'], label='Val MAE', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training vs Validation MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.show()


def main():
    """
    Main function to run the entire pipeline.
    """
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