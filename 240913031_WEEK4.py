import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ClassificationNN(nn.Module):
    def __init__(self, input_size):
        super(ClassificationNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        return x


class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.lrelu(self.layer2(x))
        x = self.lrelu(self.layer3(x))
        x = self.layer4(x)
        return x


def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        batch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')
    return epoch_losses


def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, conf_matrix


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def classification_task():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_dataset = CustomDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = ClassificationNN(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, epochs=100)

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_pred_nn = model(X_test_tensor).numpy()
        y_pred_nn = (y_pred_nn > 0.5).astype(int)

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    print("\nNeural Network Results:")
    acc_nn, prec_nn, rec_nn, conf_mat_nn = evaluate_classification(y_test, y_pred_nn)
    print(f"Accuracy: {acc_nn:.4f}")
    print(f"Precision: {prec_nn:.4f}")
    print(f"Recall: {rec_nn:.4f}")

    print("\nLogistic Regression Results:")
    acc_lr, prec_lr, rec_lr, conf_mat_lr = evaluate_classification(y_test, y_pred_lr)
    print(f"Accuracy: {acc_lr:.4f}")
    print(f"Precision: {prec_lr:.4f}")
    print(f"Recall: {rec_lr:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(conf_mat_nn, annot=True, fmt='d', ax=ax1)
    ax1.set_title('Neural Network Confusion Matrix')
    sns.heatmap(conf_mat_lr, annot=True, fmt='d', ax=ax2)
    ax2.set_title('Logistic Regression Confusion Matrix')
    plt.tight_layout()
    plt.show()


def regression_task():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).squeeze()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).squeeze()
    train_dataset = CustomDataset(X_train_scaled, y_train_scaled)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model = RegressionNN(X.shape[1])
    criterion = nn.L1Loss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-05, eps=1e-08, weight_decay=0.0001, momentum=0.75)
    train_model(model, train_loader, criterion, optimizer, epochs=500)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_pred_nn_scaled = model(X_test_tensor).numpy()
        y_pred_nn = scaler_y.inverse_transform(y_pred_nn_scaled)

    from sklearn.linear_model import LinearRegression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train_scaled)
    y_pred_lr_scaled = lr_model.predict(X_test_scaled).reshape(-1, 1)
    y_pred_lr = scaler_y.inverse_transform(y_pred_lr_scaled)
    print("\nNeural Network Results:")
    rmse_nn, mae_nn, r2_nn = evaluate_regression(y_test, y_pred_nn)
    print(f"RMSE: {rmse_nn:.4f}")
    print(f"MAE: {mae_nn:.4f}")
    print(f"R2 Score: {r2_nn:.4f}")

    print("\nLinear Regression Results:")
    rmse_lr, mae_lr, r2_lr = evaluate_regression(y_test, y_pred_lr)
    print(f"RMSE: {rmse_lr:.4f}")
    print(f"MAE: {mae_lr:.4f}")
    print(f"R2 Score: {r2_lr:.4f}")

classification_task()
regression_task()