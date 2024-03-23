import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x_data: ArrayLike, y_data: ArrayLike) -> None:
        if y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)
        self.x_data: Tensor = torch.from_numpy(x_data).float()
        self.y_data: Tensor = torch.from_numpy(y_data).float()

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.x_data[index], self.y_data[index]

    def __len__(self) -> int:
        return len(self.x_data)


def estimate_cond_mean(X: ArrayLike, Y: ArrayLike) -> NeuralNetwork:
    logger = logging.getLogger("ovg-estimator-cond-mean")
    logger.info("Training neural network to estimate a conditional mean")
    train_loader, test_loader = build_train_test_loaders(X, Y)

    input_size = X.shape[1]
    hidden_size = 64
    output_size = 1
    model = NeuralNetwork(input_size, hidden_size, output_size)
    train(model, train_loader)
    test(model, test_loader)

    return model


def estimate_cond_skew(
    model_regression_source: NeuralNetwork,
    X_source: ArrayLike,
    Y_source: ArrayLike,
    X_target: ArrayLike,
    hidden_size: int = 64,
    lr: float = 0.01,
    num_epochs: int = 50,
) -> Tuple[NeuralNetwork, StandardScaler]:
    logger = logging.getLogger("ovg-estimator-cond-skew")
    X_target_mean = np.mean(X_target)
    X_target_skew = np.mean((X_target - X_target_mean) ** 3)
    assert X_target_skew != 0

    z = (
        1
        / X_target_skew
        * (
            (
                Y_source
                - model_regression_source(torch.from_numpy(X_source).float())
                .detach()
                .numpy()
            )
            ** 3
        )
    )

    z_scaler = StandardScaler()
    logger.info("training neural network to estimate a conditional skew")
    train_loader, test_loader = build_train_test_loaders(X_source, z, y_scaler=z_scaler)

    input_size = X_source.shape[1]
    output_size = 1
    model = NeuralNetwork(input_size, hidden_size, output_size)
    train(model, train_loader, num_epochs=num_epochs, lr=lr)
    test(model, test_loader)

    return model, z_scaler


def train(
    model: NeuralNetwork,
    train_loader: DataLoader,
    verbose: bool = True,
    num_epochs: int = 10,
    lr: float = 0.01,
    weight_decay: float = 1.0e-4,
) -> float:
    logger = logging.getLogger("ovg-training")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        avg_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        avg_loss /= len(train_loader)
        if verbose and (epoch + 1) % 2 == 0:
            logger.info(f"Epoch: {epoch + 1}\tLoss: {avg_loss:.4f}")

    return avg_loss


def test(model: NeuralNetwork, test_loader: DataLoader) -> float:
    logger = logging.getLogger("ovg-test")
    criterion = nn.MSELoss()
    avg_loss = 0.0
    for inputs, targets in test_loader:
        with torch.no_grad():
            y_pred = model(inputs)
            test_loss = criterion(y_pred, targets)
            avg_loss += test_loss.item()
    avg_loss /= len(test_loader)
    logger.debug(f"Test Loss: {avg_loss:.4f}")

    return avg_loss


def build_train_test_loaders(
    X: ArrayLike,
    Y: ArrayLike,
    batch_size: int = 64,
    y_scaler: Optional[StandardScaler] = None,
) -> Tuple[DataLoader, DataLoader]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    if y_scaler:
        y_train = y_scaler.fit_transform(y_train).flatten()
        y_test = y_scaler.transform(y_test).flatten()

    train_dataset = Dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


