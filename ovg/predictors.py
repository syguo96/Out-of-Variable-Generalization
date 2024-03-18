import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Any, Callable, Optional

from .estimators import estimate_cond_mean, estimate_cond_skew


class Predictor:
    def __init__(self) -> None:
        self.mc_samples: int = 1000
        self.data_source: Optional[pd.DataFrame] = None
        self.data_target: Optional[pd.DataFrame] = None

    def fit(self, data_source: pd.DataFrame, data_target: pd.DataFrame) -> None:
        raise NotImplementedError()


def _marginalize(
    func: Callable[[np.ndarray], np.ndarray],
    x_marg: np.ndarray,
    x_pred: np.ndarray,
    num_samples: Optional[int] = None,
) -> np.ndarray:
    if not num_samples:
        num_samples = x_marg.shape[0]
    x_marg = np.random.choice(x_marg[:, 0], num_samples)
    x_marg = x_marg.reshape(-1, 1)

    result = []
    for i in range(x_pred.shape[0]):
        x = x_pred[i, ...] * np.ones((num_samples, 1))
        func_input = np.hstack([x_marg, x])
        result.append(np.mean(func(func_input)))
    return np.array(result)


class MarginalPredictor(Predictor):
    def __init__(self) -> None:
        super().__init__()
        self.model_reg_source: Optional[nn.Module] = None

    def fit(self, data_source: pd.DataFrame, data_target: pd.DataFrame) -> None:
        self.data_source = data_source
        self.data_target = data_target
        self.model_reg_source = estimate_cond_mean(
            X=data_source.loc[:, ["X_0", "X_1"]].values,
            Y=data_source.loc[:, ["Y"]].values,
        )

    def func(self, X: np.ndarray) -> np.ndarray:
        if self.model_reg_source is None:
            raise RuntimeError("MarginalPredictor: method 'fit' should be called first")
        model_output = self.model_reg_source(torch.from_numpy(X).float())
        return model_output.detach().numpy()

    def __call__(self, data: pd.DataFrame) -> np.ndarray:
        if self.data_source is None:
            raise RuntimeError("MarginalPredictor: method 'fit' should be called first")

        return _marginalize(
            self.func,
            x_marg=self.data_source.loc[:, ["X_0"]].values,
            x_pred=data.loc[:, ["X_1"]].values,
            num_samples=self.mc_samples,
        )


class ImputedPredictor(Predictor):
    def __init__(self) -> None:
        super().__init__()
        self.model_reg_source: Optional[nn.Module] = None
        self.mean_X0: Optional[float] = None

    def fit(self, data_source: pd.DataFrame, data_target: pd.DataFrame) -> None:
        self.data_source = data_source
        self.data_target = data_target

        self.mean_X0 = np.mean(data_source["X_0"])
        imputed_mean = np.mean(data_source["X_2"])
        input_source = data_source.loc[:, ["X_0", "X_1"]].values
        imputed_mean_row = np.full((input_source.shape[0], 1), fill_value=imputed_mean)
        input_data = np.hstack([input_source, imputed_mean_row])
        self.model_reg_source = estimate_cond_mean(
            X=input_data,
            Y=data_source.loc[:, ["Y"]].values,
        )

    def __call__(self, data: pd.DataFrame) -> np.ndarray:
        if self.model_reg_source is None:
            raise RuntimeError("ImputedPredictor: method 'fit' should be called first")

        data_X_1_X_2 = data.loc[:, ["X_1", "X_2"]].values
        mean_X0_row = np.full((data_X_1_X_2.shape[0], 1), fill_value=self.mean_X0)
        X = np.hstack([mean_X0_row, data_X_1_X_2])
        model_output = self.model_reg_source(torch.from_numpy(X).float())
        return model_output.detach().numpy().squeeze(axis=1)


class OptimalPredictor(Predictor):
    def __init__(self) -> None:
        super().__init__()
        self.model_reg_target: Optional[nn.Module] = None

    def fit(self, data_source: pd.DataFrame, data_target: pd.DataFrame) -> None:
        self.data_source = data_source
        self.data_target = data_target

        self.model_reg_target = estimate_cond_mean(
            X=data_target.loc[:, ["X_1", "X_2"]].values,
            Y=data_target.loc[:, ["Y"]].values,
        )

    def __call__(self, data: pd.DataFrame) -> np.ndarray:
        if self.model_reg_target is None:
            raise RuntimeError("ImputedPredictor: method 'fit' should be called first")

        X = data.loc[:, ["X_1", "X_2"]].values
        model_output = self.model_reg_target(torch.from_numpy(X).float())
        return model_output.detach().numpy().squeeze(axis=1)


class ProposedPredictor(Predictor):
    def __init__(
        self, lr: float = 0.01, hidden_size: int = 64, num_epochs: int = 50
    ) -> None:
        super().__init__()
        self.model_reg_source: Optional[nn.Module] = None
        self.model_derivative: Optional[nn.Module] = None
        self.target_mean: Optional[float] = None
        self.y_scaler: Optional[Any] = None
        self.lr: float = lr
        self.hidden_size: int = hidden_size
        self.num_epochs: int = num_epochs

    def fit(self, data_source: pd.DataFrame, data_target: pd.DataFrame) -> None:
        self.data_source = data_source
        self.data_target = data_target

        self.model_reg_source = estimate_cond_mean(
            X=data_source.loc[:, ["X_0", "X_1"]].values,
            Y=data_source.loc[:, ["Y"]].values,
        )
        self.model_derivative, self.y_scaler = estimate_cond_skew(
            model_regression_source=self.model_reg_source,
            X_source=data_source.loc[:, ["X_0", "X_1"]].values,
            Y_source=data_source.loc[:, ["Y"]].values,
            X_target=data_target.loc[:, ["X_2"]].values,
            lr=self.lr,
            hidden_size=self.hidden_size,
            num_epochs=self.num_epochs,
        )
        self.target_mean = np.mean(data_target["X_2"])
        self.target_std = np.std(data_target["X_2"])

    def func(self, X: np.ndarray) -> np.ndarray:
        if (
            self.model_reg_source is None
            or self.model_derivative is None
            or self.y_scaler is None
        ):
            raise RuntimeError("ProposedPredictor: method 'fit' should be called first")

        X = torch.from_numpy(X).float()
        cond_mean = self.model_reg_source(X[:, :2]).detach().numpy()
        output_deriv = self.y_scaler.inverse_transform(
            self.model_derivative(X[:, :2]).detach().numpy()
        )
        derivative = np.sign(output_deriv) * np.power(np.abs(output_deriv), 1 / 3)
        return cond_mean + derivative * (
            X[:, 2].unsqueeze(1).detach().numpy() - self.target_mean
        )

    def __call__(self, data: pd.DataFrame) -> np.ndarray:
        if self.data_source is None:
            raise RuntimeError("ProposedPredictor: method 'fit' should be called first")

        X = data.loc[:, ["X_1", "X_2"]].values
        return _marginalize(
            self.func,
            x_marg=self.data_source.loc[:, ["X_0"]].values,
            x_pred=X,
            num_samples=self.mc_samples,
        )
