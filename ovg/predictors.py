import numpy as np
import torch
from estimators import estimate_cond_mean, estimate_cond_skew


class Predictor:
    def __init__(self):
        self.mc_samples = 1000
        self.data_source = self.data_target = None


class MarginalPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.model_reg_source = None

    def fit(self, data_source, data_target):
        self.data_source = data_source
        self.data_target = data_target
        self.model_reg_source = estimate_cond_mean(
            X=data_source.loc[:, ["X_0", "X_1"]].values,
            Y=data_source.loc[:, ["Y"]].values,
        )

    def func(self, X):
        model_output = self.model_reg_source(torch.from_numpy(X).float())
        return model_output.detach().numpy()

    def __call__(self, data):
        return marginalize(
            self.func,
            x_marg=self.data_source.loc[:, ["X_0"]].values,
            x_pred=data.loc[:, ["X_1"]].values,
            num_samples=self.mc_samples
        )


class ImputedPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.model_reg_source = None
        self.mean_X0 = None

    def fit(self, data_source, data_target):
        self.data_source = data_source
        self.data_target = data_target

        self.mean_X0 = np.mean(data_source['X_0'])
        imputed_mean = np.mean(data_source['X_2'])
        input_source = data_source.loc[:, ["X_0", "X_1"]].values
        imputed_mean_row = np.full((input_source.shape[0], 1), fill_value=imputed_mean)
        input_data = np.hstack([input_source, imputed_mean_row])
        self.model_reg_source = estimate_cond_mean(
            X=input_data,
            Y=data_source.loc[:, ["Y"]].values,
        )

    def __call__(self, data):
        data_X_1_X_2 = data.loc[:, ["X_1", "X_2"]].values
        mean_X0_row = np.full((data_X_1_X_2.shape[0], 1), fill_value=self.mean_X0)
        X = np.hstack([mean_X0_row, data_X_1_X_2])
        print(X.shape)
        model_output = self.model_reg_source(torch.from_numpy(X).float())
        return model_output.detach().numpy().squeeze(axis=1)


class OptimalPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.model_reg_target = None

    def fit(self, data_source, data_target):
        self.data_source = data_source
        self.data_target = data_target

        self.model_reg_target = estimate_cond_mean(
            X=data_target.loc[:, ["X_1", "X_2"]].values,
            Y=data_target.loc[:, ["Y"]].values,
        )

    def __call__(self, data):
        X = data.loc[:, ["X_1", "X_2"]].values
        model_output = self.model_reg_target(torch.from_numpy(X).float())
        return model_output.detach().numpy().squeeze(axis=1)


class ProposedPredictor(Predictor):
    def __init__(self, lr=0.01, hidden_size=64, num_epochs = 50):
        super().__init__()
        self.model_reg_source = self.model_derivative = self.target_mean = None
        self.y_scaler = None
        self.lr = lr
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

    def fit(self, data_source, data_target):
        self.data_source = data_source
        self.data_target = data_target

        self.model_reg_source = estimate_cond_mean(
            X=data_source.loc[:, ["X_0", "X_1"]].values,
            Y=data_source.loc[:, ["Y"]].values,
        )
        self.model_derivative, self.deriv_scaler = estimate_cond_skew(
            self.model_reg_source,
            X_source=data_source.loc[:, ["X_0", "X_1"]].values,
            Y_source=data_source.loc[:, ["Y"]].values,
            X_target=data_target.loc[:, ["X_2"]].values,
            lr=self.lr,
            hidden_size=self.hidden_size,
            num_epochs=self.num_epochs
        )
        self.target_mean = np.mean(data_target["X_2"])
        self.target_std = np.std(data_target['X_2'])

    def func(self, X):
        X = torch.from_numpy(X).float()
        cond_mean = self.model_reg_source(X[:, :2]).detach().numpy()
        output_deriv = self.deriv_scaler.inverse_transform(self.model_derivative(X[:, :2]).detach().numpy())
        derivative = np.sign(output_deriv) * np.power(np.abs(output_deriv), 1 / 3)
        return cond_mean + derivative * (X[:, 2].unsqueeze(1).detach().numpy() - self.target_mean)

    def __call__(self, data):
        X = data.loc[:, ["X_1", "X_2"]].values
        return marginalize(
            self.func, x_marg=self.data_source.loc[:, ["X_0"]].values, x_pred=X, num_samples=self.mc_samples
        )


def marginalize(func, x_marg, x_pred, num_samples=None):
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


