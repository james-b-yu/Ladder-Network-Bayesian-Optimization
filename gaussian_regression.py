import pickle
from typing import cast
import matplotlib.pyplot as plt
import torch as t


def _min_fn(K1: t.Tensor, S: t.Tensor, y: t.Tensor, m: t.Tensor, device=t.device("cpu")) -> t.Tensor:
    det_val = (K1 + S).det() if (K1 + S).det().item() > 1e-3 else t.tensor([1e-3], dtype=t.float64, device=device)
    return 0.5 * t.log(det_val) + 0.5 * (y - m).t() @ (K1 + S).inverse() @ (y - m)


def find_priors(X: t.Tensor, y: t.Tensor, epochs=100, device=t.device("cpu")):
    # uses backpropagation to find the prior values by maximising the log likelihood wrt. prior kernel length factors l and v, and noise factor s
    # sets prior mean to average of ys
    # modified: v

    m = y.mean()  # always the mean. no better way of getting it

    s = t.tensor([1], requires_grad=True, dtype=t.float64, device=device)
    v = t.tensor([1], requires_grad=True, dtype=t.float64, device=device)
    l = t.tensor([1], requires_grad=True, dtype=t.float64, device=device)

    optim = t.optim.LBFGS([s, v, l], line_search_fn="strong_wolfe")

    def closure():
        optim.zero_grad()

        K1 = _kernel_t(X, X, v, l)
        S = s ** 2 * t.eye(X.shape[0], dtype=t.float64, device=device)

        loss = _min_fn(K1, S, y, m) + 0.01 * sum([hp ** -2 + s ** 2 for hp in [s, v, l]])
        loss.backward()
        return loss

    for i in range(epochs):
        optim.step(closure)  # type: ignore

    return m.detach(), s.detach(), v.detach(), l.detach()


def _squared_norm_t(A: t.Tensor, B: t.Tensor):
    # A: n * d
    # B: m * d
    # out shape: n * m
    # returns a matirix M where Mij = norm(Ai - Bj) ** 2
    return t.norm(A, dim=1).reshape(-1, 1) ** 2 + t.linalg.norm(B, axis=1).reshape(1, -1) ** 2 - 2 * A @ B.t()


def _kernel_t(a: t.Tensor, b: t.Tensor, v: t.Tensor, l: t.Tensor):
    return (v ** 2) * t.exp(-0.5 * _squared_norm_t(a, b) / (l ** 2))


class model:
    def __init__(self, X: t.Tensor, y: t.Tensor, m=t.tensor([0.0], dtype=t.float64), s=t.tensor([0.1], dtype=t.float64), v=t.tensor([5.0], dtype=t.float64), l=t.tensor([1.0], dtype=t.float64), device=t.device("cpu")):
        self.device = device
        self.X = X.to(device)
        self.y = y.to(device)
        self.s = s.to(device)
        self.m = m.to(device)
        self.v = v.to(device)
        self.l = l.to(device)
        self.y = y.to(device)

        self.K1 = _kernel_t(X, X, self.v, self.l)
        self.S1 = (self.s ** 2 + 1e-3) * t.eye(self.K1.shape[0], device=device)

        self.K1_S1_inv = (self.K1 + self.S1).inverse()
        self.y_minus_m = self.y - self.m
        self.K1_S1_inv_y_minus_m = self.K1_S1_inv @ self.y_minus_m

    def __call__(self, Xt_in: t.Tensor):
        Xt = Xt_in if Xt_in.dim() > 1 else Xt_in.unsqueeze(0)
        K2 = _kernel_t(self.X, Xt, self.v, self.l)
        K3 = K2.t()
        K4 = _kernel_t(Xt, Xt, self.v, self.l)

        posterior_mean = self.m + K3 @ self.K1_S1_inv_y_minus_m
        posterior_covariance = K4 - K3 @ self.K1_S1_inv @ K2

        return posterior_mean, posterior_covariance

    def dump(self, file):
        pickle.dump(self, file)

    @classmethod
    def load(cls, file):
        return cast(cls, pickle.load(file))
