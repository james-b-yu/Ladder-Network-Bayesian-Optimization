import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def f(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


# Gradient descent
x_gd = 10 * torch.ones(2, 1)
x_gd.requires_grad = True

optimizer = optim.SGD([x_gd], lr=1e-5)
h_gd = []
for i in range(100):
    optimizer.zero_grad()
    objective = f(x_gd)
    objective.backward()
    optimizer.step()
    h_gd.append(objective.item())


# L-BFGS
x_lbfgs = 10 * torch.ones(2, 1)
x_lbfgs.requires_grad = True

optimizer = optim.LBFGS([x_lbfgs],
                        history_size=10,
                        max_iter=4,
                        line_search_fn="strong_wolfe")
h_lbfgs = []
for i in range(100):
    optimizer.zero_grad()
    objective = f(x_lbfgs)
    objective.backward()
    optimizer.step(lambda: f(x_lbfgs))
    h_lbfgs.append(objective.item())


# Plotting
plt.semilogy(h_gd, label='GD')
plt.semilogy(h_lbfgs, label='L-BFGS')
plt.legend()
plt.show()
