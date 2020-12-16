import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def objective(theta):
    x = theta[0]
    y = theta[1]
    return (1 - x)**2 + (100 * (y - x**2)**2)


def objective_grad(theta):
    x = theta[0]
    y = theta[1]
    dx = 2 * x + 400 * (x**3) - 400 * x * y - 2
    dy = 200 * y - 200 * (x**2)
    return np.array([dx, dy])


def fit():
    argmaxs = []
    fs = []
    for i in range(10000):
        theta = np.random.randn(2)
        argmax, f, d = fmin_l_bfgs_b(
            objective, x0=theta, fprime=objective_grad)
        argmaxs.append(argmax)
        fs.append(f)
    fmax = -np.min(fs)
    argmax = argmaxs[np.argmin(fs)]
    print("argmax = {}".format(argmax))
    print("f = {}".format(-f))


if __name__ == "__main__":
    fit()
