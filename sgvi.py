import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

# Load data

X_train = np.genfromtxt('data/hw4/X_train.csv', delimiter=' ')
X_test = np.genfromtxt('data/hw4/X_test.csv', delimiter=' ')
y_train = np.genfromtxt('data/hw4/Y_train.csv', delimiter=' ')
y_test = np.genfromtxt('data/hw4/Y_test.csv', delimiter=' ')


def sample_prior(w):
    sample = np.random.normal(np.zeros(w.shape), 1)
    return sample


def sample_q(w):
    sample = np.random.normal(w, np.sqrt(0.5))
    return sample


def sample_reparam(w):
    sample = np.random.normal(np.zeros(w.shape), 1)
    return sample


def prior_logprob(z):
    return -np.log(np.sqrt(2 * np.pi)) - 0.5 * np.dot(z, z)


def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def elbo(w, X, y):
    z = sample_q(w)
    log_p_evidence = -np.logaddexp(np.zeros(X.shape[0]), -y * np.dot(X, z))
    log_p_prior = prior_logprob(z)
    objective = np.sum(log_p_evidence) + log_p_prior
    return -objective


def elbo_gradient(w, X, y):
    e = sample_reparam(w)
    t = w + np.sqrt(0.5) * e
    grad_log_p_evidence = y.reshape(-1, 1) * X * \
        sigmoid(-y * np.dot(X, t)).reshape(-1, 1)
    grad_log_p_prior = -t
    grad_objective = np.sum(grad_log_p_evidence, axis=0) + grad_log_p_prior
    return -grad_objective


def fit_bfgs(X, y):
    w_init = np.random.randn(5)
    argmin, f, d = fmin_l_bfgs_b(
        elbo, x0=w_init, fprime=elbo_gradient, args=(X, y), disp=1)
    return argmin


def fit(X, y, max_iter=10000, alpha=0.005, print_freq=100):
    w = np.random.randn(5)
    track_weights = np.copy(w).reshape(1, -1)
    for i in range(max_iter):
        objective = elbo(w, X, y)
        grad = elbo_gradient(w, X, y)
        w = w - alpha * grad
        if print_freq is not None and (i + 1) % print_freq == 0:
            print('Step {}: ELBO = {}'.format(i + 1, objective))
        track_weights = np.concatenate(
            (track_weights, w.reshape(1, -1)), axis=0)
    return w, track_weights


def zero_one_acc(y, y_pred):
    assert(y.size == y_pred.size)
    N = y_pred.size
    counts = np.sum(np.array([y == y_pred]).astype(float))
    acc = counts / N
    return acc


# iter = 10000
# w, t = fit(X_train, y_train, max_iter=iter, print_freq=100)
# print(w)
# xaxis = np.arange(1, iter + 1, 1)
# plt.figure(figsize=(11, 5))
# for i in range(5):
#     plt.plot(xaxis, t[1:, i], label='dim ' + str(i + 1))
# plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
# plt.xlabel('No. of iterations')
# plt.ylabel('Weight')
# plt.savefig('q4.png')

for iter in [10, 100, 1000, 10000]:
    print('Iterations: {}'.format(iter))
    for run in [1, 2, 3, 4, 5]:
        w, t = fit(X_train, y_train, max_iter=iter, print_freq=None)
        z_test = np.zeros((1000, 5))
        y_pred = []
        for i in range(1000):
            z_test[i] = sample_q(w)
        prob = 0
        for i in range(1000):
            for j in range(1000):
                prob += sigmoid(np.dot(X_test[i], z_test[j]))
            prob /= 1000
            if prob > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(-1)
        y_pred = np.array(y_pred)
        print("Run {} error: {}".format(
            run, 1 - zero_one_acc(y_pred, y_test)))
