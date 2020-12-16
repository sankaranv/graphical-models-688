import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def get_neighbours(grid_size):
    neighbours = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            nb = []
            if i - 1 >= 0:
                nb.append((i - 1) * grid_size[1] + j)
            if i + 1 <= grid_size[0] - 1:
                nb.append((i + 1) * grid_size[1] + j)
            if j - 1 >= 0:
                nb.append(i * grid_size[1] + j - 1)
            if j + 1 <= grid_size[1] - 1:
                nb.append(i * grid_size[1] + j + 1)
            neighbours.append(np.array(nb))
    return neighbours


def gibbs_sample_iteration(n_iter, w, b, grid_size):
    #prev_sample = np.random.choice([-1, 1], size=(grid_size * grid_size,))
    neighbours = get_neighbours((grid_size, grid_size))
    prev_sample = np.ones((grid_size * grid_size,))
    samples = []
    for iter in range(n_iter):
        sample = np.copy(prev_sample)
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                nb = neighbours[idx]
                sample_prob = sigmoid(2 * (b + np.sum(w * sample[nb])))
                p = np.random.uniform(0, 1)
                if p < sample_prob:
                    sample[idx] = 1
                else:
                    sample[idx] = -1
        samples.append(sample)
        prev_sample = np.copy(sample)
    return np.array(samples)


n_runs = 100
weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

avgs = []
for w in weights:
    y_avg = np.zeros(100)
    for i in tqdm(range(n_runs)):
        samples = gibbs_sample_iteration(n_iter=100, w=w, b=0, grid_size=30)
        y_avg += np.mean(samples, axis=1)
    y_avg /= n_runs
    avgs.append(y_avg)
avgs = np.array(avgs)
np.save('avgs.npy', avgs)
avgs = np.load('avgs.npy')
for y_avg, w in zip(avgs, weights):
    plt.plot(np.arange(1, 101), y_avg, label='w = {}'.format(w))
plt.ylabel('Mean value of y')
plt.xlabel('Number of iterations')
plt.legend()
plt.show()
