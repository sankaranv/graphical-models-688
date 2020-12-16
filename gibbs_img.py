import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b


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


def gibbs_sample_iteration(n_iter, X, weight, bias, grid_size):
    # prev_sample = np.random.choice([-1, 1], size = grid_size.shape)
    neighbours = get_neighbours(grid_size)
    prev_sample = np.ones(grid_size).ravel()
    w = weight * np.ones(grid_size).ravel()
    b = bias * X.ravel()
    samples = []
    for iter in tqdm(range(n_iter)):
        sample = np.copy(prev_sample)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                idx = i * grid_size[1] + j
                nb = neighbours[idx]
                sample_prob = sigmoid(
                    2 * (b[idx] + np.sum(w[nb] * sample[nb])))
                p = np.random.uniform(0, 1)
                if p < sample_prob:
                    sample[idx] = 1
                else:
                    sample[idx] = -1
        samples.append(sample)
        prev_sample = np.copy(sample)
    return np.array(samples)


noise_img = np.array(Image.open('data/hw4/im_noisy.png'))
clean_img = np.array(Image.open('data/hw4/im_clean.png'))
noise_img = np.interp(noise_img, (0, 255), (-1, +1))
clean_img = np.interp(clean_img, (0, 255), (-1, +1))

weights = [0.58]
biases = [-0.72]
min_error = np.inf
for w in weights:
    for b in biases:
        samples = gibbs_sample_iteration(
            n_iter=100, X=noise_img, weight=w, bias=b, grid_size=noise_img.shape)
        mean_img = np.mean(samples, axis=0).reshape(noise_img.shape)
        error = np.mean(np.abs(mean_img - clean_img))
        if error < min_error:
            min_error = error
        print('w: {} b: {} | MAE: {} | Min:{}'.format(w, b, error, min_error))
        plt.imshow(mean_img, cmap='gray')
        plt.show()
