import numpy as np
from itertools import product
from time import time
from scipy.special import logsumexp


class CRF:

    def __init__(self):
        self.load_model()

    def str2idx(self, str):
        idx = list(str)
        idx = [0 if x == 'e' else x for x in idx]
        idx = [1 if x == 't' else x for x in idx]
        idx = [2 if x == 'a' else x for x in idx]
        idx = [3 if x == 'i' else x for x in idx]
        idx = [4 if x == 'n' else x for x in idx]
        idx = [5 if x == 'o' else x for x in idx]
        idx = [6 if x == 's' else x for x in idx]
        idx = [7 if x == 'h' else x for x in idx]
        idx = [8 if x == 'r' else x for x in idx]
        idx = [9 if x == 'd' else x for x in idx]
        return idx

    def idx2str(self, idx):
        idx = ['e' if x == 0 else x for x in idx]
        idx = ['t' if x == 1 else x for x in idx]
        idx = ['a' if x == 2 else x for x in idx]
        idx = ['i' if x == 3 else x for x in idx]
        idx = ['n' if x == 4 else x for x in idx]
        idx = ['o' if x == 5 else x for x in idx]
        idx = ['s' if x == 6 else x for x in idx]
        idx = ['h' if x == 7 else x for x in idx]
        idx = ['r' if x == 8 else x for x in idx]
        idx = ['d' if x == 9 else x for x in idx]
        str = ''.join(idx)
        return str

    def load_model(self):
        with open('model/feature-params.txt') as f:
            feature_params = [x.split(' ')
                              for x in f.read().split('\n') if x.strip()]
            self.feature_params = np.array(feature_params).astype(float)

        with open('model/transition-params.txt') as f:
            transition_params = [x.split(' ')
                                 for x in f.read().split('\n') if x.strip()]
            self.transition_params = np.array(transition_params).astype(float)

    def energy(self, x, y):

        self.node_potential(x)
        potential = np.zeros(len(y))
        for i in range(len(y)):
            potential[i] = self.node_potentials[i, y[i]]
        energy = np.sum(potential)
        for i in range(len(y) - 1):
            energy += self.transition_params[y[i], y[i + 1]]
        return energy

    def node_potential(self, x):
        length = x.shape[0]
        self.node_potentials = np.zeros((length, 10))
        for i in range(10):
            self.node_potentials[:, i] = np.sum(
                self.feature_params[i] * x, axis=1)

    def logZ_exhaustive(self, x):
        length = x.shape[0]
        maxRange = np.full(length, 9)
        states = [i for i in product(*(range(i + 1) for i in maxRange))]
        Z = logsumexp([self.energy(x, y) for y in states])
        return Z

    def predict_exhaustive(self, x):
        length = x.shape[0]
        maxRange = np.full(length, 9)
        states = [i for i in product(*(range(i + 1) for i in maxRange))]
        probs = np.array([self.energy(x, y) for y in states])
        probs -= self.logZ_exhaustive(x)
        max_prob = np.max(probs)
        max_labeling = states[np.argmax(probs)]
        return np.exp(max_prob), crf.idx2str(max_labeling)

    def list_insert(self, a, i, j):
        b = list(a[:])
        b.insert(i, j)
        return b

    def marginals_exhaustive(self, x):
        length = x.shape[0]
        maxRange = np.full(length - 1, 9)
        probs = np.zeros((length, 10))
        for i in range(length):
            Z = 0
            for j in range(10):
                states = [self.list_insert(k, i, j) for k in product(
                    *(range(i + 1) for i in maxRange))]
                Z = logsumexp([self.energy(x, y) for y in states] + [Z])
                probs[i, j] = logsumexp([self.energy(x, y) for y in states])
            probs[i, :] -= Z
        return np.exp(probs)

    def marginals_pairwise_exhaustive(self, x):
        marginals = {}
        length = x.shape[0]
        maxRange = np.full(length - 2, 9)
        for idx in range(length - 1):
            probs = np.zeros((10, 10))
            for i in range(10):
                Z = 0
                for j in range(10):
                    states = [self.list_insert(self.list_insert(k, idx, i), idx + 1, j) for k in product(
                        *(range(l + 1) for l in maxRange))]
                    probs[i, j] = logsumexp(
                        [self.energy(x, y) for y in states])
            probs -= logsumexp(probs)
            marginals[idx + 1, idx + 2] = np.exp(probs)
        return marginals

    def message_passing(self, x):

        length = x.shape[0]
        self.messages = {}
        self.node_potential(x)

        # Backward messages
        self.messages[length + 1, length] = np.zeros(10)
        for i in range(length, 1, -1):
            self.messages[i, i - 1] = np.zeros(10)
            for xj in range(10):
                self.messages[i, i - 1][xj] = logsumexp(
                    self.node_potentials[i - 1] + self.transition_params[:, xj] + self.messages[i + 1, i])
        del self.messages[length + 1, length]

        # Forward messages
        self.messages[0, 1] = np.zeros(10)
        for i in range(1, length):
            self.messages[i, i + 1] = np.zeros(10)
            for xj in range(10):
                self.messages[i, i + 1][xj] = logsumexp(
                    self.node_potentials[i - 1] + self.transition_params[:, xj] + self.messages[i - 1, i])
        del self.messages[0, 1]

    def logZ_sumproduct(self, x):
        self.message_passing(x)
        return logsumexp(self.node_potentials[0] + self.messages[2, 1])

    def marginals_sumproduct(self, x):

        length = x.shape[0]
        self.message_passing(x)

        marginals = {}

        # Node marginals

        for i in range(length):
            node_marginal_i = self.node_potentials[i]
            if i >= 1:
                node_marginal_i += self.messages[i, i + 1]
            if i + 2 <= length:
                node_marginal_i += self.messages[i + 2, i + 1]
            node_marginal_i -= logsumexp(node_marginal_i)
            marginals[i + 1] = np.exp(node_marginal_i)

        # Pairwise marginals

        self.message_passing(x)

        for i in range(length - 1):
            j = i + 1
            pair_marginal_ij = self.node_potentials[i].reshape(
                -1, 1) + self.node_potentials[j].reshape(1, -1) + self.transition_params
            if i >= 1:
                pair_marginal_ij += self.messages[i, i + 1].reshape(-1, 1)
            if j + 2 <= length:
                pair_marginal_ij += self.messages[j + 2, j + 1].reshape(1, -1)
            pair_marginal_ij -= logsumexp(pair_marginal_ij)
            marginals[i + 1, j + 1] = np.exp(pair_marginal_ij)

        return marginals

    def predict_sumproduct(self, x):
        # Predict the label sequence of x
        # output: a label sequence (e.g. an array)
        length = x.shape[0]
        marginals = self.marginals_sumproduct(x)
        pred_string = []
        for i in range(length):
            pred_string.append(np.argmax(marginals[i + 1]))
        return self.idx2str(pred_string)


def error_string(y_pred, y_true):
    y_pred = np.array(list(y_pred))
    y_true = np.array(list(y_true))
    error = np.array([y_pred != y_true]).astype(int)[0]
    return np.sum(error), error.shape[0]


def likelihood_exhaustive(crf, n=50):
    # Compute likelihood using CRF marginals
    likelihood = 0
    for idx in range(1, n + 1):

        with open('data/hw2/train_words.txt') as f:
            train_words = [x.split(' ')
                           for x in f.read().split('\n') if x.strip()]
        with open('data/hw2/train_img' + str(idx) + '.txt') as f:
            img = [x.split(' ') for x in f.read().split('\n') if x.strip()]
            img = np.array(img).astype(float)
            label = crf.str2idx(train_words[idx - 1][0])
            length = len(label)

        crf.node_potential(img)
        for j in range(length):
            likelihood += crf.node_potentials[j, label[j]]
        for j in range(length - 1):
            likelihood += crf.transition_params[label[j], label[j + 1]]
        likelihood -= crf.logZ_sumproduct(img)

    return likelihood / n


def likelihood_sumproduct(crf, n=50):
    # Compute likelihood using CRF marginals
    likelihood = 0
    for idx in range(1, n + 1):

        with open('data/hw2/train_words.txt') as f:
            train_words = [x.split(' ')
                           for x in f.read().split('\n') if x.strip()]
        with open('data/hw2/train_img' + str(idx) + '.txt') as f:
            img = [x.split(' ') for x in f.read().split('\n') if x.strip()]
            img = np.array(img).astype(float)
            label = crf.str2idx(train_words[idx - 1][0])
            length = len(label)

        marginals = crf.marginals_sumproduct(img)
        for j in range(1, length):
            likelihood += np.log(marginals[j, j + 1]
                                 [label[j - 1], label[j]])
        for j in range(2, length):
            likelihood -= np.log(marginals[j][label[j - 1]])

    return likelihood / n


def gradients(crf, n=50):

    dWf = np.zeros((10, 321))
    dWt = np.zeros((10, 10))

    for idx in range(1, n + 1):

        with open('data/hw2/train_words.txt') as f:
            train_words = [x.split(' ')
                           for x in f.read().split('\n') if x.strip()]
        with open('data/hw2/train_img' + str(idx) + '.txt') as f:
            img = [x.split(' ') for x in f.read().split('\n') if x.strip()]
            img = np.array(img).astype(float)
            label = crf.str2idx(train_words[idx - 1][0])
            length = len(label)

        marginals = crf.marginals_sumproduct(img)

        for c in range(10):
            for f in range(321):
                for j in range(length):
                    dWf[c, f] += img[j, f] * \
                        (int(label[j] == c) - marginals[j + 1][c])

        for c in range(10):
            for d in range(10):
                for j in range(length - 1):
                    dWt[c, d] += int(label[j] == c and label[j + 1]
                                     == d) - marginals[j + 1, j + 2][c, d]

    return dWf / n, dWt / n


if __name__ == "__main__":

    crf = CRF()

    with open('data/hw2/test_words.txt') as f:
        test_words = [x.split(' ') for x in f.read().split('\n') if x.strip()]

    # Q1.1
    print('\n Q1.1 \n')
    with open('data/hw2/test_img' + str(1) + '.txt') as f:
        img = [x.split(' ') for x in f.read().split('\n') if x.strip()]
        img = np.array(img).astype(float)
        crf.node_potential(img)
        print(crf.node_potentials)
        np.savetxt("tables/q1-1.csv", crf.node_potentials, '%g')

    # Q1.2
    print('\n Q1.2 \n')
    for img_idx in [1, 2, 3]:
        with open('data/hw2/test_img' + str(img_idx) + '.txt') as f:
            img = [x.split(' ') for x in f.read().split('\n') if x.strip()]
            img = np.array(img).astype(float)
            label = crf.str2idx(test_words[img_idx - 1][0])
            print(crf.energy(img, label))

    # Q1.3
    print('\n Q1.3 \n')
    for img_idx in [1, 2, 3]:
        with open('data/hw2/test_img' + str(img_idx) + '.txt') as f:
            img = [x.split(' ') for x in f.read().split('\n') if x.strip()]
            img = np.array(img).astype(float)
            print(crf.logZ_exhaustive(img))

    # Q1.4
    print('\n Q1.4 \n')
    for img_idx in [1, 2, 3]:
        with open('data/hw2/test_img' + str(img_idx) + '.txt') as f:
            img = [x.split(' ') for x in f.read().split('\n') if x.strip()]
            img = np.array(img).astype(float)
            print(crf.predict_exhaustive(img))

    # Q1.5
    print('\n Q1.5 \n')
    with open('data/hw2/test_img' + str(1) + '.txt') as f:
        img = [x.split(' ') for x in f.read().split('\n') if x.strip()]
        img = np.array(img).astype(float)
        marginals = crf.marginals_exhaustive(img)
        print(marginals)
        np.savetxt("tables/q1-5.csv", marginals, '%g')

        # Q2.1
        print('\n Q2.1 \n')
        crf.message_passing(img)
        msg = np.vstack((
            crf.messages[1, 2], crf.messages[2, 1], crf.messages[2, 3], crf.messages[3, 2]))
        print(msg)
        np.savetxt("tables/q2-1.csv", msg, '%g')

        # Q2.2
        print('\n Q2.2 \n')
        marginals = crf.marginals_sumproduct(img)
        print(marginals)
        node_marginals = np.vstack((
            marginals[1], marginals[2], marginals[3], marginals[4]))
        np.savetxt("tables/q2-2.csv", node_marginals, '%g')
        ax = [1, 2, 7]
        np.savetxt("tables/q2-2-p12.csv",
                   marginals[1, 2][ax][:, ax], '%g')
        np.savetxt("tables/q2-2-p23.csv",
                   marginals[2, 3][ax][:, ax], '%g')
        np.savetxt("tables/q2-2-p34.csv",
                   marginals[3, 4][ax][:, ax], '%g')

    # Q2.3
    print('\n Q2.3 \n')
    error_count = 0
    len_count = 0

    for img_idx in range(1, 201):
        with open('data/hw2/test_img' + str(img_idx) + '.txt') as f:
            img = [x.split(' ') for x in f.read().split('\n') if x.strip()]
            img = np.array(img).astype(float)
            y_pred = crf.predict_sumproduct(img)
            y_true = test_words[img_idx - 1][0]
            e, n = error_string(y_pred, y_true)
            print(y_pred, y_true, e)
            error_count += e
            len_count += n
    error = error_count / len_count
    print("Accuracy = {}".format(1 - error))

    # Q3.5
    print('\n Q3.5 \n')
    print("Likelihood = {}".format(likelihood_exhaustive(crf)))
    print("Likelihood = {}".format(likelihood_sumproduct(crf)))

    # Q3.6
    print('\n Q3.6 \n')
    dWf, dWt = gradients(crf)

    with open('model/feature-gradient.txt') as f:
        feature_grad = [x.split(' ')
                        for x in f.read().split('\n') if x.strip()]
        feature_grad = np.array(feature_grad).astype(float)

    with open('model/transition-gradient.txt') as f:
        transition_grad = [x.split(' ')
                           for x in f.read().split('\n') if x.strip()]
        transition_grad = np.array(transition_grad).astype(float)

    assert(np.allclose(dWf, feature_grad))
    assert(np.allclose(dWt, transition_grad))
