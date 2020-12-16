import numpy as np
import matplotlib.pyplot as plt


def zero_one_acc(y, y_pred):
    assert(y.size == y_pred.size)
    N = y_pred.size
    counts = np.sum(np.array([y == y_pred]).astype(float))
    acc = counts / N
    return acc


# Import and prepare dataset
filename = 'data/hw1/data-train-1.txt'
with open(filename) as f:
    dataset = [x.split(',') for x in f.read().split('\n') if x.strip()]
    for i in range(len(dataset)):
        dataset[i] = [int(x) for x in dataset[i]]
dataset = np.array(dataset)

# Define Bayesian Network structure
var_names = np.array(['A', 'G', 'CP', 'BP', 'CH', 'ECG', 'HR', 'EIA', 'HD'])
dimensions = np.array([3, 2, 4, 2, 2, 2, 2, 2, 2])
parents = np.array([None, None, None, [1], [0, 1],
                    None, [0], None, [0, 1, 2, 5, 7]])


def max_likelihood_estimate(var_names, dataset, parents, verbose=True):

    # Prepare parameter dict
    param_dict = {}

    for idx, var in enumerate(var_names):
        if parents[idx] == None:
            param = np.zeros(dimensions[idx])
            param_dict[var] = param
        else:
            param_name = var + '|' + ','.join(var_names[parents[idx]])
            param_dimension = np.append(
                dimensions[idx], dimensions[parents[idx]])
            param = np.zeros(param_dimension)
            param_dict[var] = param

    # Learn params
    for idx, (key, param) in enumerate(param_dict.items()):

        if(verbose):
            print('Learning params for P(' + key + ')')
            print('Shape: ' + str(param_dict[key].shape) + '\n')
        # Take frequency counts for variables with no parents
        if parents[idx] == None:
            counts = np.bincount(dataset[:, idx])[1:]
            counts = counts / np.sum(counts)
            param_dict[key] = counts

        # For variables with parents, take combined counts
        else:
            dims_slice = np.append(idx, parents[idx])

            # Get counts of var-parent pairs
            rows, counts = np.unique(
                dataset[:, dims_slice], axis=0, return_counts=True)
            for row, count in zip(rows, counts):
                param_dict[key][tuple(row - 1)] = count

            # Divide by parent counts
            parents_sum = param_dict[key].sum(axis=0)
            param_dict[key] = param_dict[key] / parents_sum

        if(verbose):
            print(param_dict[key])
            print('----------------------------------------')

    if(verbose):
        print('Parameters Learned')
        print('========================================')
    return param_dict


param_dict = max_likelihood_estimate(
    var_names, dataset, parents, verbose=False)

# Train 5 models

models = []
models.append(param_dict)

for i in [2, 3, 4, 5]:
    filename = 'data/hw1/data-train-' + str(i) + '.txt'
    with open(filename) as f:
        train_dataset = [x.split(',')
                         for x in f.read().split('\n') if x.strip()]
        for j in range(len(train_dataset)):
            train_dataset[j] = [int(x) for x in train_dataset[j]]
    train_dataset = np.array(train_dataset)
    param_dict_i = max_likelihood_estimate(
        var_names, train_dataset, parents, verbose=False)
    models.append(param_dict_i)


def calc_heart_prob(model, datacase):
    a = datacase[0]
    g = datacase[1]
    cp = datacase[2]
    bp = datacase[3]
    ch = datacase[4]
    ecg = datacase[5]
    hr = datacase[6]
    eia = datacase[7]
    joint_hd = param_dict['HD'][:, a, g, cp, ecg, eia]
    hd_prob = joint_hd / np.sum(joint_hd)

    return hd_prob

# Test models


accuracies = []

for i, model in enumerate(models):
    filename = 'data/hw1/data-test-' + str(i + 1) + '.txt'
    with open(filename) as f:
        test_dataset = [x.split(',')
                        for x in f.read().split('\n') if x.strip()]
        for j in range(len(test_dataset)):
            test_dataset[j] = [int(x) for x in test_dataset[j]]
    test_dataset = np.array(test_dataset)

    y_test = test_dataset[:, -1] - 1
    x_test = test_dataset[:, :-1] - 1

    # Make predictions
    y_pred = []
    for n in range(x_test.shape[0]):
        datacase = x_test[n]
        pred = calc_heart_prob(model, datacase)
        y_pred.append(np.argmax(pred))
    y_pred = np.array(y_pred)
    accuracies.append(zero_one_acc(y_pred, y_test))
print("Accuracies: " + str(accuracies))
print("Max: " + str(np.max(accuracies)))
print("Mean: " + str(np.mean(accuracies)))
print("Std: " + str(np.std(accuracies)))
