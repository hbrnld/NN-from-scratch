import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

# Global constants
PATH = '../Datasets/cifar-10-batches-py/'
BATCH = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
NODES = [50, 50, 10]
ALPHA = 0.9
SHOW_PLOTS = True
BATCH_NORM = True
COARSE = False
FINE = False
SIGMA = None    # 1e-1, 1e-3, 1e-4

# Load the data from the file, from the website
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# Standardize each column (dx1) by subtracting mean and dividing by std
def preprocess(data, mean=None, std=None):
    data = np.float64(data)
    if mean is None and std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

    data -= mean
    data /= std

    return np.array(data).T, mean, std

# Create one_hot matrix from true labels
def one_hot(labels, k):
    one_hot = np.zeros((k, len(labels)))
    for i in range(len(labels)):
        one_hot[labels[i], i] = 1
    return one_hot

# Load batches and preprocess data
def load_and_preprocess(val_size):
    train_data = None
    for batch in BATCH: 
        file = unpickle(PATH+batch)
        if train_data is not None:
            train_data = np.vstack((train_data, file['data']))          # Vertically stack data
            train_label = np.hstack((train_label, file['labels']))      # Horizontally stack labels
        else:
            train_data = file['data']
            train_label = file['labels']
            
    valid_idx = np.random.choice(range(train_data.shape[0]), val_size, replace=False)   # Select random indices for validation
    valid_data = train_data[valid_idx]                                                  # Copy selected data to validation
    valid_label = train_label[valid_idx]                                                # Copy selected labels to validation
    train_data = np.delete(train_data, valid_idx, axis=0)                               # Remove validation data
    train_label = np.delete(train_label, valid_idx)                                     # Remove validation labels

    file = unpickle(PATH+'test_batch')
    test_data = file['data']
    test_label = file['labels']

    file = unpickle(PATH+'batches.meta')
    label_names = file['label_names']
    k = len(label_names)

    # Preprocess data
    train_data, train_mean, train_std = preprocess(train_data)
    valid_data = preprocess(valid_data, train_mean, train_std)[0]
    test_data = preprocess(test_data, train_mean, train_std)[0]
    
    # Create one-hot matrices from labels (k x n)
    train_label = one_hot(train_label, k)
    valid_label = one_hot(valid_label, k)
    test_label = one_hot(test_label, k)

    return train_data, train_label, valid_data, valid_label, test_data, test_label, label_names, k

# Initializing network weights and bias
def initialize_weights(data):
    weights, bias, gamma, beta = [], [], [], []

    if SIGMA is not None: # Sensitivity to initialization
        # First layer (d -> m)
        weights.append(np.random.normal(0, SIGMA, (NODES[0], data.shape[0])))
        bias.append(np.zeros((NODES[0], 1)))

        # Hidden layers (m -> m)
        for i in range(1, len(NODES)):
            weights.append(np.random.normal(0, SIGMA, (NODES[i], NODES[i-1])))
            bias.append(np.zeros((NODES[i], 1)))
    else: # 1 for Xavier, 2 for He
        num = 2         
        # First layer (d -> m)
        weights.append(np.random.normal(0, np.sqrt(num/data.shape[0]), (NODES[0], data.shape[0])))        # m x d
        bias.append(np.zeros((NODES[0], 1)))                                                              # m x 1

        # Hidden layers (m -> m)
        for i in range(1, len(NODES)):
            weights.append(np.random.normal(0, np.sqrt(num/NODES[i-1]), (NODES[i], NODES[i-1])))
            bias.append(np.zeros((NODES[i], 1)))
                
    # Gamma and beta, for hidden layers only
    for i in range(len(NODES)-1):
        gamma.append(np.ones((NODES[i], 1)))
        beta.append(np.zeros((NODES[i], 1)))
    
    return weights, bias, gamma, beta

# Matrix multiplication of inputs and weights, adding bias
def linear_scoring(data, weight, bias):
    return np.matmul(weight, data) + bias       # Wx + b

# Definition of the rectified linear unit
def ReLU(x):
    return np.maximum(0, x)                     # max(0, x)

# Definition of the Softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Definition of the batch normalization
def batch_normalize(s, mean, variance):
    return np.matmul(np.diag(pow(variance + np.finfo(float).eps, -1/2)), (s - mean[:, np.newaxis]))

# Perform the forward pass of the network
def forward_pass(data, weights, bias):
    layer_output, s_output = [np.copy(data)], []

    for i in range(len(weights)):
        s_output.append(linear_scoring(layer_output[-1], weights[i], bias[i]))
        if i != len(weights)-1:
            layer_output.append(ReLU(s_output[-1]))
        else:
            layer_output.append(softmax(s_output[-1]))
    p = layer_output[-1]

    return layer_output, s_output, p

# Perform the forward pass of the network with batch normalization
def forward_pass_batchnorm(data, weights, bias, gamma=None, beta=None, mean=None, var=None):
    layer_output, s_output = [np.copy(data)], []
    s_hats, means, vars = [], [], []

    for i in range(len(weights)):
        s_output.append(linear_scoring(layer_output[-1], weights[i], bias[i]))

        # Batch normalization
        if i != len(weights)-1:
            # For training, no mean and variance is given
            if mean is None and var is None:
                means.append(np.mean(s_output[-1], axis=1))
                vars.append(np.var(s_output[-1], axis=1))
            # For testing, mean and variance is given
            else:
                means.append(mean[i])
                vars.append(var[i])
            s_hats.append(batch_normalize(s_output[-1], means[-1], vars[-1]))
            s_tilde = gamma[i] * s_hats[-1] + beta[i]
            layer_output.append(ReLU(s_tilde))
        # Last layer
        else:
            layer_output.append(softmax(s_output[-1]))
    p = layer_output[-1]

    return layer_output, s_output, s_hats, means, vars, p

# Mini-batch cross-entropy loss
def compute_loss(data, labels, weights, bias, gamma=None, beta=None, mean=None, var=None):
    if BATCH_NORM:
        p = forward_pass_batchnorm(data, weights, bias, gamma, beta, mean, var)[-1]
    else:
        p = forward_pass(data, weights, bias)[-1]
    return np.sum(-np.log(np.sum(labels * p, axis=0))) / data.shape[1]

# Add L2-regularization to the loss
def compute_cost(data, labels, weights, bias, lmbda, gamma=None, beta=None, mean=None, var=None):
    loss = compute_loss(data, labels, weights, bias, gamma, beta, mean, var)
    reg = lmbda * np.sum([np.sum(np.square(w)) for w in weights])

    return loss + reg

# Computes accuracy given model weights and bias
def compute_accuracy(data, labels, weights, bias, gamma=None, beta=None, mean=None, var=None):
    if BATCH_NORM:
        p = forward_pass_batchnorm(data, weights, bias, gamma, beta, mean, var)[-1]
    else:
        p = forward_pass(data, weights, bias)[-1]
    pred = np.argmax(p, axis=0)
    actual = np.argmax(labels, axis=0)

    return np.sum(actual == pred) / len(actual)

# Analytical gradient computations using mini-batch efficient method
def compute_gradients(data, labels, weights, lmbda, p):
    grad_W, grad_b = [], []
    g = -(labels - p)

    for i in reversed(range(len(weights))):    
        grad_W.append(np.matmul(g, data[i].T) / data[0].shape[1] + 2 * lmbda * weights[i])
        grad_b.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])

        g = np.matmul(weights[i].T, g)           # Update g for the previous layer
        diag = np.where(data[i] > 0, 1, 0)       # ReLu derivative, 1 if x > 0, 0 otherwise
        g = g * diag

    return grad_W[::-1], grad_b[::-1]

# Gradient computations using mini-batch efficient method with batch normalization
def compute_gradients_batchnorm(data, labels, weights, lmbda, p, s_output, s_hats, gamma, means, vars):
    grad_W, grad_b, grad_gamma, grad_beta = [], [], [], []
    g = -(labels - p)
    n = data[0].shape[1]

    for i in reversed(range(len(weights))):    
        if i != len(weights)-1:     # For l = k-1, k-2, ..., 1
            grad_gamma.append(np.sum(g * s_hats[i], axis=1)[:, np.newaxis] / n)
            grad_beta.append(np.sum(g, axis=1)[:, np.newaxis] / n)
            g = g * gamma[i]
            g = backpass_batchnorm(g, s_output[i], means[i], vars[i])

        grad_W.append(np.matmul(g, data[i].T) / n + 2 * lmbda * weights[i])
        grad_b.append(np.sum(g, axis=1)[:, np.newaxis] / n)

        g = np.matmul(weights[i].T, g)           # Update g for the previous layer
        diag = np.where(data[i] > 0, 1, 0)       # ReLu derivative, 1 if x > 0, 0 otherwise
        g = g * diag

    return grad_W[::-1], grad_b[::-1], grad_gamma[::-1], grad_beta[::-1]

# Updating G_batch using backpropagation for batch normalization
def backpass_batchnorm(g, s, mean, var):
    sigma1 = np.power(var + np.finfo(float).eps, -0.5).T[:, np.newaxis]
    sigma2 = np.power(var + np.finfo(float).eps, -1.5).T[:, np.newaxis]
    g1 = g * sigma1
    g2 = g * sigma2
    D = s - mean[:, np.newaxis]
    c = np.sum(g2 * D, axis=1)[:, np.newaxis]
    g_batch = g1 - (1 / g.shape[1]) * np.sum(g1, axis=1)[:, np.newaxis] - (1 / g.shape[1]) * D * c

    return g_batch

# Schedule for the cyclic learning rate
def cyclical_update(t, n_s, eta_min, eta_max):
    l = np.floor(t / (2 * n_s))         # Number of complete cycles elapsed
    if 2 * l * n_s <= t <= (2 * l + 1) * n_s:
        eta = eta_min + (t - 2 * l * n_s) / n_s * (eta_max - eta_min)
    elif (2 * l + 1) * n_s <= t <= 2 * (l + 1) * n_s:
        eta = eta_max - (t - (2 * l + 1) * n_s) / n_s * (eta_max - eta_min)
    return eta

# Plotting function for cost, loss and accuracy
def plot_graph(train, valid, title):
    plt.plot(range(len(train)), train, label="Training", color="blue")
    plt.plot(range(len(valid)), valid, label="Validation", color="red")
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
    plt.show()

# Saving sorted lambdas and accuracies to a file
def save_to_file(name, lmbdas, accuracies):
    sorted_list = sorted(zip(lmbdas, accuracies), key=lambda x: x[1], reverse=True)
    with open(name, 'w') as f:
        for item in sorted_list:
            f.write(f"{item}\n")
    
# Reading sorted lambdas and accuracies from a file
def read_from_file(name, n=2):
    lmbdas, accuracies = [], []
    with open(name, 'r') as f:
        for line in f:
            line = line.strip('()\n').split(', ')
            lmbda = float(line[0])
            acc = float(line[1])
            lmbdas.append(lmbda)
            accuracies.append(acc)
    return lmbdas[:n], accuracies[:n]

# Mini-batch gradient descent using specified model parameters
def train_model(train_data, train_label, valid_data, valid_label,
                weights, bias, gamma, beta, lmbda,
                n_batch, cycles, n_s, eta_min, eta_max):
    
    train_cost, valid_cost, train_loss, valid_loss, train_acc, valid_acc, etas = [], [], [], [], [], [], []
    batches = train_data.shape[1] / n_batch
    n_epochs = int(2 * n_s * cycles / batches)
    eta = eta_min

    print(f"Training model for {lmbda=}")
    print(f"{batches=}, {n_epochs=}, {cycles=}")
    for epoch in tqdm(range(n_epochs)):
        mean_runavg, var_runavg = [], []
        for j in range(int(train_data.shape[1] / n_batch)):
            start = j * n_batch
            end = (j + 1) * n_batch
            if BATCH_NORM:
                layer_output, s_output, s_hats, means, vars, p = forward_pass_batchnorm(train_data[:,start:end], weights, bias, gamma, beta, None, None)
                grad_W, grad_b, grad_gamma, grad_beta = compute_gradients_batchnorm(layer_output, train_label[:,start:end], weights, 
                                                                                    lmbda, p, s_output, s_hats, gamma, means, vars)
                # Update weights, bias, gamma, beta and eta
                weights = [weights[i] - eta * grad_W[i] for i in range(len(weights))]
                bias = [bias[i] - eta * grad_b[i] for i in range(len(bias))]
                gamma = [gamma[i] - eta * grad_gamma[i] for i in range(len(gamma))]
                beta = [beta[i] - eta * grad_beta[i] for i in range(len(beta))]
                eta = cyclical_update(epoch * batches + j, n_s, eta_min, eta_max)
                
                # Update running average of mean and variance
                if j == 0: 
                    mean_runavg = means
                    var_runavg = vars
                else:
                    mean_runavg = [(ALPHA * mean_runavg[i] + (1 - ALPHA) * means[i]) for i in range(len(means))]
                    var_runavg = [(ALPHA * var_runavg[i] + (1 - ALPHA) * vars[i]) for i in range(len(vars))]
            else:
                layer_outputs, _, p = forward_pass(train_data[:,start:end], weights, bias)
                grad_W, grad_b = compute_gradients(layer_outputs, 
                                                   train_label[:,start:end], 
                                                   weights, 
                                                   lmbda, 
                                                   p)
                weights = [weights[i] - eta * grad_W[i] for i in range(len(weights))]       # Update weights
                bias = [bias[i] - eta * grad_b[i] for i in range(len(bias))]                # Update bias
                eta = cyclical_update(epoch * batches + j, n_s, eta_min, eta_max)           # Update eta using cyclical learning rate

        if BATCH_NORM and SHOW_PLOTS:
            train_loss.append(compute_loss(train_data, train_label, weights, bias, gamma, beta, mean_runavg, var_runavg))
            valid_loss.append(compute_loss(valid_data, valid_label, weights, bias, gamma, beta, mean_runavg, var_runavg))
            train_cost.append(compute_cost(train_data, train_label, weights, bias, lmbda, gamma, beta, mean_runavg, var_runavg))
            valid_cost.append(compute_cost(valid_data, valid_label, weights, bias, lmbda, gamma, beta, mean_runavg, var_runavg))
            train_acc.append(compute_accuracy(train_data, train_label, weights, bias, gamma, beta, mean_runavg, var_runavg))
            valid_acc.append(compute_accuracy(valid_data, valid_label, weights, bias, gamma, beta, mean_runavg, var_runavg))
        elif SHOW_PLOTS:
            train_loss.append(compute_loss(train_data, train_label, weights, bias))
            valid_loss.append(compute_loss(valid_data, valid_label, weights, bias))
            train_cost.append(compute_cost(train_data, train_label, weights, bias, lmbda))
            valid_cost.append(compute_cost(valid_data, valid_label, weights, bias, lmbda))
            train_acc.append(compute_accuracy(train_data, train_label, weights, bias))
            valid_acc.append(compute_accuracy(valid_data, valid_label, weights, bias))
    
    if SHOW_PLOTS:
        plot_graph(train_cost, valid_cost, "Cost")
        plot_graph(train_loss, valid_loss, "Loss")
        plot_graph(train_acc, valid_acc, "Accuracy")

    if BATCH_NORM:
        return weights, bias, gamma, beta, mean_runavg, var_runavg
    else:
        return weights, bias

def main():
    np.random.seed(14)
    # Load and preprocess data
    train_data, train_label, valid_data, valid_label, test_data, test_label, _, k = load_and_preprocess(val_size=5000)
    
    # Initialize model parameters
    n_batch = 100
    eta_min = 1e-5
    eta_max = 1e-1
    cycles = 2
    lmbda = 0.005
    n_s = 5 * np.floor(train_data.shape[1] / n_batch)

    # Perform coarse search for lambda
    if COARSE:
        l_min, l_max = -5, -1
        lmbdas = 10 ** np.random.uniform(l_min, l_max, 10)
        l_accuracies = []
        for lmbda in lmbdas: 
            weights, bias, gamma, beta = initialize_weights(train_data)
            weights, bias, gamma, beta, final_mean, final_var = train_model(train_data, train_label, valid_data, valid_label, weights, bias, gamma, beta, lmbda, n_batch, cycles, n_s, eta_min, eta_max)
            l_accuracies.append(compute_accuracy(test_data, test_label, weights, bias, gamma, beta, None, None))
        save_to_file('coarse.txt', lmbdas, l_accuracies)

    # Perform fine search for lambda
    if FINE: 
        fine1, fine2 = np.log10(read_from_file('coarse.txt')[0])
        l_min = np.min([fine1, fine2]) - 0.25
        l_max = np.max([fine1, fine2]) + 0.25
        lmbdas = 10 ** np.random.uniform(l_min, l_max, 10)
        l_accuracies = []
        for lmbda in lmbdas: 
            weights, bias, gamma, beta = initialize_weights(train_data)
            weights, bias, gamma, beta, final_mean, final_var = train_model(train_data, train_label, valid_data, valid_label, weights, bias, gamma, beta, lmbda, n_batch, cycles, n_s, eta_min, eta_max)
            l_accuracies.append(compute_accuracy(test_data, test_label, weights, bias, gamma, beta, None, None))
        save_to_file('fine.txt', lmbdas, l_accuracies)

    # Training the final model
    if BATCH_NORM:
        weights, bias, gamma, beta = initialize_weights(train_data)
        weights, bias, gamma, beta, final_mean, final_var = train_model(train_data, train_label, valid_data, valid_label, weights, bias, gamma, beta, lmbda, n_batch, cycles, n_s, eta_min, eta_max)
        print("Accuracy on test set: {:.3f}%".format(compute_accuracy(test_data, test_label, weights, bias, gamma, beta, final_mean, final_var)*100))
    else:
        weights, bias, gamma, beta = initialize_weights(train_data)
        weights, bias = train_model(train_data, train_label, valid_data, valid_label, weights, bias, gamma, beta, lmbda, n_batch, cycles, n_s, eta_min, eta_max)
        print("Accuracy on test set: {:.3f}%".format(compute_accuracy(test_data, test_label, weights, bias, gamma, beta, None, None)*100))

if __name__ == "__main__":
    main()