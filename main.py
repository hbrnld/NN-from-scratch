import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

# Global constants
PATH = '../Datasets/cifar-10-batches-py/'
BATCH = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
HIDDEN_NODES = 50
ALL_DATA = False
SHOW_PLOTS = True
CHECK_GRADS = False
COARSE = False
FINE = False

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
    if ALL_DATA:
        # Load all batches for training data
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

    else:
        # Load one batch for training data
        file = unpickle(PATH+'data_batch_1')
        train_data = file['data']
        train_label = file['labels']
        
        file = unpickle(PATH+'data_batch_2')
        valid_data = file['data']
        valid_label = file['labels']

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

# Initializing weights and bias, using Xavier initialization
def initialize_weights(data, k):
    weights, bias = [], []
    # First layer (d inputs -> m hidden nodes)
    weights.append(np.random.normal(0, 1/np.sqrt(data.shape[0]), (HIDDEN_NODES, data.shape[0])))    # m x d
    bias.append(np.zeros((HIDDEN_NODES, 1)))                                                        # m x 1
    # Second layer (m hidden nodes -> k outputs)
    weights.append(np.random.normal(0, 1/np.sqrt(HIDDEN_NODES), (k, HIDDEN_NODES)))                 # k x m
    bias.append(np.zeros((k, 1)))                                                                   # k x 1
                
    return weights, bias

# Matrix multiplication of inputs and weights, adding bias
def linear_scoring(data, weight, bias):
    return np.matmul(weight, data) + bias       # Wx + b

# Definition of the rectified linear unit
def ReLU(x):
    return np.maximum(0, x)                     # max(0, x)

# Definition of the Softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Perform the forward pass of the network
def forward_pass(data, weights, bias):
    s1 = linear_scoring(data, weights[0], bias[0])
    h = ReLU(s1)
    s2 = linear_scoring(h, weights[1], bias[1])
    p = softmax(s2)
    
    return s1, h, s2, p

# Mini-batch cross-entropy loss
def compute_loss(data, labels, weights, bias):
    _, _, _, p = forward_pass(data, weights, bias)
    loss = np.sum(-np.log(np.sum(labels * p, axis=0))) / data.shape[1]
    
    return loss

# Add L2-regularization to the loss
def compute_cost(data, labels, weights, bias, lmbda):
    loss = compute_loss(data, labels, weights, bias)
    reg = lmbda * np.sum([np.sum(np.square(w)) for w in weights])

    return loss + reg

# Computes accuracy given model weights and bias
def compute_accuracy(data, labels, weights, bias):
    _, _, _, p = forward_pass(data, weights, bias)
    pred = np.argmax(p, axis=0)
    actual = np.argmax(labels, axis=0)

    return np.sum(actual == pred) / len(actual)

# Analytical gradient computations using mini-batch efficient method
def compute_gradients(data, labels, weights, bias, lmbda):
    grad_W, grad_b = [], []
    s1, h, s2, p = forward_pass(data, weights, bias)
    
    g = -(labels - p)
    grad_W.append(np.matmul(g, h.T) / data.shape[1] + 2 * lmbda * weights[1])
    grad_b.append(np.sum(g, axis=1)[:, np.newaxis] / data.shape[1])
    
    g = np.matmul(weights[1].T, g)      # Update g for the previous layer
    diag = np.where(s1 > 0, 1, 0)       # ReLu derivative, 1 if s1 > 0, 0 otherwise
    g = g * diag
    grad_W.append(np.matmul(g, data.T) / data.shape[1] + 2 * lmbda * weights[0])
    grad_b.append(np.sum(g, axis=1)[:, np.newaxis] / data.shape[1])

    return grad_W[::-1], grad_b[::-1]

# Numerical computations of the gradients, using centered difference formula
def compute_grads_num_slow(data, labels, weights, bias, lmbda, h):
    grad_weights = []
    grad_bias = []

    for j in range(len(bias)):
        grad_bias.append(np.zeros(len(bias[j])))
        for i in range(len(bias[j])):
            b_try = []
            [b_try.append(np.copy(x)) for x in bias]
            b_try[j][i] = b_try[j][i] - h
            c1 = compute_cost(data, labels, weights, b_try, lmbda)
            b_try = []
            [b_try.append(np.copy(x)) for x in bias]
            b_try[j][i] = b_try[j][i] + h
            c2 = compute_cost(data, labels, weights, b_try, lmbda)
            grad_bias[j][i] = (c2 - c1) / (2 * h)

    for j in range(len(weights)):
        grad_weights.append(np.zeros(weights[j].shape))
        for i in tqdm(range(grad_weights[-1].shape[0])):
            for k in range(grad_weights[-1].shape[1]):
                w_try = []
                [w_try.append(np.copy(x)) for x in weights]
                w_try[j][i, k] = w_try[j][i, k] - h
                c1 = compute_cost(data, labels, w_try, bias, lmbda)
                w_try = []
                [w_try.append(np.copy(x)) for x in weights]
                w_try[j][i, k] = w_try[j][i, k] + h
                c2 = compute_cost(data, labels, w_try, bias, lmbda)
                grad_weights[j][i, k] = (c2 - c1) / (2 * h)

    return grad_weights, grad_bias

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
def train_model(train_data, train_label, valid_data, valid_label, weights, bias, lmbda, n_batch, cycles, n_s, eta_min, eta_max):
    train_cost, valid_cost, train_loss, valid_loss, train_acc, valid_acc, etas = [], [], [], [], [], [], []
    batches = train_data.shape[1] / n_batch
    n_epochs = int(2 * n_s * cycles / batches)
    eta = eta_min

    print(f"Training model for lambda {lmbda}")
    print(f"{batches=}, {n_epochs=}, {cycles=}")
    for epoch in tqdm(range(n_epochs)):
        for j in range(int(train_data.shape[1] / n_batch)):
            start = j * n_batch
            end = (j + 1) * n_batch
            grad_W, grad_b = compute_gradients(train_data[:,start:end], 
                                               train_label[:,start:end],
                                               weights, 
                                               bias,
                                               lmbda)
            weights = [weights[i] - eta * grad_W[i] for i in range(len(weights))]       # Update weights
            bias = [bias[i] - eta * grad_b[i] for i in range(len(bias))]                # Update bias
            eta = cyclical_update(epoch * batches + j, n_s, eta_min, eta_max)           # Update eta using cyclical learning rate
            etas.append(eta)

        train_cost.append(compute_cost(train_data, train_label, weights, bias, lmbda))
        valid_cost.append(compute_cost(valid_data, valid_label, weights, bias, lmbda))
        train_loss.append(compute_loss(train_data, train_label, weights, bias))
        valid_loss.append(compute_loss(valid_data, valid_label, weights, bias))
        train_acc.append(compute_accuracy(train_data, train_label, weights, bias))
        valid_acc.append(compute_accuracy(valid_data, valid_label, weights, bias))
    
    if SHOW_PLOTS:
        plot_graph(train_cost, valid_cost, "Cost")
        plot_graph(train_loss, valid_loss, "Loss")
        plot_graph(train_acc, valid_acc, "Accuracy")
        plot_graph(etas, etas, "Etas")

    return weights, bias, compute_accuracy(valid_data, valid_label, weights, bias)

def main():
    # Load and preprocess data
    train_data, train_label, valid_data, valid_label, test_data, test_label, _, k = load_and_preprocess(val_size=1000)
    
    # Initialize model parameters
    np.random.seed(8)
    weights, bias = initialize_weights(train_data, k)
    n_batch = 100                                           # Batch size: Larger batch size -> fewer batches -> faster training
    eta_min = 1e-5
    eta_max = 1e-1
    lmbda = 0.01
    cycles = 3
    n_s = 2 * np.floor(train_data.shape[1] / n_batch)       # Step size for changing the learning rate, higher -> more cycles & epochs

    # weights, bias, _ = train_model(train_data, train_label, valid_data, valid_label, weights, bias, lmbda, n_batch, cycles, n_s, eta_min, eta_max)
    # print("Accuracy on test set: {:.3f}%".format(compute_accuracy(test_data, test_label, weights, bias)*100))

    # Comparing analytical and numerical gradient computations
    if CHECK_GRADS: 
        grad_W, grad_b = compute_gradients(train_data[:,:10], train_label[:,:10], weights,  bias, lmbda)
        grad_W2, grad_b2 = compute_grads_num_slow(train_data[:,:10], train_label[:,:10], weights, bias, lmbda, h=1e-5)

        print([np.mean(abs(grad_W[i] - grad_W2[i])) for i in range(len(grad_W))])
        print([np.mean(abs(grad_b[i] - grad_b2[i])) for i in range(len(grad_b))])

    # Perform coarse search for lambda
    if COARSE:
        l_min, l_max = -5, -1
        lmbdas = 10 ** np.random.uniform(l_min, l_max, 8)
        l_accuracies = []
        for lmbda in lmbdas: 
            _, _, acc = train_model(train_data, train_label, valid_data, valid_label, weights, bias, lmbda, n_batch, cycles, n_s, eta_min, eta_max)    
            l_accuracies.append(acc)
        save_to_file('coarse.txt', lmbdas, l_accuracies)
    
    # Perform fine search for lambda    
    if FINE: 
        fine1, fine2 = np.log10(read_from_file('coarse.txt')[0])
        l_min = np.min([fine1, fine2]) - 0.25
        l_max = np.max([fine1, fine2]) + 0.25
        lmbdas = 10 ** np.random.uniform(l_min, l_max, 8)
        l_accuracies = []
        for lmbda in lmbdas: 
            _, _, acc = train_model(train_data, train_label, valid_data, valid_label, weights, bias, lmbda, n_batch, cycles, n_s, eta_min, eta_max)    
            l_accuracies.append(acc)
        save_to_file('fine.txt', lmbdas, l_accuracies)

    # Overwriting model parameters
    best_lmbda = read_from_file('fine.txt')[0][0]
    cycles = 3
    n_batch = 100
    n_s = 4 * np.floor(train_data.shape[1] / n_batch)
    print(f"{n_s=}")

    # Training the final model
    weights, bias, _ = train_model(train_data, train_label, valid_data, valid_label, weights, bias, best_lmbda, n_batch, cycles, n_s, eta_min, eta_max)
    print("Accuracy on test set: {:.3f}%".format(compute_accuracy(test_data, test_label, weights, bias)*100))

if __name__ == "__main__":
    main()