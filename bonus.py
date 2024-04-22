import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

# Global constants
PATH = '../Datasets/cifar-10-batches-py/'
BATCH = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
HIDDEN_NODES = 100
ALL_DATA = True
SHOW_PLOTS = True
ADAM = True
AUGMENT = False
DROPOUT = 1       # Probability of keeping a neuron active

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

# Perform the forward pass of the network, including dropout
def forward_pass(data, weights, bias):
    s1 = linear_scoring(data, weights[0], bias[0])
    h = ReLU(s1)
    u = np.random.choice([0, 1], size=s1.shape, p=[1 - DROPOUT, DROPOUT])
    h = h * u
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

# Schedule for the cyclic learning rate
def cyclical_update(t, n_s, eta_min, eta_max):
    l = np.floor(t / (2 * n_s))         # Number of complete cycles elapsed
    if 2 * l * n_s <= t <= (2 * l + 1) * n_s:
        eta = eta_min + (t - 2 * l * n_s) / n_s * (eta_max - eta_min)
    elif (2 * l + 1) * n_s <= t <= 2 * (l + 1) * n_s:
        eta = eta_max - (t - (2 * l + 1) * n_s) / n_s * (eta_max - eta_min)
    return eta

# Flip images horizontally
def flip_images(data):
    aa = np.arange(0, 32)
    bb = np.arange(32, 0, -1)
    vv = np.tile(32 * aa, (32, 1)).T                                                    # Pixel data stored row by row in Python, not col by col
    bb_tiled = np.tile(bb, (32, 1))
    ind_flip = vv.flatten() + bb_tiled.flatten() - 1
    inds_flip = np.vstack((ind_flip, 1024 + ind_flip, 2048 + ind_flip))

    flipped_data = data[inds_flip, :]
    flipped_data = flipped_data.reshape(data.shape[0], -1)
    flip_idx = np.random.choice(range(data.shape[1]), int(0.5*data.shape[1]), replace=False)   # Horizontally flip 50% of images in training data
    data[:, flip_idx] = flipped_data[:, flip_idx]

    return data

# Translate images by tx in x-axis and ty in y-axis
def translate_images(data, tx, ty):
    if tx == 0 and ty == 0:
        return data
    
    aa = np.arange(0, 32)
    vv = np.tile(32 * aa, (32, 1)).T
    aa_tiled = np.tile(aa, (32, 1))
    ind_normal = vv + aa_tiled
    padding = np.copy(ind_normal)

    if tx == 0:
        if ty > 0:
            padding[ty:, :] = padding[:-ty, :]
        else:
            padding[:ty, :] = padding[-ty:, :]
    elif ty == 0:
        if tx > 0:
            padding[:, tx:] = padding[:, :-tx]
        else:
            padding[:, :tx] = padding[:, -tx:]
    else:
        if tx > 0 and ty > 0:
            padding[ty:, tx:] = padding[:-ty, :-tx]
        elif tx > 0 and ty < 0:
            padding[:ty, tx:] = padding[-ty:, :-tx]
        elif tx < 0 and ty > 0:
            padding[ty:, :tx] = padding[:-ty, -tx:]
        else:
            padding[:ty, :tx] = padding[-ty:, -tx:]

    padding_flat = padding.flatten()
    inds_translate = np.vstack((padding_flat, 1024 + padding_flat, 2048 + padding_flat))

    translated_data = data[inds_translate, :]
    translated_data = translated_data.reshape(data.shape[0], -1)
    translate_idx = np.random.choice(range(data.shape[1]), int(0.5*data.shape[1]), replace=False)   # Horizontally flip 50% of images in training data
    data[:, translate_idx] = translated_data[:, translate_idx]

    return data

# Adjust gradients using ADAM optimization
def adam_optimizer(grad_W, grad_b, m_W, v_W, m_b, v_b, beta_1=0.9, beta_2=0.999, epsilon=1e-8, eta=0.001):
    t = len(m_W)

    m_W.append([beta_1 * m_W[-1][i] + (1 - beta_1) * grad_W[i] for i in range(len(grad_W))])
    v_W.append([beta_2 * v_W[-1][i] + (1 - beta_2) * np.square(grad_W[i]) for i in range(len(grad_W))])
    m_b.append([beta_1 * m_b[-1][i] + (1 - beta_1) * grad_b[i] for i in range(len(grad_b))])
    v_b.append([beta_2 * v_b[-1][i] + (1 - beta_2) * np.square(grad_b[i]) for i in range(len(grad_b))])

    m_W_hat = [m_W[-1][i] / (1 - beta_1 ** t) for i in range(len(m_W[-1]))]
    v_W_hat = [v_W[-1][i] / (1 - beta_2 ** t) for i in range(len(v_W[-1]))]
    m_b_hat = [m_b[-1][i] / (1 - beta_1 ** t) for i in range(len(m_b[-1]))]
    v_b_hat = [v_b[-1][i] / (1 - beta_2 ** t) for i in range(len(v_b[-1]))]

    delta_W = [eta * m_W_hat[i] / (np.sqrt(v_W_hat[i]) + epsilon) for i in range(len(m_W_hat))]
    delta_b = [eta * m_b_hat[i] / (np.sqrt(v_b_hat[i]) + epsilon) for i in range(len(m_b_hat))]

    return m_W, v_W, m_b, v_b, delta_W, delta_b

# Plotting function for cost, loss and accuracy
def plot_graph(train, valid, title):
    plt.plot(range(len(train)), train, label="Training", color="blue")
    plt.plot(range(len(valid)), valid, label="Validation", color="red")
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
    plt.show()
    
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

        if AUGMENT and epoch % 2 == 0: 
            print("Augmenting data...")
            tx, ty = np.random.randint(-2, 3, 2)                        # Randomly select translation values
            augmented_data = flip_images(train_data)                    # Horizontally flip 50% of images in training data
            augmented_data = translate_images(augmented_data, tx, ty)   # Translate 50% of images by tx and ty

        for j in range(int(train_data.shape[1] / n_batch)):
            start = j * n_batch
            end = (j + 1) * n_batch
            if AUGMENT:
                grad_W, grad_b = compute_gradients(augmented_data[:,start:end],
                                                   train_label[:,start:end],
                                                   weights,
                                                   bias,
                                                   lmbda)
            else:
                grad_W, grad_b = compute_gradients(train_data[:,start:end], 
                                                   train_label[:,start:end],
                                                   weights, 
                                                   bias,
                                                   lmbda)
            if ADAM: 
                # Initialize ADAM optimizer parameters
                if epoch == 0 and j == 0:
                    m_W = [[np.zeros(w.shape) for w in grad_W]]
                    v_W = [[np.zeros(w.shape) for w in grad_W]]
                    m_b = [[np.zeros(b.shape) for b in grad_b]]
                    v_b = [[np.zeros(b.shape) for b in grad_b]]

                # Update weights and bias using ADAM optimizer
                m_W, v_W, m_b, v_b, delta_W, delta_b = adam_optimizer(grad_W, grad_b, m_W, v_W, m_b, v_b)
                weights = [weights[i] - delta_W[i] for i in range(len(weights))]            # Update weights
                bias = [bias[i] - delta_b[i] for i in range(len(bias))]                     # Update bias
            else:
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
        if not ADAM:
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
    cycles = 3
    #best_lmbda = read_from_file('fine.txt')[0][0]
    lmbda = 0.005
    n_s = 4 * np.floor(train_data.shape[1] / n_batch)

    # Training the final model
    weights, bias, _ = train_model(train_data, train_label, valid_data, valid_label, weights, bias, lmbda, n_batch, cycles, n_s, eta_min, eta_max)
    print("Accuracy on test set: {:.3f}%".format(compute_accuracy(test_data, test_label, weights, bias)*100))

if __name__ == "__main__":
    main()