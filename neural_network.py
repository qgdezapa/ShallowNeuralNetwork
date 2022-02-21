import numpy as np
from PIL import Image
import copy
import matplotlib.pyplot as plt

# Global variables
NUM_PX = 64
NUM_IMG = 500
TOTAL_NUM_IMG = 12500
CLASSES = {0: "dog", 1: "cat"}


def load_image(image_link):
    image = Image.open(image_link).resize((NUM_PX, NUM_PX))
    image_data = np.array(image)
    return image_data


def load_training_data():
    num_train_cat_images = NUM_IMG
    num_train_dog_images = NUM_IMG

    train_data_x = np.zeros((NUM_IMG * 2, NUM_PX, NUM_PX, 3))
    train_data_y = np.zeros((1, NUM_IMG * 2))

    for i in range(num_train_cat_images):
        train_data_x[i] = load_image(f'./images/train/cat.{i}.jpg')
        train_data_y[0, i] = 1

    for i in range(num_train_dog_images):
        train_data_x[i + num_train_cat_images] = load_image(f'./images/train/dog.{i}.jpg')
        # train_data_y for dogs are already zero

    return train_data_x, train_data_y


def transform_data(x):
    x = x.reshape(x.shape[0], -1).T
    return standardize(x)


def standardize(x):
    return x / 255


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1), dtype=float)
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1), dtype=float)

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


def partition(x, y, percentage):
    m = x.shape[1]

    if m % 10 == 0:
        return np.concatenate((x[:, :int(m * percentage) // 2], x[:, NUM_IMG:NUM_IMG + int(m * percentage) // 2]),
                              axis=1), \
               np.concatenate((y[0, :int(m * percentage) // 2], y[0, NUM_IMG:NUM_IMG + int(m * percentage) // 2]),
                              axis=0), \
               np.concatenate((x[:, int(m * percentage) // 2:NUM_IMG], x[:, NUM_IMG + int(m * percentage) // 2:]),
                              axis=1), \
               np.concatenate((y[0, int(m * percentage) // 2:NUM_IMG], y[0, NUM_IMG + int(m * percentage) // 2:]),
                              axis=0)

    return np.concatenate((x[:, :int(m * percentage) // 2], x[:, NUM_IMG:NUM_IMG + int(m * percentage) // 2]), axis=1), \
           np.concatenate((y[0, :int(m * percentage) // 2], y[0, NUM_IMG:NUM_IMG + int(m * percentage) // 2]), axis=0), \
           np.concatenate(
               (x[:, (int(m * percentage) + 1) // 2:NUM_IMG], x[:, NUM_IMG + (int(m * percentage) + 1) // 2:]), axis=1), \
           np.concatenate(
               (y[0, (int(m * percentage) + 1) // 2:NUM_IMG], y[0, NUM_IMG + (int(m * percentage) + 1) // 2:]), axis=0)


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
    }

    return cache


def compute_cost(A2, Y):
    m = Y.shape[1]

    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m

    return float(np.squeeze(cost))


# here
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = W2.T * dZ2 * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {
        "dZ1": dZ1,
        "dW1": dW1,
        "db1": db1,
        "dZ2": dZ2,
        "dW2": dW2,
        "db2": db2,
    }

    return gradients


def update_parameters(parameters, gradients, learning_rate):
    W1 = copy.deepcopy(parameters["W1"])
    b1 = parameters["b1"]
    W2 = copy.deepcopy(parameters["W2"])
    b2 = parameters["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def neural_network_model(X, Y, n_h, iterations, learning_rate=0.5, print_cost=False):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    for i in range(iterations):
        parameters = initialize_parameters(n_x, n_h, n_y)
        cache = forward_propagation(X, parameters)
        cost = compute_cost(cache["A2"], Y)
        gradients = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, gradients, learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(X, parameters):
    cache = forward_propagation(X, parameters)

    predictions = (cache['A2'] > 0.5)

    return predictions


data_x, data_y = load_training_data()
data_x = transform_data(data_x)

train_x, train_y, test_x, test_y = partition(data_x, data_y, 0.7)
train_y = train_y.reshape((1, train_y.shape[0]))
test_y = test_y.reshape((1, test_y.shape[0]))

parameters = neural_network_model(train_x, train_y, n_h=3, iterations=10000, learning_rate=1.2, print_cost=True)
