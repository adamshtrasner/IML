#########################
#      Exercise 3       #
# Name: Adam Shtrasner  #
# ID: 208837260         #
#########################


import numpy as np
import matplotlib.pyplot as plt


train_data = np.loadtxt("mnist_train.csv",
                        delimiter=",")
test_data = np.loadtxt("mnist_test.csv",
                       delimiter=",")


train_images = np.logical_or((train_data[:, 0] == 0), (train_data[:, 0] == 1))
test_images = np.logical_or((test_data[:, 0] == 0), (test_data[:, 0] == 1))
train_data = train_data[train_images]
test_data = test_data[test_images]

x_train, y_train = train_data.T[1:].T, train_data[:, 0]
x_test, y_test = test_data.T[1:].T, test_data[:, 0]


def draw(X, y):
    ids = np.random.randint(y.size, size=3)
    samples_y_zero = y[ids]
    while not np.all(samples_y_zero == 0):
        ids = np.random.randint(y.size, size=3)
        samples_y_zero = y[ids]

    js = np.random.randint(y.size, size=3)
    samples_y_one = y[js]
    while not np.all(samples_y_one == 1):
        js = np.random.randint(y.size, size=3)
        samples_y_one = y[js]

    samples_x_zero = X[ids, :]
    samples_x_one = X[js, :]

    samples_x_zero = np.reshape(samples_x_zero, (3, 28, 28))
    samples_x_one = np.reshape(samples_x_one, (3, 28, 28))

    for i in range(3):
        plt.imshow(samples_x_zero[i], cmap='gray', vmin=0, vmax=255)
        plt.show()

    for j in range(3):
        plt.imshow(samples_x_one[j], cmap='gray', vmin=0, vmax=255)
        plt.show()


def rearrange_data(X):
    return np.reshape(X, (-1, 784))


draw(x_train, y_train)
