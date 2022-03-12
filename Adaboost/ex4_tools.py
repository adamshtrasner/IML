"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

This module provides some useful tools for Ex4.


"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import product
from matplotlib.pyplot import imread
import os
from sklearn.model_selection import train_test_split
from adaboost import *


def find_threshold(D, X, y, sign, j):
    """
    Finds the best threshold.
    D =  distribution
    S = (X, y) the data
    """
    # sort the data so that x1 <= x2 <= ... <= xm
    sort_idx = np.argsort(X[:, j])
    X, y, D = X[sort_idx], y[sort_idx], D[sort_idx]

    thetas = np.concatenate([[-np.inf], (X[1:, j] + X[:-1, j]) / 2, [np.inf]])
    minimal_theta_loss = np.sum(D[y == sign])  # loss of the smallest possible theta
    losses = np.append(minimal_theta_loss, minimal_theta_loss - np.cumsum(D * (y * sign)))
    min_loss_idx = np.argmin(losses)
    return losses[min_loss_idx], thetas[min_loss_idx]


class DecisionStump(object):
    """
    Decision stump classifier for 2D samples
    """

    def __init__(self, D, X, y):
        self.theta = 0
        self.j = 0
        self.sign = 0
        self.train(D, X, y)

    def train(self, D, X, y):
        """
        Train the classifier over the sample (X,y) w.r.t. the weights D over X
        Parameters
        ----------
        D : weights over the sample
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        """
        loss_star, theta_star = np.inf, np.inf
        for sign, j in product([-1, 1], range(X.shape[1])):
            loss, theta = find_threshold(D, X, y, sign, j)
            if loss < loss_star:
                self.sign, self.theta, self.j = sign, theta, j
                loss_star = loss

    def predict(self, X):
        """
        Parameters
        ----------
        X : shape=(num_samples, num_features)
        Returns
        -------
        y_hat : a prediction vector for X shape=(num_samples)
        """
        y_hat = self.sign * ((X[:, self.j] >= self.theta) * 2 - 1)
        return y_hat


def decision_boundaries(classifier, X, y, num_classifiers=1, weights=None):
    """
    Plot the decision boundaries of a binary classfiers over X \subseteq R^2

    Parameters
    ----------
    classifier : a binary classifier, implements classifier.predict(X)
    X : samples, shape=(num_samples, 2)
    y : labels, shape=(num_samples)
    title_str : optional title
    weights : weights for plotting X
    """
    cm = ListedColormap(['#AAAAFF', '#FFAAAA'])
    cm_bright = ListedColormap(['#0000FF', '#FF0000'])
    h = .003  # step size in the mesh
    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
    y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()], num_classifiers)
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cm)
    # Plot also the training points
    if weights is not None:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=weights, cmap=cm_bright)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks([])
    plt.yticks([])
    plt.title(f'num classifiers = {num_classifiers}')
    plt.draw()


def generate_data(num_samples, noise_ratio):
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X = np.random.rand(num_samples, 2) * 2 - 1
    radius = 0.5 ** 2
    in_circle = np.sum(X ** 2, axis=1) < radius
    y = np.ones(num_samples)
    y[in_circle] = -1
    y[np.random.choice(num_samples, int(noise_ratio * num_samples))] *= -1

    return X, y


def questions_13_14_15_16(noise_ratio=0):
    #######
    # Q13 #
    #######

    X_train, y_train = generate_data(5000, noise_ratio)
    adaboost = AdaBoost(DecisionStump, 500)
    Dt = adaboost.train(X_train, y_train)

    X_test, y_test = generate_data(200, noise_ratio)
    T = np.arange(1, 501)
    train_error = list()
    test_error = list()
    for t in T:
        train_error.append(adaboost.error(X_train, y_train, t))
        test_error.append(adaboost.error(X_test, y_test, t))

    plt.plot(T, train_error, label="Training error")
    plt.plot(T, test_error, label="Test error")
    plt.xlabel('T')
    plt.ylabel("Error rate")
    plt.legend()
    plt.show()

    #######
    # Q14 #
    #######

    for i, t in enumerate([5, 10, 50, 100, 200, 500]):
        plt.subplot(2, 3, i + 1)
        decision_boundaries(adaboost, X_train, y_train, t)

    plt.show()

    #######
    # Q15 #
    #######

    T_hat = np.argmin(test_error)
    print("===Question 15:===")
    print("The T that minimizes the test error is: ", T_hat + 1)
    print("Its test error is: ", test_error[T_hat])
    decision_boundaries(adaboost, X_test, y_test, T_hat + 1)
    plt.show()

    #######
    # Q16 #
    #######

    # Before normalizing the weights:
    decision_boundaries(adaboost, X_train, y_train, 500, Dt)
    plt.show()

    Dt = Dt / np.max(Dt) * 10
    # After normalizing the weights:
    decision_boundaries(adaboost, X_train, y_train, 500, Dt)
    plt.show()


if __name__ == "__main__":
    # questions_13_14_15_16()

    #######
    # Q17 #
    #######

    # questions_13_14_15_16(0.01)
    # questions_13_14_15_16(0.4)

    Dt = np.array([9.6179 * np.exp(-1),
                   2.787 * np.exp(-2),
                   6.7 * np.exp(-4),
                   5.82 * np.exp(-3),
                   3.85 * np.exp(-3)])
    y = np.array([1, 1, -1, -1, 1])
    ht = np.array([1, -1, -1, -1, -1])

    eps = np.sum((np.abs(ht - y) / 2) * Dt)
    f = lambda x: 0.5 * np.log((1 / x - 1))
    w = f(eps)

    Dt *= np.exp((-1) * w * y * ht)
    Dt /= np.sum(Dt)

    print(Dt)