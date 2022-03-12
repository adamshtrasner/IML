#########################
#      Exercise 3       #
# Name: Adam Shtrasner  #
# ID: 208837260         #
#########################


import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class Perceptron:

    def __init__(self):
        self.w = None

    def fit(self, X, y):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.w = np.zeros(X.shape[1])

        while np.any(np.sign(X @ self.w) - y):
            i = np.nonzero(np.sign(X @ self.w) - y)[0][0]
            self.w += y[i] * X[i]

    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.sign(X @ self.w)

    def score(self, X, y):
        return scr(self, X, y)


class LDA:

    def __init__(self):
        self.mu_y = None
        self.sigma = None
        self.sigma_inv = None
        self.prob_y = None

    def fit(self, X, y):
        mu_y_plus1 = X[y == 1].mean(axis=0)
        mu_y_minus1 = X[y == -1].mean(axis=0)
        self.sigma = ((X[y == 1] - mu_y_plus1).T @ (X[y == 1] - mu_y_plus1) +
                      (X[y == -1] - mu_y_minus1).T @ (X[y == -1] - mu_y_minus1)) / y.size
        self.sigma_inv = np.linalg.inv(self.sigma)
        prob_y_plus1 = (y == 1).mean()
        prob_y_minus1 = (y == -1).mean()

        self.mu_y = np.array([mu_y_plus1, mu_y_minus1]).T
        self.prob_y = np.array([prob_y_plus1, prob_y_minus1])

    def predict(self, X):
        delta = X @ self.sigma_inv @ self.mu_y - \
                    0.5 * (np.diag(self.mu_y.T @ self.sigma_inv @ self.mu_y)) + \
                    np.log(self.prob_y)
        return -2 * np.argmax(delta, axis=1) + 1

    def score(self, X, y):
        return scr(self, X, y)


class SVM(SVC):
    def __init__(self):
        SVC.__init__(self, C=1e10, kernel="linear")

    def score(self, X, y):
        return scr(self, X, y)


class Logistic(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self, solver="liblinear")

    def score(self, X, y):
        return scr(self, X, y)


class DecisionTree(DecisionTreeClassifier):
    def __init__(self):
        DecisionTreeClassifier.__init__(self, max_depth=1)

    def score(self, X, y):
        return scr(self, X, y)


def scr(model, X, y):
    """
    This function is the score function which fits to each one of the models below.
    """

    TP = np.sum(np.logical_and(y == 1, model.predict(X) == 1))
    FP = np.sum(np.logical_and(y == -1, model.predict(X) == 1))
    TN = np.sum(np.logical_and(y == -1, model.predict(X) == -1))
    FN = np.sum(np.logical_and(y == 1, model.predict(X) == -1))

    num_samples = len(y)
    error = max(1, FP + FN) / num_samples
    accuracy = max(1, TP + TN) / num_samples
    FPR = FP / (y == -1).sum()
    TPR = TP / (y == 1).sum()
    precision = TP / max(1, TP + FP)
    specificity = TN / (y == -1).sum()

    dc = {"num_samples": num_samples, "error": error, "accuracy": accuracy,
          "FPR": FPR, "TPR": TPR, "precision": precision, "specificity": specificity}

    return dc