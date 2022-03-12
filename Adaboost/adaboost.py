"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finishing the training return the weights of the samples in the last iteration.
        """
        D = np.ones(y.size) / y.size
        epsilon = np.array([None] * self.T)
        f = lambda x: 0.5*np.log((1/x - 1))
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            pred = self.h[t].predict(X)
            v = (y != pred).astype(int)
            epsilon[t] = np.sum(np.multiply(D, v))
            self.w[t] = f(epsilon[t])
            D *= np.exp((-1) * self.w[t] * y * pred)
            D /= np.sum(D)

        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        num_samples = X.shape[0]
        y_hat = np.zeros(num_samples)
        for t in range(0, max_t):
            y_hat += (self.h[t].predict(X) * self.w[t])

        return np.sign(y_hat)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predicting only with max_t weak learners (float)
        """
        #err = np.sum((y != self.predict(X, max_t)).astype(int)) / y.size
        #return err
        y_hat = self.predict(X, max_t)
        return sum(abs(y - y_hat)/2)/y.size

