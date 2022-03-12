# Introduction to Machine Learning (67577)
# Exercise 1, Q11-Q16
# Full Name: Adam Shtrasner, ID: 208837260

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

####################
# Helper functions #
####################


# from 3d_gaussian: #
def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q

# My own helpers: #


def plot_upper_bound(epsilon, m, q16c=False, mean=0):
    cheb_bound = 1 / (4 * m * epsilon ** 2)
    hoef_bound = 2 * np.exp(-2 * m * epsilon ** 2)
    plt.plot(m, np.clip(cheb_bound, 0, 1), label="Chebyshev's upper bound")
    plt.plot(m, np.clip(hoef_bound, 0, 1), label="Hoeffding's upper bound")
    if q16c:
        plt.plot(m, np.mean(np.abs(mean - 0.25) >= epsilon, axis=0), label="Percentage of sequences that"
                                                                           " satisfy the condition")
    plt.legend(loc="upper center")
    plt.title("Epsilon = " + str(epsilon))
    plt.xlabel("Number of tosses")
    plt.ylabel("Upper Bound")
    plt.show()

#########################
# Solutions start here: #
#########################

###############
# Question 11 #
###############
mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T
plot_3d(x_y_z)

###############
# Question 12 #
###############
# TODO: Add an explanation
S = np.diag((0.1, 0.5, 2))
x_y_z_scaled = S @ x_y_z
plot_3d(x_y_z_scaled)
print("Question 12: We can conclude from the graph that the scaled data is still centered around the same"
      "area as before, but it shrank/stretched.")

cov_scaled_mat = np.cov(x_y_z_scaled)
print("The covariance matrix after scaling is: \n", cov_scaled_mat)

###############
# Question 13 #
###############
U = get_orthogonal_matrix(3)
x_y_z_scaled_by_orth = U @ x_y_z_scaled
plot_3d(x_y_z_scaled_by_orth)
print("Question 13: We can conclude from the graph that by multiplying the scaled data with the"
      "orthogonal matrix - the data is being rotated, but its size remains the same.")

new_cov_scaled_mat = np.cov(x_y_z_scaled_by_orth)
print("The covariance matrix after rotation is: \n", new_cov_scaled_mat)

###############
# Question 14 #
###############
x_y_data_projection = x_y_z[:2]
plot_2d(x_y_data_projection)
print("Question 14: The points are centered around an area and are uncorrelated.")

###############
# Question 15 #
###############
x_y_points_projection = x_y_z[:, (-.4 < x_y_z[2, :]) & (x_y_z[2, :] < .1)]
plot_2d(x_y_points_projection)
print("Question 15: The graph almost looks the same as the previous question, but with less points")

###############
# Question 16 #
###############

#######
# (a) #
#######
data = np.random.binomial(1, 0.25, (10000, 1000))
m = np.arange(1, 1001)

mean = np.cumsum(data[:5], axis=1) / m
plt.plot(mean.T)
plt.xlabel("Number of Tosses")
plt.ylabel("Estimation of accumulative mean")
plt.title("Estimation of accumulative mean as a function of number of tosses")
plt.show()

#######
# (b) #
#######
epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
for e in epsilon:
    plot_upper_bound(e, m)

#######
# (c) #
#######
mean = np.cumsum(data, axis=1) / m
for e in epsilon:
    plot_upper_bound(e, m, True, mean)
